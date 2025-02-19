
#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "fem/fe_coll.hpp"
#include "fem/qspace.hpp"
#include "general/forall.hpp"
#include "mfem.hpp"
#include "mpi.h"

namespace mfem
{

inline real_t sigmoid(const real_t x)
{
   return x > 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

inline real_t sigmoid(const real_t x, const real_t l, const real_t u)
{
   return l + (u - l) * sigmoid(x);
}

class MappedGridFunctionCoefficient : public GridFunctionCoefficient
{
protected:
   std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
   F;
public:
   MappedGridFunctionCoefficient(GridFunction &gf,
                                 std::function<real_t(const real_t x, ElementTransformation &T, const IntegrationPoint &ip)>
                                 F)
      : GridFunctionCoefficient(&gf), F(F) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return F(GridFunctionCoefficient::Eval(T, ip), T, ip);
   }
};

/// @brief Hessian of Differentiable Objective
/// @details At: x -> H_x(F)
///          Mult: x -> y = H_x(F)*y
///          InvMult: x -> y = H_x(F)^{-1}*y
class ObjectiveHessian : public Operator
{
public:
   using Operator::Operator;
   virtual void At(const Vector &x) = 0;
   virtual void InvMult(const Vector &x, Vector &y) const
   {
      MFEM_ABORT("Not Implemented");
   }
};

/// @brief F(u)
/// @details Eval: x -> F(u)
///          Mult: x -> y = grad F(x)
///          GetGradient: x -> y = H_x(F)
class DifferentiableObjective : public Operator
{
protected:
   std::unique_ptr<ObjectiveHessian> hessian;
   MPI_Comm comm;
public:
   using Operator::Operator;
   virtual real_t Eval(const Vector &x) = 0;
   Operator & GetGradient(const Vector &x) const override
   {
      MFEM_ASSERT(hessian, "Hessian is not set");
      hessian->At(x);
      return *hessian;
   }
   MPI_Comm GetComm() { return comm; }
};

class L2Obj : public DifferentiableObjective
{
private:
protected:
   Coefficient &targ_cf;
   Vector proj_targ;
   Vector diff_targ;
   std::unique_ptr<ParGridFunction> targ_gf;
   std::unique_ptr<ParBilinearForm> mass_form;
   std::unique_ptr<QuadratureFunction> targ_qf;
   MPI_Comm comm;
public:

private:
protected:
public:
   L2Obj(ParFiniteElementSpace &fes, Coefficient &targ_cf): targ_cf(targ_cf)
   {
      proj_targ.SetSize(fes.GetVSize());
      targ_gf.reset(new ParGridFunction(&fes, proj_targ.GetData()));
      targ_gf->ProjectCoefficient(targ_cf);
      mass_form.reset(new ParBilinearForm(&fes));
      comm = fes.GetComm();
      mass_form->AddDomainIntegrator(new MassIntegrator());
      mass_form->Assemble();
   }

   L2Obj(QuadratureSpaceBase &qspace, Coefficient &targ_cf): targ_cf(targ_cf)
   {
      proj_targ.SetSize(qspace.GetSize());
      targ_qf.reset(new QuadratureFunction(&qspace, proj_targ.GetData()));
      targ_cf.Project(*targ_qf);
      comm = dynamic_cast<ParMesh*>(qspace.GetMesh())->GetComm();
   }

   real_t Eval(const Vector &x) override
   {
      real_t obj = 0.0;
      if (targ_gf)
      {
         ParGridFunction x_gf(targ_gf->ParFESpace(), x.GetData());
         obj = x_gf.ComputeL2Error(targ_cf);
         obj = obj * obj * 0.5;
      }
      else
      {
         const Vector &weights = targ_qf->GetSpace()->GetWeights();
         for (int i=0; i<proj_targ.Size(); i++)
         {
            const real_t val = proj_targ[i] - x[i];
            obj += weights[i] * val*val;
         }
         obj *= 0.5;
         MPI_Allreduce(MPI_IN_PLACE, &obj, 1, MFEM_MPI_REAL_T,
                       MPI_SUM, comm);
      }
      return obj;
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      subtract(x, proj_targ, y);
   }
};

class LatentVolumeProjector
{
private:
protected:
   const real_t vdim;
   const Vector &targetVolume;
   FiniteElementSpace *fespace=nullptr;
   QuadratureSpace *qspace=nullptr;
   MPI_Comm comm;
public:
   enum PrimalType
   {
      GF, QF
   };
   PrimalType ptype;

private:
protected:
public:
   LatentVolumeProjector(const Vector &targetVolume,
                         ParFiniteElementSpace &fes):vdim(targetVolume.Size()),
      targetVolume(targetVolume),
      fespace(&fes), ptype(GF),
      comm(dynamic_cast<ParFiniteElementSpace*>(&fes)->GetComm())
   { }
   LatentVolumeProjector(const Vector &targetVolume,
                         QuadratureSpaceBase &qspace):vdim(targetVolume.Size()),
      targetVolume(targetVolume), qspace(static_cast<QuadratureSpace*>(&qspace)),
      comm(static_cast<ParMesh*>(qspace.GetMesh())->GetComm()), ptype(QF)
   { }
   MPI_Comm GetComm() { return comm; }
   virtual void Apply(Vector &x, const Vector &lower, const Vector &upper,
                      const real_t step_size, const Vector &search_l, const Vector &search_r) = 0;
};


/**
   * @brief Projector for scalar volume int u = targetVolume
**/
class ScalarLatentVolumeProjector : public LatentVolumeProjector
{
private:
   Vector &primal;
   const Vector &pos;
   std::unique_ptr<ParGridFunction> primal_gf;
   std::unique_ptr<QuadratureFunction> primal_qf;
   std::unique_ptr<ParLinearForm> integrator;
public:
   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               const Vector &pos,
                               ParFiniteElementSpace &fes,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, fes), primal(primal),
        pos(pos),
        primal_gf(new ParGridFunction(&fes, primal))
   { }
   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               const Vector &pos,
                               QuadratureSpaceBase &qspace,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, qspace), primal(primal),
        pos(pos)
   { primal_qf.reset(new QuadratureFunction(this->qspace, primal.GetData()));}

   real_t calculateMass(const QuadratureFunction &q1)
   {
      Mesh *mesh = qspace->GetMesh();
      const int NE = mesh->GetNE();
      real_t integral = 0.0;
      for (int e = 0; e < NE; e++)
      {
         const IntegrationRule &ir = qspace->GetElementIntRule(e);
         const int nqp = ir.GetNPoints();

         // Transformation w.r.t. the given mesh positions.
         IsoparametricTransformation Tr;
         mesh->GetElementTransformation(e, pos, &Tr);

         Vector q1_vals(nqp);
         q1.GetValues(e, q1_vals);

         for (int q = 0; q < nqp; q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            integral += Tr.Weight() * ip.weight * q1_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM, comm);
      return integral;
   }

   real_t calculateMass(const ParGridFunction &g) const
   {
      real_t mass = 0.0;
      const int NE = g.ParFESpace()->GetNE();
      for (int e = 0; e < NE; e++)
      {
         auto el = g.ParFESpace()->GetFE(e);
         auto ir = IntRules.Get(el->GetGeomType(), el->GetOrder() + 2);
         IsoparametricTransformation Tr;
         // Must be w.r.t. the given positions.
         g.ParFESpace()->GetParMesh()->GetElementTransformation(e, pos, &Tr);

         Vector g_vals(ir.GetNPoints());
         g.GetValues(Tr, ir, g_vals);

         for (int q = 0; q < ir.GetNPoints(); q++)
         {
            const IntegrationPoint &ip = ir.IntPoint(q);
            Tr.SetIntPoint(&ip);
            mass += Tr.Weight() * ip.weight * g_vals(q);
         }
      }
      MPI_Allreduce(MPI_IN_PLACE, &mass, 1, MPI_DOUBLE, MPI_SUM,
                    g.ParFESpace()->GetComm());
      return mass;
   }

   void Apply(Vector &x, const Vector &lower, const Vector &upper,
              const real_t step_size, const Vector &search_l, const Vector &search_r)
   {
      real_t bisec_lower = search_l[0];
      real_t bisec_upper = search_r[0];

      const bool use_dev = primal.UseDevice() || x.UseDevice();
      const int N = primal.Size();
      auto primal_rw = primal.ReadWrite(use_dev);
      auto x_r = x.Read(use_dev);
      auto l_r = lower.Read(use_dev);
      auto u_r = upper.Read(use_dev);
      real_t vol;
      real_t mid;
      while (bisec_upper - bisec_lower > 1e-08)
      {
         mid = 0.5*(bisec_lower + bisec_upper);
         mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { primal_rw[i] = sigmoid(x_r[i] + step_size*mid, l_r[i], u_r[i]); });
         vol = 0.0;
         if (qspace)
         {
            vol = calculateMass(*primal_qf);
         }
         else
         {
            vol = calculateMass(*primal_gf);
         }
         if (vol < targetVolume[0])
         {
            bisec_lower = mid;
         }
         else
         {
            bisec_upper = mid;
         }
      }
      x += step_size*mid;
      if (Mpi::Root()) { out << " vol-diff: " << vol << " - " << targetVolume[0] << " = " << vol - targetVolume[0] << std::flush; }
   }
};

class BoxMirrorDescent
{
private:
protected:
   DifferentiableObjective &obj;
   const Vector &lower, &upper;
   Vector &primal;
   Vector grad, xnew;
   LatentVolumeProjector *projector;
   int max_iter;
   real_t tol;
   int verbose = 0;
public:

private:
protected:
public:
   BoxMirrorDescent(DifferentiableObjective &obj, Vector &primal,
                    const Vector &lower, const Vector &upper,
                    int max_iter = 1000, real_t tol = 1e-08)
      : obj(obj), primal(primal), lower(lower), upper(upper), max_iter(max_iter),
        tol(tol)
   {}

   void SetVerbose(int lv=1) { verbose = lv; }

   void AddProjector(LatentVolumeProjector &projector)
   {
      this->projector = &projector;
   }

   void UpdatePrimal(const Vector &x)
   {
      for (int i=0; i<x.Size(); i++)
      {
         primal[i] = sigmoid(x[i], lower[i], upper[i]);
      }
   }

   void Step(const Vector &x, real_t step_size, Vector &y)
   {
      UpdatePrimal(x);
      grad.SetSize(primal.Size());
      obj.Mult(primal, grad);

      add(x, -step_size, grad, y);
   }
   void Step(Vector &x, real_t step_size)
   {
      UpdatePrimal(x);
      grad.SetSize(primal.Size());
      obj.Mult(primal, grad);

      x.Add(-step_size, grad);
   }

   void Optimize(Vector &x)
   {
      xnew.SetSize(x.Size());
      real_t step_size = 1.0;
      if (projector)
      {
         Vector search_l(1), search_r(1);
         search_l[0] = -1e6;
         search_r[0] = 1e6;
         projector->Apply(x, lower, upper, 1.0, search_l, search_r);
      }
      for (int i=0; i<max_iter; i++)
      {
         /* step_size = i+1.0; */
         if (verbose) { if (Mpi::Root()) std::cout << "Iter: " << i << " step-size: " << step_size << std::flush; }
         Step(x, step_size);
         if (projector)
         {
            Vector search_l(1), search_r(1);
            search_l[0] = -grad.Normlinf();
            search_r[0] = +grad.Normlinf();
            MPI_Allreduce(MPI_IN_PLACE, search_l.GetData(), 1, MFEM_MPI_REAL_T, MPI_MIN,
                          projector->GetComm());
            MPI_Allreduce(MPI_IN_PLACE, search_r.GetData(), 1, MFEM_MPI_REAL_T, MPI_MAX,
                          projector->GetComm());
            projector->Apply(x, lower, upper, step_size, search_l, search_r);
         }
         UpdatePrimal(x);
         real_t val = obj.Eval(primal);
         if (verbose) { if (Mpi::Root()) { std::cout << " val: " << val << std::endl; } }
      }
   }
};


}

#endif // REMHOS_LVPP_HPP
