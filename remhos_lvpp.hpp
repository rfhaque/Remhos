
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
   if (x > 0) { return 1.0 / (1.0 + std::exp(-x)); }
   else { return std::exp(x) / (1.0 + std::exp(x)); }
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
   Vector proj_targ;
   Vector diff_targ;
   std::unique_ptr<ParGridFunction> targ_gf;
   std::unique_ptr<BilinearForm> mass_form;
   std::unique_ptr<QuadratureFunction> targ_qf;
   MPI_Comm comm;
public:

private:
protected:
public:
   L2Obj(ParFiniteElementSpace &fes, Coefficient &targ_cf)
   {
      proj_targ.SetSize(fes.GetVSize());
      targ_gf.reset(new ParGridFunction(&fes, proj_targ.GetData()));
      targ_gf->ProjectCoefficient(targ_cf);
      mass_form.reset(new ParBilinearForm(&fes));
      comm = fes.GetComm();
      mass_form->AddDomainIntegrator(new MassIntegrator());
      mass_form->Assemble();
   }

   L2Obj(QuadratureSpaceBase &qspace, Coefficient &targ_cf)
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
         proj_targ -= x;
         obj = mass_form->InnerProduct(proj_targ, proj_targ) * 0.5;
         proj_targ += x;
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
      }
      MPI_Allreduce(MPI_IN_PLACE, &obj, 1, MFEM_MPI_REAL_T,
                    MPI_SUM, comm);
      return obj;
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      y = x;
      y -= proj_targ;
   }
};

class LatentVolumeProjector
{
private:
protected:
   const real_t vdim;
   const Vector &targetVolume;
   FiniteElementSpace *fespace=nullptr;
   QuadratureSpaceBase *qspace=nullptr;
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
      targetVolume(targetVolume), qspace(&qspace),
      comm(dynamic_cast<ParMesh*>(qspace.GetMesh())->GetComm()), ptype(QF)
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
   std::unique_ptr<ParGridFunction> primal_gf;
   std::unique_ptr<GridFunctionCoefficient> primal_cf;
   std::unique_ptr<ParLinearForm> integrator;
public:
   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               ParFiniteElementSpace &fes,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, fes), primal(primal),
        primal_gf(new ParGridFunction(&fes, primal)),
        primal_cf(new GridFunctionCoefficient(primal_gf.get()))
   {
      integrator.reset(new ParLinearForm(&fes));
      integrator->AddDomainIntegrator(new DomainLFIntegrator(*primal_cf));
   }
   ScalarLatentVolumeProjector(const Vector &targetVolume,
                               QuadratureSpaceBase &qspace,
                               Vector &primal)
      : LatentVolumeProjector(targetVolume, qspace), primal(primal)
   { }

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
      while (bisec_upper - bisec_lower > 1e-08)
      {
         const real_t mid = 0.5*(bisec_lower + bisec_upper);
         mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { primal_rw[i] = sigmoid(x_r[i] + step_size*mid, l_r[i], u_r[i]); });
         vol = 0.0;
         if (qspace)
         {
            QuadratureFunction qf(qspace, primal.GetData());
            vol = qf.Integrate();
         }
         else
         {
            integrator->Assemble();
            vol = integrator->Sum();
         }
         MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MFEM_MPI_REAL_T,
                       MPI_SUM, MPI_COMM_WORLD);
         if (vol < targetVolume[0])
         {
            bisec_lower = mid;
         }
         else
         {
            bisec_upper = mid;
         }
      }
      if (Mpi::Root()) { out << " vol-diff: " << vol - targetVolume[0] << std::flush; }
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
   BoxMirrorDescent(DifferentiableObjective &obj,
                    Vector &primal,
                    const Vector &lower, const Vector &upper, int max_iter = 100,
                    real_t tol = 1e-08)
      : obj(obj), primal(primal), lower(lower), upper(upper), max_iter(max_iter),
        tol(tol)
   {}

   void SetVerbose(int lv=1) { verbose = lv; }

   void AddProjector(LatentVolumeProjector &projector)
   {
      this->projector = &projector;
   }
   void GetPrimal(Vector &x)
   {
      x.SetSize(primal.Size());
      x = primal;
   }

   void UpdatePrimal(const Vector &x)
   {
      const bool use_dev = primal.UseDevice() || x.UseDevice();
      const int N = primal.Size();
      auto primal_w = primal.Write(use_dev);
      auto x_r = x.Read(use_dev);
      auto l_r = lower.Read(use_dev);
      auto u_r = upper.Read(use_dev);
      mfem::forall_switch(use_dev, N, [=] MFEM_HOST_DEVICE (int i) { primal_w[i] = sigmoid(x_r[i], l_r[i], u_r[i]); });
   }

   void Step(const Vector &x, real_t step_size, Vector &y)
   {
      UpdatePrimal(x);
      grad.SetSize(primal.Size());
      obj.Mult(primal, grad);

      y = x;
      y.Add(-step_size, grad);
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
         std::cout << "HI" << std::endl;
      }
      for (int i=0; i<max_iter; i++)
      {
         /* step_size = i+1.0; */
         if (verbose) { if (Mpi::Root()) std::cout << "Iter: " << i << " step-size: " << step_size << std::flush; }
         Step(x, step_size);
         if (projector)
         {
            Vector search_l(1), search_r(1);
            search_l[0] = -grad.Max();
            search_r[0] = -grad.Min();
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
