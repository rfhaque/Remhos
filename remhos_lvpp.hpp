#ifndef REMHOS_LVPP_HPP
#define REMHOS_LVPP_HPP
#include "fem/coefficient.hpp"
#include "fem/fe_coll.hpp"
#include "fem/pfespace.hpp"
#include "linalg/solvers.hpp"
#include "linalg/vector.hpp"
#include "mfem.hpp"
#include "mpi.h"

namespace mfem
{

inline real_t sigmoid(const real_t x)
{
   if (x > 0) { return 1.0 / (1.0 + std::exp(-x)); }
   else { return std::exp(x) / (1.0 + std::exp(x)); }
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

class L2Projector
{
   // attributes
private:
   ParFiniteElementSpace &fes;
   const IntegrationRule *ir;
protected:
public:

   // methods
private:
protected:
public:
   L2Projector(ParFiniteElementSpace &fes,
               const IntegrationRule *ir = nullptr):fes(fes), ir(ir)
   {
      if (!dynamic_cast<const L2_FECollection*>(fes.FEColl()))
      {
         MFEM_ABORT("Only L2_FECollection is supported");
      }
   }
   void Project(Coefficient &source, Vector &dest) const
   {
      InverseIntegrator inv_mass(new MassIntegrator(ir));
      DomainLFIntegrator int_source(source, ir);
      Vector int_val;
      Vector loc_result;
      Array<int> dofs;
      for (int el = 0; el < fes.GetNE(); el++)
      {
         int_source.AssembleRHSElementVect(
            *fes.GetFE(el), *fes.GetElementTransformation(el), int_val);
         inv_mass.AssembleElementVector(
            *fes.GetFE(el), *fes.GetElementTransformation(el), int_val, loc_result);
         fes.GetElementDofs(el, dofs);
         dest.SetSubVector(dofs, loc_result);
      }
   }
};

class ElementwiseConstantCoefficient : public Coefficient
{
   // attributes
private:
   const Vector &values;
protected:
public:

   // methods
private:
protected:
public:
   ElementwiseConstantCoefficient(const Vector &values)
      :values(values) {}
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      return values[T.ElementNo];
   }
};

class LVPP_BoxCoeff : public GridFunctionCoefficient
{
   // attributes
private:
   Coefficient * lower;
   Coefficient * upper;
   bool own_coeffs;
protected:
public:

   // methods
private:
protected:
public:
   LVPP_BoxCoeff(GridFunction &gf, const real_t lower = 0, const real_t upper = 0)
      : GridFunctionCoefficient(&gf), lower(new ConstantCoefficient(lower)),
        upper(new ConstantCoefficient(upper)), own_coeffs(true) {}
   LVPP_BoxCoeff(GridFunction &gf, Coefficient &lower, Coefficient &upper)
      : GridFunctionCoefficient(&gf), lower(&lower), upper(&upper),
        own_coeffs(false) {}
   ~LVPP_BoxCoeff()
   {
      if (own_coeffs)
      {
         delete lower;
         delete upper;
      }
   }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t val = GridFunctionCoefficient::Eval(T, ip);
      real_t l = lower->Eval(T, ip);
      real_t u = upper->Eval(T, ip);
      return l + (u-l) * sigmoid(val);
   }
};

class LVPP_BoxQCoeff : public QuadratureFunctionCoefficient
{
   // attributes
private:
   QuadratureFunctionCoefficient * lower;
   QuadratureFunctionCoefficient * upper;
   GridFunctionCoefficient gf;
   bool own_coeffs;
protected:
public:

   // methods
private:
protected:
public:
   LVPP_BoxQCoeff(GridFunction &gf, QuadratureFunctionCoefficient &lower,
                  QuadratureFunctionCoefficient &upper)
      : QuadratureFunctionCoefficient(lower.GetQuadFunction()), lower(&lower),
        upper(&upper), gf(&gf),
        own_coeffs(false) {}
   ~LVPP_BoxQCoeff()
   {
      if (own_coeffs)
      {
         delete lower;
         delete upper;
      }
   }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t val = gf.Eval(T, ip);
      real_t l = lower->Eval(T, ip);
      real_t u = upper->Eval(T, ip);
      return l + (u-l) * sigmoid(val);
   }
};

class LVPP_BoxDerCoeff : public GridFunctionCoefficient
{
   // attributes
private:
   Coefficient * lower;
   Coefficient * upper;
   bool own_coeffs;
protected:
public:

   // methods
private:
protected:
public:
   LVPP_BoxDerCoeff(GridFunction &gf, const real_t lower = 0,
                    const real_t upper = 0)
      : GridFunctionCoefficient(&gf), lower(new ConstantCoefficient(lower)),
        upper(new ConstantCoefficient(upper)), own_coeffs(true) {}
   LVPP_BoxDerCoeff(GridFunction &gf, Coefficient &lower, Coefficient &upper)
      : GridFunctionCoefficient(&gf), lower(&lower), upper(&upper),
        own_coeffs(false) {}
   ~LVPP_BoxDerCoeff()
   {
      if (own_coeffs)
      {
         delete lower;
         delete upper;
      }
   }
   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip)
   {
      real_t val = GridFunctionCoefficient::Eval(T, ip);
      real_t l = lower->Eval(T, ip);
      real_t u = upper->Eval(T, ip);
      real_t sigval = sigmoid(val);
      return (u-l) * sigval *(1.0 - sigval);
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

class HessQuadForm : public ObjectiveHessian
{
protected:
   ParBilinearForm &inner;
   ParBilinearForm *inv_inner;
public:
   HessQuadForm(ParBilinearForm &inner,
                ParBilinearForm *inv_inner=nullptr): ObjectiveHessian(
                      inner.ParFESpace()->GetVSize()), inner(inner), inv_inner(inv_inner) { }
   void At(const Vector &x) override
   {
      // nothing to do
   }
   void Mult(const Vector &x, Vector &y) const override
   {
      inner.Mult(x, y);
   }
   void InvMult(const Vector &x, Vector &y) const override
   {
      if (inv_inner) { inv_inner->Mult(x, y); }
      else {MFEM_ABORT("Inverse not set"); }
   }
};


class L2Objective : public DifferentiableObjective
{
   // attributes
private:
protected:
   ParFiniteElementSpace &fes;
   const L2Projector projector;
   std::unique_ptr<ParGridFunction> gf_view;
   std::unique_ptr<ParGridFunction> zero_gf;
   mutable std::unique_ptr<Coefficient> gf_cf;
   mutable std::unique_ptr<SumCoefficient> diff_cf;
   Coefficient &targ;
   std::unique_ptr<ParBilinearForm> inner;
   std::unique_ptr<ParBilinearForm> inv_inner;
public:
private:
   // methods
protected:
   virtual void SetCoefficients() const
   {
      gf_cf.reset(new GridFunctionCoefficient(gf_view.get()));
      diff_cf.reset(new SumCoefficient(*gf_cf, targ, 1.0, -1.0));
   }
public:
   L2Objective(ParFiniteElementSpace &fes, ParGridFunction &gf, Coefficient &targ,
               const IntegrationRule *irs = nullptr)
      : DifferentiableObjective(fes.GetVSize()), fes(fes), projector(fes), targ(targ)
   {
      comm = fes.GetComm();
      gf_view.reset(new ParGridFunction(&fes, (real_t*)nullptr));
      zero_gf.reset(new ParGridFunction(&fes));
      *zero_gf=0.0;

      inner.reset(new ParBilinearForm(&fes));
      inner->AddDomainIntegrator(new MassIntegrator(irs));
      inner->Assemble();
      inv_inner.reset(new ParBilinearForm(&fes));
      inv_inner->AddDomainIntegrator(new InverseIntegrator(new MassIntegrator(irs)));
      inv_inner->Assemble();
      hessian.reset(new HessQuadForm(*inner, inv_inner.get()));
   }

   real_t Eval(const Vector &x) override
   {
      gf_view->SetData(x.GetData());
      SetCoefficients();
      const real_t val = zero_gf->ComputeL2Error(*diff_cf);
      return 0.5 * val * val;
   }

   void Mult(const Vector &x, Vector &y) const override
   {
      gf_view->SetData(x.GetData());
      SetCoefficients();
      projector.Project(*diff_cf, y);
   }
};

class LVPP_L2Objective : public L2Objective
{
   // attributes
protected:
   Coefficient *lower;
   Coefficient *upper;
   const QuadratureFunction *lower_qf;
   const QuadratureFunction *upper_qf;
   const IntegrationRule *ir;
   std::unique_ptr<QuadratureFunction> u_qf;

   // methods
protected:
   void SetCoefficients() const override
   {
      if (lower_qf)
      {
         u_qf->ProjectGridFunction(*gf_view);
         int idx = 0;
         Vector vals;
         for (int i=0; i<gf_view->ParFESpace()->GetNE(); i++)
         {
            const IntegrationRule &ir = u_qf->GetIntRule(i);
            gf_view->GetValues(i, ir, vals);
            for (int j=0; j<ir.GetNPoints(); j++)
            {
               (*u_qf)[idx] = (*lower_qf)[idx] + ((*upper_qf)[idx] - (*lower_qf)[idx]) *
                              sigmoid(vals[j]);
               idx++;
            }
         }
         gf_cf.reset(new QuadratureFunctionCoefficient(*u_qf));
      }
      else
      {
         gf_cf.reset(new LVPP_BoxCoeff(*gf_view, *lower, *upper));
      }
      diff_cf.reset(new SumCoefficient(*gf_cf, targ, 1.0, -1.0));
   }
public:
   LVPP_L2Objective(ParFiniteElementSpace &fes, ParGridFunction &gf,
                    Coefficient &targ, Coefficient &lower, Coefficient &upper,
                    const IntegrationRule *ir = nullptr)
      : L2Objective(fes, gf, targ, ir), lower(&lower), upper(&upper)
   {
      auto lower_qfcf = dynamic_cast<QuadratureFunctionCoefficient*>(&lower);
      if (lower_qfcf)
      {
         auto upper_qfcf = dynamic_cast<QuadratureFunctionCoefficient*>(&upper);
         lower_qf = &lower_qfcf->GetQuadFunction();
         upper_qf = &upper_qfcf->GetQuadFunction();
         u_qf.reset(new QuadratureFunction(*lower_qf));
      }
   }
};

class LVPP_BoxOptimizer
{
private:
public:
   DifferentiableObjective &obj;
   real_t tol;
   int max_it;
   Vector grad;
   Vector H_invGrad;
   ParLinearForm *int_vol;
   QuadratureFunction *vol_qf;
   real_t targ_vol;
   real_t step_size;
public:
   LVPP_BoxOptimizer(DifferentiableObjective &obj, real_t tol, int max_it)
      :obj(obj), tol(tol), max_it(max_it), grad(obj.Width()),
       H_invGrad(obj.Width()) { }
   void SetVolumeConstrant(ParLinearForm &int_volume, real_t targ_volume)
   { int_vol = &int_volume; targ_vol = targ_volume; }
   /* void SetVolumeConstrant(QuadratureFunction &volume_qf, real_t targ_volume) */
   /* { vol_qf = &volume_qf; targ_vol = targ_volume; QuadratureFunctionCoefficient qfcf(volume_qf);} */

   void Step(Vector &x)
   {
      const real_t val = obj.Eval(x);
      obj.Mult(x, grad);
      /* static_cast<HessQuadForm&>(obj.GetGradient(x)).InvMult(grad, H_invGrad); */
      x.Add(-step_size, grad);

      if (!int_vol) {return;}
      MPI_Comm comm = obj.GetComm();

      real_t maxval = grad.Max();
      real_t minval = grad.Min();
      MPI_Allreduce(MPI_IN_PLACE, &maxval, 1, MFEM_MPI_REAL_T, MPI_MAX,
                    comm);
      MPI_Allreduce(MPI_IN_PLACE, &minval, 1, MFEM_MPI_REAL_T, MPI_MIN,
                    comm);
      x += step_size*maxval;
      int_vol->Assemble();
      real_t upper = InnerProduct(*int_vol, x);
      MPI_Allreduce(MPI_IN_PLACE, &upper, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
      x += step_size*(minval - maxval);
      int_vol->Assemble();
      real_t lower = InnerProduct(*int_vol, x);
      MPI_Allreduce(MPI_IN_PLACE, &lower, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
      real_t dc = (maxval - minval)*0.5;
      x -= step_size*minval;
      // bisection
      real_t vol;
      while (dc > 1e-10)
      {
         int_vol->Assemble();
         vol = InnerProduct(*int_vol, x);
         MPI_Allreduce(MPI_IN_PLACE, &vol, 1, MFEM_MPI_REAL_T, MPI_SUM, comm);
         if (vol > targ_vol)
         {
            x -= step_size*dc;
            dc *= 0.5;
         }
         else if (vol < targ_vol)
         {
            x += step_size*dc;
            dc *= 0.5;
         }
         else
         {
            break;
         }
      }
      if (Mpi::Root()) { out << "Volume constraint satisfied with residual " << std::fabs(vol - targ_vol) << std::endl; }
   }

   void Optimize(Vector &x)
   {
      real_t succ_diff = infinity();
      real_t obj_prev = obj.Eval(x);
      real_t obj_curr = obj.Eval(x);
      int i=0;
      step_size = 1.0;
      while (std::fabs(succ_diff) > tol && i < max_it)
      {
         i++;
         if (Mpi::Root()) { out << "Step " << i << ": " << std::flush; }
         Step(x);
         obj_curr = obj.Eval(x);
         if (Mpi::Root()) { out << obj_curr << std::endl; }
         succ_diff = obj_prev - obj_curr;
         obj_prev = obj_curr;
      }
   }
};

}

#endif // REMHOS_LVPP_HPP
