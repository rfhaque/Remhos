// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include "remhos_gslib.hpp"
#include "remhos_tools.hpp"
#include "remhos_HiOp.hpp"
#include "remhos_lvpp.hpp"

#include "examples/remap_opt.hpp"
#include <algorithm>

using namespace std;

namespace mfem
{

void InitializeQuadratureFunction(Coefficient &c,
                                  const Vector &pos_mesh,
                                  QuadratureFunction &q)
{
   auto qspace = dynamic_cast<QuadratureSpace *>(q.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   const int NE  = qspace->GetMesh()->GetNE();
   real_t *q_data = q.GetData();
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace->GetElementIntRule(e);
      const int nip = ir.GetNPoints();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      qspace->GetMesh()->GetElementTransformation(e, pos_mesh, &Tr);

      for (int q = 0; q < nip; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         q_data[e*nip + q] = c.Eval(Tr, ip);
      }
   }
}

void VisQuadratureFunction(ParMesh &pmesh, QuadratureFunction &q,
                           std::string info, int x, int y)
{
   osockstream sol_sock(19916, "localhost");
   sol_sock << "parallel " << pmesh.GetNRanks() << " "
                           << pmesh.GetMyRank() << "\n";
   sol_sock << "quadrature\n" << pmesh << q << std::flush;
   sol_sock << "window_title '" << info << "'\n";
   sol_sock << "window_geometry " << x << " " << y << " 400 400\n";
   sol_sock << "keys rmj\n";
   sol_sock.send();
}

void InterpolationRemap::Remap(const ParGridFunction &u_initial,
                               const ParGridFunction &pos_final,
                               ParGridFunction &u_final, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   ParFiniteElementSpace pfes_tmp(&pmesh_final, u_final.ParFESpace()->FEColl());

   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   ParFiniteElementSpace &pfes_final = *u_final.ParFESpace();

   {
      ParaViewDataCollection pvdc("initla_mesh", &pmesh_init);
      pvdc.SetDataFormat(VTKFormat::BINARY32);
      pvdc.SetCycle(0);
      pvdc.SetTime(1.0);

      pvdc.RegisterField("val", const_cast<ParGridFunction*>(&u_initial));
      pvdc.Save();
   }

   // Generate list of points where u_initial will be interpolated.
   Vector pos_dof_final;
   GetDOFPositions(pfes_final, pos_final, pos_dof_final);

   // Interpolate u_initial.
   const int nodes_cnt = pos_dof_final.Size() / dim;
   Vector interp_vals(nodes_cnt);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, u_initial, interp_vals);
   finder.FreeData();

   // This assumes L2 ordering of the DOFs (as the ordering of the quad points).
   ParGridFunction u_interpolated(&pfes_final);
   ParGridFunction u_interpolated_initial(&pfes_final);
   u_interpolated = interp_vals;
   u_interpolated_initial = interp_vals;

   // Report masses.
   double mass_0 = Mass(*pmesh_init.GetNodes(), u_initial),
          mass_f = Mass(pos_final, u_interpolated);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial (old mesh):  " << mass_0 << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f) / mass_0 * 100 << endl;
   }

   // Compute min / max bounds.
   Vector u_final_min, u_final_max;
   CalcDOFBounds(u_initial, pfes_final, pos_final,
                 u_final_min, u_final_max, false);
   if (visualization)
   {
      ParGridFunction gf_min(u_initial), gf_max(u_initial);
      gf_min = u_final_min, gf_max = u_final_max;

      socketstream vis_min, vis_max;
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_min.precision(8);
      vis_max.precision(8);


      *x = pos_final;
      VisualizeField(vis_min, vishost, visport, gf_min, "u min",
                     0, 500, 300, 300);
      VisualizeField(vis_max, vishost, visport, gf_max, "u max",
                     300, 500, 300, 300);

      {
         ParaViewDataCollection pvdc("bounds", &pmesh_init);
         pvdc.SetDataFormat(VTKFormat::BINARY32);
         pvdc.SetCycle(0);
         pvdc.SetTime(1.0);
         pvdc.RegisterField("field_min", &gf_min);
         pvdc.RegisterField("field_max", &gf_max);
         pvdc.RegisterField("val", &u_interpolated);
         pvdc.Save();
      }

      *x = pos_init;
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 0)
   {
      u_final = u_interpolated;
   }
   else if (opt_type == 1)
   {
      *x = pos_final;
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         optsolver = new HiopNlpOptimizer(MPI_COMM_WORLD);
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      const int max_iter = 20;
      const double rtol = 1.e-7;
      const double atol = 1.e-7;
      Vector y_out(u_interpolated.Size());

      const int numContraints = 1;
      RemhosHiOpProblem ot_prob(pfes_final,
                                u_interpolated_initial, u_interpolated,
                                u_final_min, u_final_max,
                                mass_0, numContraints, h1_seminorm);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(atol);
      optsolver->SetRelTol(rtol);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u_interpolated, y_out);

      // fix parallel. u_interpolated and y_out should be true vectors
      u_final = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      MDSolver md(pfes_tmp, mass_0, u_interpolated, u_final_min, u_final_max);
      md.Optimize(1000, 1000, 1000);
      md.SetFinal(u_final);
   }
   else if (opt_type == 3)
   {
      GridFunctionCoefficient u_interpolated_cf(&u_interpolated);
      L2Obj obj(*u_final.ParFESpace(), u_interpolated_cf);
      BoxMirrorDescent md(obj, u_final, u_final_min, u_final_max);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final, *u_final.ParFESpace(), u_final);
      md.AddProjector(projector);
      ParGridFunction psi(u_final);
      psi = 0.0;
      md.SetVerbose(1);
      md.Optimize(psi);
      md.UpdatePrimal(psi);
   }
   else if (opt_type == 4)
   {
      GridFunctionCoefficient u_interpolated_cf(&u_interpolated);
      L2Obj obj(*u_final.ParFESpace(), u_interpolated_cf);
      BoxMirrorDescent md(obj, u_final, u_final_min, u_final_max);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final, *u_final.ParFESpace(), u_final);
      md.AddProjector(projector);
      ParGridFunction psi(u_final);
      for (int i=0; i<u_interpolated.Size(); i++)
      {
         psi[i] = inv_sigmoid(u_interpolated[i], u_final_min[i], u_final_max[i]);
      }
      Vector search_l({-1e09}), search_r({1e09}), lambda(1);
      projector.Apply(psi, u_final_min, u_final_max, 1.0, search_l, search_r, lambda);

      /* md.SetVerbose(1); */
      /* md.Optimize(psi); */
      /* md.UpdatePrimal(psi); */
   }

   // Report masses.
   mass_f = Mass(pos_final, u_final);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass_0 - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass_0 - mass_f) / mass_0 * 100 << endl;
   }
}

void InterpolationRemap::Remap(const QuadratureFunction &u_0,
                               const ParGridFunction &pos_final,
                               QuadratureFunction &u, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   QuadratureSpace qspace_tmp(pmesh_final, u_0.GetIntRule(0));

   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   auto qspace = dynamic_cast<QuadratureSpace *>(u.GetSpace());
   MFEM_VERIFY(qspace, "Broken QuadratureSpace.");

   // Generate list of points where u_initial will be interpolated.
   Vector pos_quad_final;
   GetQuadPositions(*qspace, pos_final, pos_quad_final);

   // Generate the Low-Order-Refined GridFunction for interpolation.
   const int order = u_0.GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                                            BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);
   ParGridFunction u_0_lor(&pfes_lor);
   MFEM_VERIFY(u_0.Size() == u_0_lor.Size(), "Size mismatch");
   u_0_lor = u_0;

   // Visualize the initial LOR GridFunction.
   if (visualization)
   {
      socketstream sock;
      VisualizeField(sock, "localhost", 19916, u_0_lor, "u_0 LOR", 800, 0, 400, 400);
   }

   // Interpolate u_initial.
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, u_0_lor, u);
   finder.FreeData();

   // Report mass error.
   double mass_0 = Integrate(*pmesh_init.GetNodes(), &u_0,
                             nullptr, nullptr),
          mass_f = Integrate(pos_final, &u, nullptr, nullptr);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial (old mesh):  " << mass_0 << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f)/mass_0*100 << endl;
   }

   // Compute min / max bounds.
   Vector u_min, u_max;
   CalcQuadBounds(u_0, pos_final, u_min, u_max);
   if (visualization)
   {
      QuadratureFunction gf_min(qspace), gf_max(qspace);
      gf_min = u_min, gf_max = u_max;

      *x = pos_final;
      VisQuadratureFunction(pmesh_init, gf_min, "u_min QF", 0, 500);
      VisQuadratureFunction(pmesh_init, gf_max, "u_max QF", 400, 500);
      *x = pos_init;
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 1)
   {
      QuadratureFunction u_desing(u), u_initial(u);
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer(MPI_COMM_WORLD);
         optsolver = tmp_opt_ptr;
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      const int max_iter = 100;
      const double rtol = 1.e-6;
      const double atol = 1.e-6;
      Vector y_out(u_desing.Size());

      const int numContraints = 1;
      const double H1SeminormWeight = 0.0;

      RemhosQuadHiOpProblem ot_prob(*qspace, pos_final,
                                    u_initial, u_desing,
                                    u_min, u_max, mass_0,
                                    numContraints, h1_seminorm);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(atol);
      optsolver->SetRelTol(rtol);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u_desing, y_out);

      u = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      QuadratureFunction u_target(u);
      QDSolver qd(qspace_tmp, mass_0, u_target, u_min, u_max);

      qd.Optimize(1000, 1000, 1000);
      qd.SetFinal(u);
   }
   else if (opt_type == 3)
   {
      QuadratureFunction u_target(u);
      QuadratureFunctionCoefficient u_target_cf(u_target);
      L2Obj obj(*u.GetSpace(), u_target_cf);
      BoxMirrorDescent md(obj, u, u_min, u_max);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final, *u.GetSpace(), u);
      md.AddProjector(projector);
      QuadratureFunction psi(u);
      psi = 0.0;
      md.SetVerbose(1);
      md.Optimize(psi);
   }
   else if (opt_type == 4)
   {
      QuadratureFunction u_target(u);
      QuadratureFunctionCoefficient u_target_cf(u_target);
      L2Obj obj(*u.GetSpace(), u_target_cf);
      BoxMirrorDescent md(obj, u, u_min, u_max);
      Vector target_volume(1); target_volume[0] = mass_0;
      ScalarLatentVolumeProjector projector(target_volume, pos_final, *u.GetSpace(), u);
      QuadratureFunction psi(u);
      for (int i=0; i<u.Size(); i++)
      {
         psi[i] = inv_sigmoid(u[i], u_min[i], u_max[i]);
      }
      Vector search_l({-1e09}), search_r({1e09}), lambda(1);
      projector.Apply(psi, u_min, u_max, 1.0, search_l, search_r, lambda);
   }

   // Report final masses.
   mass_f = Integrate(pos_final, &u, nullptr, nullptr);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass_0 - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass_0 - mass_f)/mass_0*100 << endl;
   }
}

void InterpolationRemap::Remap(std::function<real_t(const Vector &)> func,
                               double mass, const ParGridFunction &pos_final,
                               ParGridFunction &u, int opt_type)
{
   pmesh_final.SetNodes(pos_final);
   ParFiniteElementSpace pfes_tmp(&pmesh_final, u.ParFESpace()->FEColl());

   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");

   // Generate list of points where u_initial will be interpolated.
   // The interpolation is to Gauss-Legendre to keep optimal order.
   L2_FECollection fec_GL(u.ParFESpace()->FEColl()->GetOrder(),
                          dim, BasisType::GaussLegendre);
   ParFiniteElementSpace pfes_GL(u.ParFESpace()->GetParMesh(), &fec_GL);
   Vector pos_dof_final;
   GetDOFPositions(pfes_GL, pos_final, pos_dof_final);

   // Interpolate the function.
   const int nodes_cnt = pos_dof_final.Size() / dim;
   Vector interp_vals(nodes_cnt), node_pos(dim);
   for (int i = 0; i < nodes_cnt; i++)
   {
      for (int d = 0; d < dim; d++)
      {
         node_pos(d) = pos_dof_final(d * nodes_cnt + i);
      }
      interp_vals(i) = func(node_pos);
   }

   // This assumes L2 ordering of the DOFs (as the ordering of the quad points).
   ParGridFunction u_GL(&pfes_GL);
   u_GL = interp_vals;
   // Go Gauss-Legendre -> Bernstein.
   u.ProjectGridFunction(u_GL);

   // Report masses.
   double mass_f = Mass(pos_final, u);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass initial (analytic):  " << mass   << std::endl
                << "Mass interpolated:        " << mass_f << std::endl
                << "Mass interpolated diff:   "
                << fabs(mass - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass - mass_f) / mass * 100 << endl;
   }

   // Compute min / max bounds.
   // Projects to a GridFunction to get some reasonable min/max per element.
   Vector u_final_min, u_final_max;
   ParGridFunction func_gf(u.ParFESpace());
   FunctionCoefficient coeff(func);
   func_gf.ProjectCoefficient(coeff);
   CalcDOFBounds(func_gf, *u.ParFESpace(), pos_final,
                 u_final_min, u_final_max, true);
   if (visualization)
   {
      ParGridFunction gf_min(func_gf), gf_max(func_gf);
      gf_min = u_final_min, gf_max = u_final_max;

      socketstream vis_min, vis_max;
      char vishost[] = "localhost";
      int  visport   = 19916;
      vis_min.precision(8);
      vis_max.precision(8);

      *x = pos_final;
      VisualizeField(vis_min, vishost, visport, gf_min, "u min",
                     0, 500, 300, 300);
      VisualizeField(vis_max, vishost, visport, gf_max, "u max",
                     300, 500, 300, 300);
      *x = pos_init;
   }

   //
   // Optimize to fix the masses, using the min/max bounds.
   //
   if (opt_type == 1)
   {
      *x = pos_final;
      OptimizationSolver* optsolver = NULL;
      {
#ifdef MFEM_USE_HIOP
         HiopNlpOptimizer *tmp_opt_ptr = new HiopNlpOptimizer(MPI_COMM_WORLD);
         optsolver = tmp_opt_ptr;
#else
         MFEM_ABORT("MFEM is not built with HiOp support!");
#endif
      }

      const int max_iter = 100;
      const double rtol = 1.e-7;
      const double atol = 1.e-7;
      Vector y_out(u.Size());

      const int numContraints = 1;

      RemhosHiOpProblem ot_prob(*u.ParFESpace(), u, u,
                                u_final_min, u_final_max, mass,
                                numContraints, h1_seminorm);
      optsolver->SetOptimizationProblem(ot_prob);

      optsolver->SetMaxIter(max_iter);
      optsolver->SetAbsTol(atol);
      optsolver->SetRelTol(rtol);
      optsolver->SetPrintLevel(3);
      optsolver->Mult(u, y_out);

      u = y_out;

      delete optsolver;
   }
   else if (opt_type == 2)
   {
      ParGridFunction u_interpolated(u);
      MDSolver md(pfes_tmp, mass, u_interpolated, u_final_min, u_final_max);

      md.Optimize(100, 1000, 1000);
      md.SetFinal(u);
   }

   // Report masses.
   mass_f = Mass(pos_final, u);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Mass optimized:           " << mass_f << std::endl
                << "Mass optimized diff:      "
                << fabs(mass - mass_f) << endl
                << "Mass optimized diff %:    "
                << fabs(mass - mass_f) / mass * 100 << endl;
   }
}

void InterpolationRemap::RemapIndRhoE(const Vector ind_rho_e_0,
    const ParGridFunction &pos_final,
    Vector &ind_rho_e, int opt_type)
{
   const int dim = pmesh_init.Dimension();
   MFEM_VERIFY(dim > 1, "Interpolation remap works only in 2D and 3D.");
   MFEM_VERIFY(pfes_e && qspace, "Spaces are not specified.");

   // Extract initial data from the BlockVector.
   const int size_qf = qspace->GetSize();
   Vector *ire_ptr = const_cast<Vector *>(&ind_rho_e_0);
   QuadratureFunction ind_0(qspace, ire_ptr->GetData()),
       rho_0(qspace, ire_ptr->GetData() + size_qf);
   ParGridFunction e_0(pfes_e, ire_ptr->GetData() + 2*size_qf);

   // Generate list of points where ire_initial will be interpolated.
   Vector pos_dof_final, pos_quad_final;
   GetDOFPositions(*pfes_e, pos_final, pos_dof_final);
   GetQuadPositions(*qspace, pos_final, pos_quad_final);

   // Generate the Low-Order-Refined GridFunctions for
   // interpolating the QuadratureFunctions.
   const int order = qspace->GetIntRule(0).GetOrder() / 2;
   const int ref_factor = order + 1;
   ParMesh pmesh_lor = ParMesh::MakeRefined(pmesh_init, ref_factor,
                                            BasisType::ClosedGL);
   L2_FECollection fec_lor(0, dim);
   ParFiniteElementSpace pfes_lor(&pmesh_lor, &fec_lor);
   ParGridFunction ind_0_lor(&pfes_lor), rho_0_lor(&pfes_lor);
   MFEM_VERIFY(ind_0.Size() == ind_0_lor.Size(), "Size mismatch ind LOR.");
   MFEM_VERIFY(rho_0.Size() == rho_0_lor.Size(), "Size mismatch rho LOR.");
   ind_0_lor = ind_0;
   rho_0_lor = rho_0;

   // Visualize the initial LOR GridFunctions.
   if (visualization)
   {
      socketstream sock_ind, sock_rho;
      VisualizeField(sock_ind, "localhost", 19916, ind_0_lor, "ind_0 LOR", 0, 500, 400, 400);
      VisualizeField(sock_rho, "localhost", 19916, rho_0_lor, "rho_0 LOR", 400, 500, 400, 400);
   }

   // Interpolate into ind_rho_e.
   QuadratureFunction ind(qspace, ind_rho_e.GetData()),
       rho(qspace, ind_rho_e.GetData() + size_qf);
   ParGridFunction e(pfes_e, ind_rho_e.GetData() + 2*size_qf);
   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_lor);
   finder.Interpolate(pos_quad_final, ind_0_lor, ind);
   finder.Interpolate(pos_quad_final, rho_0_lor, rho);
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_dof_final, e_0, e);
   finder.FreeData();

   // Report conservation errors of ire_final.
   const double volume_0 = Integrate(*pmesh_init.GetNodes(), &ind_0,
                                     nullptr, nullptr),
       volume_f = Integrate(pos_final, &ind,
                            nullptr, nullptr),
       mass_0   = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0,
                          nullptr),
       mass_f   = Integrate(pos_final, &ind, &rho,
                          nullptr),
       energy_0 = Integrate(*pmesh_init.GetNodes(), &ind_0, &rho_0, &e_0),
       energy_f = Integrate(pos_final, &ind, &rho, &e);
   if (pmesh_init.GetMyRank() == 0)
   {
      std::cout << "Volume initial:             " << volume_0 << std::endl
                << "Volume interpolated:        " << volume_f << std::endl
                << "Volume interpolated diff:   "
                << fabs(volume_0 - volume_f) << endl
                << "Volume interpolated diff %: "
                << fabs(volume_0 - volume_f) / volume_0 * 100
                << endl << "*\n"
                << "Mass initial:               " << mass_0 << std::endl
                << "Mass interpolated:          " << mass_f << std::endl
                << "Mass interpolated diff:     "
                << fabs(mass_0 - mass_f) << endl
                << "Mass interpolated diff %: "
                << fabs(mass_0 - mass_f) / mass_0 * 100
                << endl << "*\n"
                << "Energy initial:             " << energy_0 << std::endl
                << "Energy interpolated:        " << energy_f << std::endl
                << "Energy interpolated diff:   "
                << fabs(energy_0 - energy_f) << endl
                << "Energy interpolated diff %: "
                << fabs(energy_0 - energy_f) / energy_0 * 100
                << endl;
   }

   // Compute min / max bounds.
   Vector ind_min, ind_max;
   CalcQuadBounds(ind_0, pos_final, ind_min, ind_max);
   Vector rho_min, rho_max;
   CalcQuadBounds(rho_0, pos_final, rho_min, rho_max);
   Vector e_min, e_max;
   CalcDOFBounds(e_0, *pfes_e, pos_final, e_min, e_max, true);

   // Optimize ire_final here.
   // ...
}

void InterpolationRemap::GetDOFPositions(const ParFiniteElementSpace &pfes,
                                         const Vector &pos_mesh,
                                         Vector &pos_dofs)
{
   const int NE  = pfes.GetNE(), dim = pmesh_init.Dimension();
   const int nsp = pfes.GetFE(0)->GetNodes().GetNPoints();

   pos_dofs.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = pfes.GetFE(e)->GetNodes();

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh_init.GetElementTransformation(e, pos_mesh, &Tr);

      // Node positions of pfes for pos_mesh.
      Vector rowx(pos_dofs.GetData() + e*nsp, nsp),
             rowy(pos_dofs.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_dofs.GetData() + e*nsp + 2*NE*nsp, nsp);
      }

      DenseMatrix pos_nodes;
      Tr.Transform(ir, pos_nodes);
      pos_nodes.GetRow(0, rowx);
      pos_nodes.GetRow(1, rowy);
      if (dim == 3) { pos_nodes.GetRow(2, rowz); }
   }
}

void InterpolationRemap::GetQuadPositions(const QuadratureSpace &qspace,
                                          const Vector &pos_mesh,
                                          Vector &pos_quads)
{
   const int NE  = qspace.GetMesh()->GetNE(), dim = pmesh_init.Dimension();
   const int nsp = qspace.GetElementIntRule(0).GetNPoints();

   pos_quads.SetSize(nsp * NE * dim);
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir = qspace.GetElementIntRule(e);

      // Transformation of the element with the pos_mesh coordinates.
      IsoparametricTransformation Tr;
      pmesh_init.GetElementTransformation(e, pos_mesh, &Tr);

      // Node positions of pfes for pos_mesh.
      DenseMatrix pos_quads_e;
      Tr.Transform(ir, pos_quads_e);
      Vector rowx(pos_quads.GetData() + e*nsp, nsp),
             rowy(pos_quads.GetData() + e*nsp + NE*nsp, nsp), rowz;
      if (dim == 3)
      {
         rowz.SetDataAndSize(pos_quads.GetData() + e*nsp + 2*NE*nsp, nsp);
      }
      pos_quads_e.GetRow(0, rowx);
      pos_quads_e.GetRow(1, rowy);
      if (dim == 3) { pos_quads_e.GetRow(2, rowz); }
   }
}

double InterpolationRemap::Mass(const Vector &pos, const ParGridFunction &g)
{
   double mass = 0.0;

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
                 pmesh_init.GetComm());
   return mass;
}

double InterpolationRemap::Integrate(const Vector &pos,
                                     const QuadratureFunction *q1,
                                     const QuadratureFunction *q2,
                                     const ParGridFunction *g1)
{
   MFEM_VERIFY(q1 || q2 || g1, "At least one function must be specified.");

   const QuadratureSpace *qspace = nullptr;
   if (q1) { qspace = dynamic_cast<const QuadratureSpace *>(q1->GetSpace()); }
   if (q2) { qspace = dynamic_cast<const QuadratureSpace *>(q2->GetSpace()); }

   auto mesh = (qspace) ? qspace->GetMesh() : g1->ParFESpace()->GetMesh();
   const int NE = mesh->GetNE();
   double integral = 0.0;
   for (int e = 0; e < NE; e++)
   {
      const IntegrationRule &ir =
          (qspace) ? qspace->GetElementIntRule(e)
                   : IntRules.Get(g1->ParFESpace()->GetFE(e)->GetGeomType(), 7);
      const int nqp = ir.GetNPoints();

      // Transformation w.r.t. the given mesh positions.
      IsoparametricTransformation Tr;
      mesh->GetElementTransformation(e, pos, &Tr);

      Vector q1_vals(nqp), q2_vals(nqp), g1_vals(nqp);
      if (q1) { q1->GetValues(e, q1_vals); } else { q1_vals = 1.0; }
      if (q2) { q2->GetValues(e, q2_vals); } else { q2_vals = 1.0; }
      if (g1) { g1->GetValues(Tr, ir, g1_vals); } else { g1_vals = 1.0; }

      for (int q = 0; q < nqp; q++)
      {
         const IntegrationPoint &ip = ir.IntPoint(q);
         Tr.SetIntPoint(&ip);
         integral += Tr.Weight() * ip.weight *
                     q1_vals(q) * q2_vals(q) * g1_vals(q);
      }
   }
   MPI_Allreduce(MPI_IN_PLACE, &integral, 1, MPI_DOUBLE, MPI_SUM,
                 pmesh_init.GetComm());
   return integral;
}

void InterpolationRemap::CalcDOFBounds(const ParGridFunction &g_init,
                                       const ParFiniteElementSpace &pfes,
                                       const Vector &pos_final,
                                       Vector &g_min, Vector &g_max,
                                       bool use_nbr)
{
   const int size_res = pfes.GetVSize(), NE = pmesh_init.GetNE();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);

   // Form the min and max functions on every MPI task.
   L2_FECollection fec_L2(0, pmesh_init.Dimension());
   ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
   ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
   for (int e = 0; e < NE; e++)
   {
      Vector g_vals;
      g_init.GetElementDofValues(e, g_vals);
      g_el_min(e) = g_vals.Min();
      g_el_max(e) = g_vals.Max();
   }

   Vector pos_nodes_final;
   GetDOFPositions(pfes, pos_final, pos_nodes_final);

   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_nodes_final, g_el_min, g_min);
   finder.Interpolate(pos_nodes_final, g_el_max, g_max);
   finder.FreeData();

   if (use_nbr)
   {
      for (int e = 0; e < NE; e++)
      {
         Array<int> dofs;
         pfes.GetElementDofs(e, dofs);
         const int s = dofs.Size();

         Vector g_vals;
         g_min.GetSubVector(dofs, g_vals);
         const double minv = g_vals.Min();
         g_max.GetSubVector(dofs, g_vals);
         const double maxv = g_vals.Max();

         for (int i = 0; i < s; i++)
         {
            g_min(s * e + i) = minv;
            g_max(s * e + i) = maxv;
         }
      }
   }
}

void InterpolationRemap::CalcQuadBounds(const QuadratureFunction &qf_init,
                                        const Vector &pos_final,
                                        Vector &g_min, Vector &g_max)
{
   const int size_res = qf_init.Size(), NE = pmesh_init.GetNE();
   g_min.SetSize(size_res);
   g_max.SetSize(size_res);

   // Form the min and max functions on every MPI task.
   L2_FECollection fec_L2(0, pmesh_init.Dimension());
   ParFiniteElementSpace pfes_L2(&pmesh_init, &fec_L2);
   ParGridFunction g_el_min(&pfes_L2), g_el_max(&pfes_L2);
   for (int e = 0; e < NE; e++)
   {
      Vector q_vals;
      qf_init.GetValues(e, q_vals);
      g_el_min(e) = q_vals.Min();
      g_el_max(e) = q_vals.Max();
   }

   Vector pos_quads_final;
   auto qspace = dynamic_cast<const QuadratureSpace *>(qf_init.GetSpace());
   GetQuadPositions(*qspace, pos_final, pos_quads_final);

   FindPointsGSLIB finder(pmesh_init.GetComm());
   finder.Setup(pmesh_init);
   finder.Interpolate(pos_quads_final, g_el_min, g_min);
   finder.Interpolate(pos_quads_final, g_el_max, g_max);
   finder.FreeData();

   // int el_e_idx = 0;
   // for (int e = 0; e < NE; e++)
   // {
   //    const IntegrationRule &ir = qspace->GetElementIntRule(e);
   //    const int nqp = ir.GetNPoints();

   //    double minv = g_min(el_e_idx), maxv = g_max(el_e_idx);
   //    for (int q = 1; q < nqp; q++)
   //    {
   //       minv = fmin(minv, g_min(el_e_idx + q));
   //       maxv = fmax(maxv, g_max(el_e_idx + q));
   //    }

   //    for (int q = 0; q < nqp; q++)
   //    {
   //       g_min(el_e_idx + q) = minv;
   //       g_max(el_e_idx + q) = maxv;
   //    }

   //    el_e_idx += nqp;
   // }
}

} // namespace mfem
