#include "remhos_lvpp.hpp"

namespace mfem
{
void L2obj::ConstructGradient()
{
   // Construct the gradient operator
   block_offsets.SetSize(gfs.size() + 1);
   block_true_offsets.SetSize(gfs.size() + 1);
   for (size_t i=0; i<gfs.size(); i++)
   {
      block_offsets[i] = i*gfs[i]->Size();
      tvecs.push_back(gfs[i]->GetTrueDofs());
      block_true_offsets[i] = i*tvecs[i]->Size();
   }
   gradient.reset(new L2grad(block_offsets, block_offsets));
   for (size_t i=0; i<gfs.size(); i++)
   {
      ParBilinearForm *grad = new ParBilinearForm(fes[i]);
      gradient->SetBlock(i, i, grad);
      gradient->SetBlockCoef(i, i, 0.5);
      grad->AddDomainIntegrator(new MassIntegrator());
      grad->Assemble();
   }
}
}
