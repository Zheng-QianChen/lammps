/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_eam_alloy_kokkos.h"

#include <typeinfo>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template <class DeviceType>
PairEAMAlloyKokkos<DeviceType>::PairEAMAlloyKokkos(LAMMPS *lmp) : PairEAMKokkos<DeviceType>(lmp)
{
  this->fileformat = PairEAM::SETFL;
  this->one_coeff = 1;
}

/* ----------------------------------------------------------------------
   local energy changes for atom type swaps, see pair_swap_local.h
   uses the PairEAMKokkos implementation.  only enabled for this exact class,
   derived classes must opt in explicitly.
------------------------------------------------------------------------- */

template <class DeviceType> PairSwapLocal *PairEAMAlloyKokkos<DeviceType>::swap_local()
{
  if (typeid(*this) != typeid(PairEAMAlloyKokkos<DeviceType>)) return nullptr;
  return this;
}

namespace LAMMPS_NS {
template class PairEAMAlloyKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairEAMAlloyKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
