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
   Contributing authors: Stephen Foiles (SNL), Murray Daw (SNL)
------------------------------------------------------------------------- */

#include "pair_eam_alloy.h"

#include <typeinfo>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairEAMAlloy::PairEAMAlloy(LAMMPS *lmp) : PairEAM(lmp)
{
  fileformat = SETFL;
  one_coeff = 1;
}

/* ----------------------------------------------------------------------
   local energy changes for atom type swaps, see pair_swap_local.h
   uses the PairEAM implementation.  only enabled for this exact class,
   derived classes must opt in explicitly.
------------------------------------------------------------------------- */

PairSwapLocal *PairEAMAlloy::swap_local()
{
  if (typeid(*this) != typeid(PairEAMAlloy)) return nullptr;
  return this;
}
