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

#include "pair_swap_local.h"

#include "atom.h"
#include "neigh_list.h"

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   build reverse lists: for every ghost atom the owned atoms that have it
   in their full neighbor list.  the reverse lists of owned atoms are not
   stored, since a full list with a uniform cutoff is symmetric for pairs
   of owned atoms.
------------------------------------------------------------------------- */

int SwapLocalCenters::setup(Atom *atom, NeighList *list)
{
  clear();
  if (!list || (list->inum != atom->nlocal)) return 1;

  nlocal = atom->nlocal;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  const int nghost = atom->nghost;

  gfirst.assign(nghost + 1, 0);
  for (int c = 0; c < nlocal; c++) {
    const int *jlist = firstneigh[c];
    const int jnum = numneigh[c];
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (j >= nlocal) gfirst[j - nlocal + 1]++;
    }
  }
  for (int g = 0; g < nghost; g++) gfirst[g + 1] += gfirst[g];

  gcenter.resize(gfirst[nghost]);
  std::vector<int> next(gfirst.begin(), gfirst.end() - 1);
  for (int c = 0; c < nlocal; c++) {
    const int *jlist = firstneigh[c];
    const int jnum = numneigh[c];
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      if (j >= nlocal) gcenter[next[j - nlocal]++] = c;
    }
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

void SwapLocalCenters::clear()
{
  nlocal = 0;
  numneigh = nullptr;
  firstneigh = nullptr;
  gfirst.clear();
  gcenter.clear();
}
