/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut,PairLJCut);
// clang-format on
#else

#ifndef LMP_PAIR_LJ_CUT_H
#define LMP_PAIR_LJ_CUT_H

#include "pair.h"
#include "pair_swap_local.h"

namespace LAMMPS_NS {

class PairLJCut : public Pair, public PairSwapLocal {
 public:
  PairLJCut(class LAMMPS *);
  ~PairLJCut() override;
  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void born_matrix(int, int, int, int, double, double, double, double &, double &) override;
  void *extract(const char *, int &) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;

  // local energy changes for atom type swaps

  PairSwapLocal *swap_local() override;
  SwapLocalCaps swap_local_caps(std::string &) const override;
  void swap_local_begin(class NeighList *) override;
  int swap_local_delta_energy_batch(int, const SwapProposal *, double *) const override;
  void swap_local_commit(const SwapProposal &) override {}
  void swap_local_end() override;

 protected:
  double cut_global;
  double **cut;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double *cut_respa;

  // data for local swap energies, only pointers so that copies made by derived
  // accelerator styles stay cheap

  class NeighList *swaplocal_list;              // full neighbor list
  class SwapLocalCenters *swaplocal_centers;    // owned atoms near owned or ghost atoms

  virtual void allocate();
  double swaplocal_pair(int, int, double) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
