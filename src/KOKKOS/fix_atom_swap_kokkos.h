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

#ifdef FIX_CLASS
// clang-format off
FixStyle(atom/swap/kk,FixAtomSwapKokkos<LMPDeviceType>);
FixStyle(atom/swap/kk/device,FixAtomSwapKokkos<LMPDeviceType>);
FixStyle(atom/swap/kk/host,FixAtomSwapKokkos<LMPHostType>);
// clang-format on
#else
#ifndef LMP_FIX_ATOM_SWAP_KOKKOS_H
#define LMP_FIX_ATOM_SWAP_KOKKOS_H

#include "fix_atom_swap.h"
#include "kokkos_base.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template <class DeviceType> class FixAtomSwapKokkos : public FixAtomSwap, public KokkosBase {
 public:
  FixAtomSwapKokkos(class LAMMPS *, int, char **);
  void init() override;
  void init_list(int, class NeighList *) override;
  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d &, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d &) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 protected:
  PairSwapLocal *local_pair_interface() override;
  void request_swap_list(double) override;
  void pre_exchange_begin_hook() override;
  void pre_exchange_end_hook() override;
  void swap_begin_hooks() override;
  void swap_end_hooks() override;
  void swap_apply_trial_hook(const SwapProposal &, int) override;
  void swap_apply_accepted(int, const SwapProposal *, const int *) override;
  void swap_forward_comm() override;

 private:
  // The scheduler retains its ordinary host list for build_graph().  This
  // adapter passes the separately requested KOKKOS list to the pair style.
  struct LocalInterface : PairSwapLocal {
    PairSwapLocal *pair = nullptr;
    NeighList *list = nullptr;
    SwapLocalCaps swap_local_caps(std::string &reason) const override
    {
      return pair->swap_local_caps(reason);
    }
    void swap_local_begin(NeighList *) override { pair->swap_local_begin(list); }
    int swap_local_delta_energy_batch(int n, const SwapProposal *p, double *e) const override
    {
      return pair->swap_local_delta_energy_batch(n, p, e);
    }
    void swap_local_commit(const SwapProposal &p) override { pair->swap_local_commit(p); }
    void swap_local_commit_batch(int n, const SwapProposal *p, const int *a) override
    {
      pair->swap_local_commit_batch(n, p, a);
    }
    void swap_local_end() override { pair->swap_local_end(); }
  } local_interface;
  NeighList *device_list = nullptr;
  int saved_auto_sync = 0;
  uint64_t read_mask() const;
  uint64_t write_mask() const;
  uint64_t typeq_mask() const;
};

}    // namespace LAMMPS_NS
#endif
#endif
