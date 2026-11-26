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
PairStyle(meam/kk,PairMEAMKokkos<LMPDeviceType>);
PairStyle(meam/kk/device,PairMEAMKokkos<LMPDeviceType>);
PairStyle(meam/kk/host,PairMEAMKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_MEAM_KOKKOS_H
#define LMP_PAIR_MEAM_KOKKOS_H

#include "kokkos_base.h"
#include "pair_kokkos.h"
#include "pair_meam.h"
#include "meam_kokkos.h"
#include "meam_swap_local_kokkos.h"
#include "neigh_list_kokkos.h"
#include "pair_swap_local_kokkos.h"

namespace LAMMPS_NS {

struct TagPairMEAMNeighStrip{};
struct TagPairMEAMOffsets{};
struct TagPairMEAMPackForwardComm{};
struct TagPairMEAMUnpackForwardComm{};
struct TagPairMEAMPackReverseComm{};
struct TagPairMEAMUnpackReverseComm{};

template<class DeviceType>
class MEAMKokkos;

template<class DeviceType>
class PairMEAMKokkos : public PairMEAM, public KokkosBase {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef int value_type;

  PairMEAMKokkos(class LAMMPS *);
  ~PairMEAMKokkos() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void reinit() override;

  PairSwapLocal *swap_local() override;
  SwapLocalCaps swap_local_caps(std::string &) const override;
  void swap_local_begin(class NeighList *) override;
  int swap_local_delta_energy_batch(int, const SwapProposal *, double *) const override;
  void swap_local_commit(const SwapProposal &) override;
  void swap_local_commit_batch(int, const SwapProposal *, const int *) override;
  void swap_local_end() override;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMPackForwardComm, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMUnpackForwardComm, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMPackReverseComm, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMUnpackReverseComm, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMNeighStrip,  const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagPairMEAMOffsets,  const int, int&) const;

  int pack_forward_comm_kokkos(int, DAT::tdual_int_1d, DAT::tdual_double_1d&,
                               int, int *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_reverse_comm_kokkos(int, int, DAT::tdual_double_1d&) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm_kokkos(int, DAT::tdual_int_1d,
                                  DAT::tdual_double_1d&) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  class MEAMKokkos<DeviceType> *meam_inst_kk;
  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_int_1d type;

  DAT::ttransform_kkacc_1d k_eatom;
  DAT::ttransform_kkacc_1d_6 k_vatom;
  typename AT::t_kkacc_1d d_eatom;
  typename AT::t_kkacc_1d_6 d_vatom;

  typename AT::t_int_1d d_offset;

  DAT::tdual_int_1d k_map;
  typename AT::t_int_1d d_map;
  typename AT::t_kkfloat_2d d_scale;
  typename AT::t_int_1d d_ilist_half;
  typename AT::t_int_1d d_numneigh_half;
  typename AT::t_neighbors_2d d_neighbors_half;
  typename AT::t_int_1d d_numneigh_full;
  typename AT::t_neighbors_2d d_neighbors_full;
  typename AT::t_int_1d d_sendlist;
  typename AT::t_double_1d_um v_buf;

  int first;
  int neighflag,nlocal,nall,eflag,vflag;

  typename AT::t_kkfloat_1d d_rho, d_rho0, d_rho1, d_rho2, d_rho3, d_frhop;
  typename AT::t_kkfloat_1d d_gamma, d_dgamma1, d_dgamma2, d_dgamma3, d_arho2b;
  typename AT::t_kkfloat_2d d_arho1, d_arho2, d_arho3, d_arho3b, d_t_ave, d_tsq_ave;
  // msmeam params
  typename AT::t_kkfloat_1d d_arho2mb;
  typename AT::t_kkfloat_2d d_arho1m, d_arho2m, d_arho3m, d_arho3mb;

  NeighListKokkos<DeviceType> *swap_list_kk = nullptr;
  SwapLocalCentersKokkos<DeviceType> swap_centers_kk;
  mutable SwapBatchImagesKokkos<DeviceType> swap_images_kk;
  mutable SwapLocalAffectedKokkos<DeviceType> swap_affected_kk;
  mutable Kokkos::View<double **, DeviceType> d_swap_dens;
  mutable typename AT::t_int_1d d_swap_dens_valid;
  mutable Kokkos::View<double **, DeviceType> d_swap_ddens;
  mutable Kokkos::View<double *, DeviceType> d_swap_energy;
  mutable Kokkos::View<double *, LMPPinnedHostType> h_swap_energy;
  mutable Kokkos::View<int *, DeviceType> d_swap_accepted;
  mutable Kokkos::View<int *, LMPPinnedHostType> h_swap_accepted;
  mutable Kokkos::View<int, DeviceType> d_swap_error;
  Kokkos::View<MeamSwapParams, DeviceType> d_swap_params;
  mutable double swap_rdepsq = 0.0;
  mutable int swap_eval_valid = 0;
  mutable int swap_eval_n = 0;
  mutable Kokkos::View<SwapProposal *, LMPPinnedHostType> swap_eval_proposals;

  void update_meam_views();
  void swap_ensure_dens(int) const;

  friend void pair_virial_fdotr_compute<PairMEAMKokkos>(PairMEAMKokkos*);
};

}
#endif
#endif

