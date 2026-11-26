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

#include "fix_atom_swap_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "force.h"
#include "kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"

using namespace LAMMPS_NS;

template <class DeviceType>
FixAtomSwapKokkos<DeviceType>::FixAtomSwapKokkos(LAMMPS *lmp, int narg, char **arg) :
    FixAtomSwap(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = static_cast<AtomKokkos *>(atom);
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  // Hooks own the synchronization: the scheduler operates on the host while
  // pair kernels and communication can use another execution space.
  datamask_read = datamask_modify = EMPTY_MASK;
  forward_comm_device = 1;
}

template <class DeviceType> uint64_t FixAtomSwapKokkos<DeviceType>::typeq_mask() const
{
  return TYPE_MASK | (atom->q_flag ? Q_MASK : 0);
}

template <class DeviceType> uint64_t FixAtomSwapKokkos<DeviceType>::write_mask() const
{
  return typeq_mask() | (ke_flag ? V_MASK : 0) | (atom->rmass_flag ? RMASS_MASK : 0);
}

template <class DeviceType> uint64_t FixAtomSwapKokkos<DeviceType>::read_mask() const
{
  return X_MASK | TYPE_MASK | TAG_MASK | MASK_MASK | (atom->q_flag ? Q_MASK : 0) |
      (ke_flag ? V_MASK : 0) | (atom->rmass_flag ? RMASS_MASK : 0);
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::init()
{
  atomKK->sync(Host, read_mask());
  // Identical requests retain their lists across consecutive runs.  Only
  // init_list() replaces the saved pointer when Neighbor creates a new list.
  FixAtomSwap::init();
}

template <class DeviceType> PairSwapLocal *FixAtomSwapKokkos<DeviceType>::local_pair_interface()
{
  local_interface.pair = force->pair->swap_local();
  return local_interface.pair ? &local_interface : nullptr;
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::request_swap_list(double cut)
{
  // A host list with only rmax is sufficient for graph construction.  Avoid
  // copying a potentially much larger device dependency list back to the host.
  if (rmax > 0.0) {
    auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
    const double cutcomm = MAX(comm->cutghostuser, force->pair->cutforce + neighbor->skin);
    req->set_cutoff_fixed(MIN(rmax * (1.0 + 1.0e-9), MAX(rmax, cutcomm)));
  }
  if (use_local) {
    auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
    req->set_id(1);
    req->set_cutoff_fixed(cut);
    if (force->pair->kokkosable) {
      req->set_kokkos_device(force->pair->execution_space == Device);
      req->set_kokkos_host(force->pair->execution_space == HostKK);
    }
  }
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::init_list(int id, NeighList *ptr)
{
  if (id == 1) {
    device_list = ptr;
    local_interface.list = ptr;
    if (rmax <= 0.0) swap_list = ptr;
  } else
    swap_list = ptr;
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::pre_exchange_begin_hook()
{
  atomKK->sync(Host, read_mask());
  saved_auto_sync = lmp->kokkos->auto_sync;
  // Migration and the inherited full-energy fallback use raw host arrays.
  lmp->kokkos->auto_sync = 1;
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::swap_begin_hooks()
{
  atomKK->sync(Host, read_mask());
  if (device_list != swap_list) neighbor->build_one(device_list);
  if (force->pair && force->pair->kokkosable) lmp->kokkos->auto_sync = 0;
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::swap_end_hooks()
{
  atomKK->sync(Host, write_mask());
  lmp->kokkos->auto_sync = 1;
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::pre_exchange_end_hook()
{
  atomKK->modified(Host, write_mask());
  atomKK->sync(execution_space, write_mask());
  lmp->kokkos->auto_sync = saved_auto_sync;
}

template <class DeviceType>
void FixAtomSwapKokkos<DeviceType>::swap_apply_trial_hook(const SwapProposal &p, int /*flag*/)
{
  const uint64_t mask = TYPE_MASK | (p.qflag ? Q_MASK : 0);
  atomKK->modified(Host, mask);
  atomKK->sync(Device, mask);
  atomKK->sync(HostKK, mask);
}

template <class DeviceType>
void FixAtomSwapKokkos<DeviceType>::swap_apply_accepted(int n, const SwapProposal *p, const int *a)
{
  FixAtomSwap::swap_apply_accepted(n, p, a);
  atomKK->modified(Host, write_mask());
}

template <class DeviceType> void FixAtomSwapKokkos<DeviceType>::swap_forward_comm()
{
  if (comm_mode == COMM_FLAG) {
    atomKK->sync(Host, read_mask());
    const int saved = forward_comm_device;
    forward_comm_device = 0;
    comm->forward_comm(this);
    forward_comm_device = saved;
    return;
  }
  atomKK->sync(Host, typeq_mask());
  atomKK->modified(Host, typeq_mask());
  comm->forward_comm(this);
  atomKK->sync(Host, typeq_mask());
  // The pair style can be /kk/host even when the fix uses the device.
  atomKK->sync(Device, typeq_mask());
  atomKK->sync(HostKK, typeq_mask());
}

template <class DeviceType>
int FixAtomSwapKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf, int pbc_flag,
                                                     int *pbc)
{
  if (comm_mode == COMM_TYPEQ) atomKK->sync(Host, typeq_mask());
  return FixAtomSwap::pack_forward_comm(n, list, buf, pbc_flag, pbc);
}

template <class DeviceType>
void FixAtomSwapKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  FixAtomSwap::unpack_forward_comm(n, first, buf);
  if (comm_mode == COMM_TYPEQ) atomKK->modified(Host, typeq_mask());
}

template <class DeviceType>
int FixAtomSwapKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d send,
                                                            DAT::tdual_double_1d &buf,
                                                            int /*pbc_flag*/, int * /*pbc*/)
{
  // auto_sync also covers legacy atom/swap's direct host trial writes.
  atomKK->sync(execution_space, typeq_mask());
  const auto list = send.template view<DeviceType>();
  const auto buffer = buf.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto q = atomKK->k_q.template view<DeviceType>();
  const int qflag = atom->q_flag;
  Kokkos::parallel_for(
      "atom/swap:pack types", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) {
        const int j = list(i);
        buffer(i * (1 + qflag)) = type(j);
        if (qflag) buffer(i * 2 + 1) = q(j);
      });
  return n * (1 + qflag);
}

template <class DeviceType>
void FixAtomSwapKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first,
                                                               DAT::tdual_double_1d &buf)
{
  const auto buffer = buf.template view<DeviceType>();
  const auto type = atomKK->k_type.template view<DeviceType>();
  const auto q = atomKK->k_q.template view<DeviceType>();
  const int qflag = atom->q_flag;
  Kokkos::parallel_for(
      "atom/swap:unpack types", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int i) {
        type(i + first) = static_cast<int>(buffer(i * (1 + qflag)));
        if (qflag) q(i + first) = buffer(i * 2 + 1);
      });
  atomKK->modified(execution_space, typeq_mask());
}

namespace LAMMPS_NS {
template class FixAtomSwapKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixAtomSwapKokkos<LMPHostType>;
#endif
}    // namespace LAMMPS_NS
