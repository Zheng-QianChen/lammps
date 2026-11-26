// clang-format off
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

#include "pair_lj_cut_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "tune_kokkos.h"
#include "update.h"

#include <typeinfo>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutKokkos<DeviceType>::PairLJCutKokkos(LAMMPS *lmp) : PairLJCut(lmp)
{
  respa_enable = 0;
  tuner = nullptr;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutKokkos<DeviceType>::~PairLJCutKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }

  delete tuner;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = static_cast<KK_FLOAT>(force->special_lj[0]);
  special_lj[1] = static_cast<KK_FLOAT>(force->special_lj[1]);
  special_lj[2] = static_cast<KK_FLOAT>(force->special_lj[2]);
  special_lj[3] = static_cast<KK_FLOAT>(force->special_lj[3]);

  // loop over neighbors of my atoms

  copymode = 1;

  if (lmp->kokkos->autotuning && tuner) tuner->tuning_kernel_params();

  EV_FLOAT ev = pair_compute<PairLJCutKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);

  if (eflag_global) eng_vdwl += static_cast<double>(ev.evdwl);
  if (vflag_global) {
    virial[0] += static_cast<double>(ev.v[0]);
    virial[1] += static_cast<double>(ev.v[1]);
    virial[2] += static_cast<double>(ev.v[2]);
    virial[3] += static_cast<double>(ev.v[3]);
    virial[4] += static_cast<double>(ev.v[4]);
    virial[5] += static_cast<double>(ev.v[5]);
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLJCutKokkos<DeviceType>::
compute_fpair(const KK_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {

  const KK_FLOAT r2inv = static_cast<KK_FLOAT>(1.0) / rsq;
  const KK_FLOAT r6inv = r2inv*r2inv*r2inv;

  KK_FLOAT lj1, lj2;
  if constexpr (STACKPARAMS) {
    lj1 = m_params[itype][jtype].lj1;
    lj2 = m_params[itype][jtype].lj2;
  } else {
    lj1 = params(itype,jtype).lj1;
    lj2 = params(itype,jtype).lj2;
  }

  return r6inv * Kokkos::fma(lj1, r6inv, -lj2) * r2inv;

}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairLJCutKokkos<DeviceType>::
compute_evdwl(const KK_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {

  const KK_FLOAT r2inv = static_cast<KK_FLOAT>(1.0) / rsq;
  const KK_FLOAT r6inv = r2inv*r2inv*r2inv;

  KK_FLOAT lj3, lj4, offset;
  if constexpr (STACKPARAMS) {
    lj3 = m_params[itype][jtype].lj3;
    lj4 = m_params[itype][jtype].lj4;
    offset = m_params[itype][jtype].offset;
  } else {
    lj3 = params(itype,jtype).lj3;
    lj4 = params(itype,jtype).lj4;
    offset = params(itype,jtype).offset;
  }

  return Kokkos::fma(r6inv, Kokkos::fma(lj3, r6inv, -lj4), -offset);

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::allocate()
{
  PairLJCut::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType>("PairLJCut::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::init_style()
{
  PairLJCut::init_style();

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();

  if (lmp->kokkos->autotuning > 0 && !tuner) {
    if (!force->newton_pair)
      tuner = new TuneKokkos(lmp, TuneKokkos::PAIR, lmp->kokkos->autotuning,
        2, "pair-lj-cut");
    else
      error->warning(FLERR,"Autotuner for lj/cut/kk is disabled with 'newton on'");
  }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairLJCutKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairLJCut::init_one(i,j);

  k_params.view_host()(i,j).lj1 = static_cast<KK_FLOAT>(lj1[i][j]);
  k_params.view_host()(i,j).lj2 = static_cast<KK_FLOAT>(lj2[i][j]);
  k_params.view_host()(i,j).lj3 = static_cast<KK_FLOAT>(lj3[i][j]);
  k_params.view_host()(i,j).lj4 = static_cast<KK_FLOAT>(lj4[i][j]);
  k_params.view_host()(i,j).offset = static_cast<KK_FLOAT>(offset[i][j]);
  k_params.view_host()(i,j).cutsq = static_cast<KK_FLOAT>(cutone*cutone);
  k_params.view_host()(j,i) = k_params.view_host()(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.view_host()(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = static_cast<KK_FLOAT>(cutone*cutone);
  }

  k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = cutone*cutone;
  k_cutsq.modify_host();
  k_params.modify_host();

  return cutone;
}

template <class DeviceType> PairSwapLocal *PairLJCutKokkos<DeviceType>::swap_local()
{ return (typeid(*this) == typeid(PairLJCutKokkos<DeviceType>)) ? this : nullptr; }

template <class DeviceType>
SwapLocalCaps PairLJCutKokkos<DeviceType>::swap_local_caps(std::string &reason) const
{
  SwapLocalCaps caps = {1, 1, cutforce, cutforce};
  if (typeid(*this) != typeid(PairLJCutKokkos<DeviceType>)) {
    caps.supported = 0;
    reason = "derived pair style has not been validated for local swap energies";
  } else if (!allocated || (cutforce <= 0.0)) {
    caps.supported = 0;
    reason = "pair coefficients are not initialized";
  } else if (tail_flag) {
    caps.supported = 0;
    reason = "pair_modify tail yes is not supported";
  }
  return caps;
}

template <class DeviceType> void PairLJCutKokkos<DeviceType>::swap_local_begin(NeighList *full_list)
{
  swap_list_kk = nullptr;
  // A host fix can query this interface too.  Decline a list from another
  // memory space so the caller can fall back to full energy evaluations.
  if (!full_list || !full_list->kokkos || (full_list->execution_space != execution_space)) return;
  swap_list_kk = static_cast<NeighListKokkos<DeviceType> *>(full_list);
  if (swap_centers_kk.setup(atom, swap_list_kk)) swap_list_kk = nullptr;
  atomKK->sync(execution_space, X_MASK | TYPE_MASK | TAG_MASK);
  k_params.template sync<DeviceType>();
}

template <class DeviceType> void PairLJCutKokkos<DeviceType>::swap_local_end()
{ swap_list_kk = nullptr; }

template <class DeviceType>
int PairLJCutKokkos<DeviceType>::swap_local_delta_energy_batch(int n, const SwapProposal *proposals,
                                                               double *dE) const
{
  if (!swap_list_kk) return 1;
  if (!n) return 0;
  atomKK->sync(execution_space, X_MASK | TYPE_MASK | TAG_MASK);
  const auto sx = atomKK->k_x.template view<DeviceType>();
  const auto stype = atomKK->k_type.template view<DeviceType>();
  const auto stag = atomKK->k_tag.template view<DeviceType>();
  swap_images_kk.stage(atom, n, proposals);
  if ((int) d_swap_energy.extent(0) < n) {
    d_swap_energy = decltype(d_swap_energy)("lj/cut:swap energy", n + n / 4 + 1);
    h_swap_energy = decltype(h_swap_energy)("lj/cut:host swap energy", d_swap_energy.extent(0));
  }
  const auto energy = d_swap_energy;
  const auto props = swap_images_kk.d_proposals;
  const auto offset = swap_images_kk.d_img_offset;
  const auto images = swap_images_kk.d_img_atom;
  const auto slot = swap_images_kk.d_img_slot;
  const auto first = swap_centers_kk.d_gfirst;
  const auto gcenter = swap_centers_kk.d_gcenter;
  const auto num = swap_list_kk->d_numneigh;
  const auto neighbors = swap_list_kk->d_neighbors;
  const auto coeff = k_params.template view<DeviceType>();
  const int owned = atom->nlocal;
  // Pairwise independent edits can share an unedited site.  Sum only the
  // changed pair terms of each proposal, so such sites need no shared stamp.
  Kokkos::parallel_for(
      "lj/cut:swap owned energies", Kokkos::RangePolicy<DeviceType>(0, n), KOKKOS_LAMBDA(int k) {
        const SwapProposal p = props(k);
        double de = 0.0;
        for (int im = offset(k); im < offset(k + 1); ++im) {
          const int a = images(im);
          const int m = slot(im);
          const int told = p.oldtype[m];
          const int tnew = p.newtype[m];
          if (a < owned) {
            for (int jj = 0; jj < num(a); ++jj) {
              const int j = neighbors(a, jj) & NEIGHMASK;
              const double dx = sx(a, 0) - sx(j, 0);
              const double dy = sx(a, 1) - sx(j, 1);
              const double dz = sx(a, 2) - sx(j, 2);
              const double rsq = dx * dx + dy * dy + dz * dz;
              const int jo = swap_kk_oldtype(p, stag(j), stype(j));
              const int jn = swap_kk_newtype(p, stag(j), stype(j));
              de += 0.5 * (coeff(tnew, jn).swap_energy(rsq) - coeff(told, jo).swap_energy(rsq));
            }
          }
          const int start = (a < owned) ? 0 : first(a - owned);
          const int end = (a < owned) ? num(a) : first(a - owned + 1);
          for (int jj = start; jj < end; ++jj) {
            const int c = (a < owned) ? (neighbors(a, jj) & NEIGHMASK) : gcenter(jj);
            if ((c >= owned) || (swap_kk_which(p, stag(c)) >= 0)) continue;
            const double dx = sx(c, 0) - sx(a, 0);
            const double dy = sx(c, 1) - sx(a, 1);
            const double dz = sx(c, 2) - sx(a, 2);
            const double rsq = dx * dx + dy * dy + dz * dz;
            const int tc = stype(c);
            de += 0.5 * (coeff(tc, tnew).swap_energy(rsq) - coeff(tc, told).swap_energy(rsq));
          }
        }
        energy(k) = de;
      });
  Kokkos::deep_copy(Kokkos::subview(h_swap_energy, Kokkos::pair<int, int>(0, n)),
                    Kokkos::subview(d_swap_energy, Kokkos::pair<int, int>(0, n)));
  for (int k = 0; k < n; ++k) dE[k] = h_swap_energy(k);
  return 0;
}

namespace LAMMPS_NS {
template class PairLJCutKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCutKokkos<LMPHostType>;
#endif
}

