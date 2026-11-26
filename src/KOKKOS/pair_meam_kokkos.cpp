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

/* ----------------------------------------------------------------------
   Contributing authors: Naga Vydyanathan (NVIDIA), Stan Moore (SNL)
------------------------------------------------------------------------- */

#include "pair_meam_kokkos.h"
#include "meam_impl_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cmath>
#include <typeinfo>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMEAMKokkos<DeviceType>::PairMEAMKokkos(LAMMPS *lmp) : PairMEAM(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  reverse_comm_device = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  delete meam_inst;
  meam_inst_kk = new MEAMKokkos<DeviceType>(memory);
  meam_inst = meam_inst_kk;
  myname = "meam/kk";
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairMEAMKokkos<DeviceType>::~PairMEAMKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
  delete meam_inst_kk;
  meam_inst = nullptr;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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

  // neighbor list info

  int inum_half = listhalf->inum;
  NeighListKokkos<DeviceType>* k_halflist = static_cast<NeighListKokkos<DeviceType>*>(listhalf);
  d_ilist_half = k_halflist->d_ilist;
  d_numneigh_half = k_halflist->d_numneigh;
  d_neighbors_half = k_halflist->d_neighbors;

  NeighListKokkos<DeviceType>* k_fulllist = static_cast<NeighListKokkos<DeviceType>*>(listfull);
  d_numneigh_full = k_fulllist->d_numneigh;
  d_neighbors_full = k_fulllist->d_neighbors;

  EV_FLOAT ev;

  copymode = 1;
  meam_inst_kk->copymode = 1;

  // strip neighbor lists of any special bond flags before using with MEAM
  // necessary before doing neigh_f2c and neigh_c2f conversions each step

  if (neighbor->ago == 0)
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairMEAMNeighStrip >(0,inum_half),*this);

  // check size of scrfcn based on half neighbor list

  nlocal = atom->nlocal;
  nall = nlocal + atom->nghost;

  int n = 0;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagPairMEAMOffsets>(0,inum_half),*this,n);

  meam_inst_kk->meam_dens_setup(atom->nmax, nall, n);
  update_meam_views();

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  atomKK->sync(execution_space,datamask_read);

  int ntype = atom->ntypes;

  // 3 stages of MEAM calculation
  // loop over my atoms followed by communication

  int errorflag = 0;

  d_offset = typename AT::t_int_1d("pair:offset",inum_half+1);
  {
    // local variables for lambda capture

    auto l_ilist_half = d_ilist_half;
    auto l_numneigh_half = d_numneigh_half;
    auto l_offset = d_offset;

    Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType>(0,inum_half), LAMMPS_LAMBDA(int ii, int &m_fill, bool final) {
      int i = l_ilist_half[ii];
      m_fill += l_numneigh_half[i];
      if (final)
        l_offset[ii+1] = m_fill;
    });
  }

  int need_dup = lmp->kokkos->need_dup<DeviceType>();

  meam_inst_kk->meam_dens_init(inum_half,ntype,type,d_map,x,d_numneigh_half,d_numneigh_full,
    d_ilist_half,d_neighbors_half, d_neighbors_full, d_offset, neighflag, need_dup);

  meam_inst_kk->k_rho0.template modify<DeviceType>();
  meam_inst_kk->k_arho2b.template modify<DeviceType>();
  meam_inst_kk->k_arho1.template modify<DeviceType>();
  meam_inst_kk->k_arho2.template modify<DeviceType>();
  meam_inst_kk->k_arho3.template modify<DeviceType>();
  meam_inst_kk->k_arho3b.template modify<DeviceType>();
  meam_inst_kk->k_t_ave.template modify<DeviceType>();
  meam_inst_kk->k_tsq_ave.template modify<DeviceType>();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.template modify<DeviceType>();
    meam_inst_kk->k_arho1m.template modify<DeviceType>();
    meam_inst_kk->k_arho2m.template modify<DeviceType>();
    meam_inst_kk->k_arho3m.template modify<DeviceType>();
    meam_inst_kk->k_arho3mb.template modify<DeviceType>();
  }

  comm->reverse_comm(this);

  meam_inst_kk->k_rho0.template sync<DeviceType>();
  meam_inst_kk->k_arho2b.template sync<DeviceType>();
  meam_inst_kk->k_arho1.template sync<DeviceType>();
  meam_inst_kk->k_arho2.template sync<DeviceType>();
  meam_inst_kk->k_arho3.template sync<DeviceType>();
  meam_inst_kk->k_arho3b.template sync<DeviceType>();
  meam_inst_kk->k_t_ave.template sync<DeviceType>();
  meam_inst_kk->k_tsq_ave.template sync<DeviceType>();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.template sync<DeviceType>();
    meam_inst_kk->k_arho1m.template sync<DeviceType>();
    meam_inst_kk->k_arho2m.template sync<DeviceType>();
    meam_inst_kk->k_arho3m.template sync<DeviceType>();
    meam_inst_kk->k_arho3mb.template sync<DeviceType>();
  }

  meam_inst_kk->meam_dens_final(nlocal,eflag_either,eflag_global,eflag_atom,
                   d_eatom,ntype,type,d_map,d_scale,errorflag,ev);

  if (errorflag)
    error->one(FLERR,"MEAM library error {}",errorflag);

  meam_inst_kk->k_rho0.template modify<DeviceType>();
  meam_inst_kk->k_rho1.template modify<DeviceType>();
  meam_inst_kk->k_rho2.template modify<DeviceType>();
  meam_inst_kk->k_rho3.template modify<DeviceType>();
  meam_inst_kk->k_frhop.template modify<DeviceType>();
  meam_inst_kk->k_gamma.template modify<DeviceType>();
  meam_inst_kk->k_dgamma1.template modify<DeviceType>();
  meam_inst_kk->k_dgamma2.template modify<DeviceType>();
  meam_inst_kk->k_dgamma3.template modify<DeviceType>();
  meam_inst_kk->k_arho2b.template modify<DeviceType>();
  meam_inst_kk->k_arho1.template modify<DeviceType>();
  meam_inst_kk->k_arho2.template modify<DeviceType>();
  meam_inst_kk->k_arho3.template modify<DeviceType>();
  meam_inst_kk->k_arho3b.template modify<DeviceType>();
  meam_inst_kk->k_t_ave.template modify<DeviceType>();
  meam_inst_kk->k_tsq_ave.template modify<DeviceType>();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.template modify<DeviceType>();
    meam_inst_kk->k_arho1m.template modify<DeviceType>();
    meam_inst_kk->k_arho2m.template modify<DeviceType>();
    meam_inst_kk->k_arho3m.template modify<DeviceType>();
    meam_inst_kk->k_arho3mb.template modify<DeviceType>();
  }

  comm->forward_comm(this);

  meam_inst_kk->k_rho0.template sync<DeviceType>();
  meam_inst_kk->k_rho1.template sync<DeviceType>();
  meam_inst_kk->k_rho2.template sync<DeviceType>();
  meam_inst_kk->k_rho3.template sync<DeviceType>();
  meam_inst_kk->k_frhop.template sync<DeviceType>();
  meam_inst_kk->k_gamma.template sync<DeviceType>();
  meam_inst_kk->k_dgamma1.template sync<DeviceType>();
  meam_inst_kk->k_dgamma2.template sync<DeviceType>();
  meam_inst_kk->k_dgamma3.template sync<DeviceType>();
  meam_inst_kk->k_arho2b.template sync<DeviceType>();
  meam_inst_kk->k_arho1.template sync<DeviceType>();
  meam_inst_kk->k_arho2.template sync<DeviceType>();
  meam_inst_kk->k_arho3.template sync<DeviceType>();
  meam_inst_kk->k_arho3b.template sync<DeviceType>();
  meam_inst_kk->k_t_ave.template sync<DeviceType>();
  meam_inst_kk->k_tsq_ave.template sync<DeviceType>();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.template sync<DeviceType>();
    meam_inst_kk->k_arho1m.template sync<DeviceType>();
    meam_inst_kk->k_arho2m.template sync<DeviceType>();
    meam_inst_kk->k_arho3m.template sync<DeviceType>();
    meam_inst_kk->k_arho3mb.template sync<DeviceType>();
  }

  meam_inst_kk->meam_force(inum_half,eflag_global,eflag_atom,vflag_global,
                           vflag_atom,d_eatom,ntype,type,d_map,x,
                           d_numneigh_half, d_numneigh_full,f,d_vatom,
                           d_ilist_half, d_offset, d_neighbors_half, d_neighbors_full,
                           neighflag, need_dup, ev);

  if (eflag_global) eng_vdwl += static_cast<double>(ev.evdwl);
  if (vflag_global) {
    virial[0] += static_cast<double>(ev.v[0]);
    virial[1] += static_cast<double>(ev.v[1]);
    virial[2] += static_cast<double>(ev.v[2]);
    virial[3] += static_cast<double>(ev.v[3]);
    virial[4] += static_cast<double>(ev.v[4]);
    virial[5] += static_cast<double>(ev.v[5]);
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);
  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  copymode = 0;
  meam_inst_kk->copymode = 0;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */
template<class DeviceType>
void PairMEAMKokkos<DeviceType>::coeff(int narg, char **arg)
{
  PairMEAM::coeff(narg,arg);

  // sync map and scale

  int n = atom->ntypes;
  MemKK::realloc_kokkos(d_map,"pair:map",n+1);
  MemKK::realloc_kokkos(d_scale,"pair:scale",n+1,n+1);
  auto h_map = Kokkos::create_mirror_view(d_map);
  auto h_scale = Kokkos::create_mirror_view(d_scale);

  // PairMEAM::coeff() only writes the upper triangle of scale[][]; the lower
  // triangle stays uninitialised until init_one() mirrors it.  Seed the whole
  // device copy with the same default instead of reading that memory here, and
  // let init_one() install the real values once they exist

  for (int i = 1; i <= n; i++) {
    h_map[i] = map[i];
    for (int j = 1; j <= n; j++)
      h_scale(i,j) = static_cast<KK_FLOAT>(1.0);
  }

  Kokkos::deep_copy(d_map,h_map);
  Kokkos::deep_copy(d_scale,h_scale);
}

/* ----------------------------------------------------------------------
   Pair::init() calls init_one() for every type pair after coeff() has run, and
   PairMEAM::init_one() is what makes scale[][] symmetric.  Refresh the device
   copy here so it never holds the pre-symmetrised values
------------------------------------------------------------------------- */

template<class DeviceType>
double PairMEAMKokkos<DeviceType>::init_one(int i, int j)
{
  const double cut = PairMEAM::init_one(i,j);

  if (d_scale.data()) {
    const int n = atom->ntypes;
    auto h_scale = Kokkos::create_mirror_view(d_scale);
    for (int ii = 1; ii <= n; ii++)
      for (int jj = 1; jj <= n; jj++)
        h_scale(ii,jj) = scale[ii][jj];
    Kokkos::deep_copy(d_scale,h_scale);
  }

  return cut;
}

/* ----------------------------------------------------------------------
   refresh the device copy of scale[][]

   fix adapt writes scale[][] through PairMEAM::extract() and then calls
   Pair::reinit(), so without this override the device keeps the values
   captured at pair_coeff time and the requested scaling never takes effect
------------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::reinit()
{
  PairMEAM::reinit();

  if (!d_scale.data()) return;

  const int n = atom->ntypes;
  auto h_scale = Kokkos::create_mirror_view(d_scale);
  for (int i = 1; i <= n; i++)
    for (int j = 1; j <= n; j++)
      h_scale(i,j) = scale[i][j];
  Kokkos::deep_copy(d_scale,h_scale);

  // meam_inst_kk->d_scale is refreshed from this view on the next meam_dens_final()
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */
template<class DeviceType>
void PairMEAMKokkos<DeviceType>::init_style()
{
  PairMEAM::init_style();

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this,1);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  request = neighbor->find_request(this,2);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);

  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair meam/kk");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMEAMKokkos<DeviceType>::pack_forward_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf,
                                int /*pbc_flag*/, int * /*pbc*/)
{
  d_sendlist = k_sendlist.view<DeviceType>();
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairMEAMPackForwardComm>(0,n),*this);
  return n*comm_forward;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMPackForwardComm, const int &i) const {
  int j = d_sendlist(i);
  int m = i*comm_forward;
  v_buf[m++] = static_cast<double>(d_rho0[j]);
  v_buf[m++] = static_cast<double>(d_rho1[j]);
  v_buf[m++] = static_cast<double>(d_rho2[j]);
  v_buf[m++] = static_cast<double>(d_rho3[j]);
  v_buf[m++] = static_cast<double>(d_frhop[j]);
  v_buf[m++] = static_cast<double>(d_gamma[j]);
  v_buf[m++] = static_cast<double>(d_dgamma1[j]);
  v_buf[m++] = static_cast<double>(d_dgamma2[j]);
  v_buf[m++] = static_cast<double>(d_dgamma3[j]);
  v_buf[m++] = static_cast<double>(d_arho2b[j]);
  v_buf[m++] = static_cast<double>(d_arho1(j,0));
  v_buf[m++] = static_cast<double>(d_arho1(j,1));
  v_buf[m++] = static_cast<double>(d_arho1(j,2));
  v_buf[m++] = static_cast<double>(d_arho2(j,0));
  v_buf[m++] = static_cast<double>(d_arho2(j,1));
  v_buf[m++] = static_cast<double>(d_arho2(j,2));
  v_buf[m++] = static_cast<double>(d_arho2(j,3));
  v_buf[m++] = static_cast<double>(d_arho2(j,4));
  v_buf[m++] = static_cast<double>(d_arho2(j,5));
  for (int k = 0; k < 10; k++) v_buf[m++] = static_cast<double>(d_arho3(j,k));
  v_buf[m++] = static_cast<double>(d_arho3b(j,0));
  v_buf[m++] = static_cast<double>(d_arho3b(j,1));
  v_buf[m++] = static_cast<double>(d_arho3b(j,2));
  v_buf[m++] = static_cast<double>(d_t_ave(j,0));
  v_buf[m++] = static_cast<double>(d_t_ave(j,1));
  v_buf[m++] = static_cast<double>(d_t_ave(j,2));
  v_buf[m++] = static_cast<double>(d_tsq_ave(j,0));
  v_buf[m++] = static_cast<double>(d_tsq_ave(j,1));
  v_buf[m++] = static_cast<double>(d_tsq_ave(j,2));
  if (msmeamflag) {
    v_buf[m++] = static_cast<double>(d_arho2mb[j]);
    v_buf[m++] = static_cast<double>(d_arho1m(j,0));
    v_buf[m++] = static_cast<double>(d_arho1m(j,1));
    v_buf[m++] = static_cast<double>(d_arho1m(j,2));
    v_buf[m++] = static_cast<double>(d_arho2m(j,0));
    v_buf[m++] = static_cast<double>(d_arho2m(j,1));
    v_buf[m++] = static_cast<double>(d_arho2m(j,2));
    v_buf[m++] = static_cast<double>(d_arho2m(j,3));
    v_buf[m++] = static_cast<double>(d_arho2m(j,4));
    v_buf[m++] = static_cast<double>(d_arho2m(j,5));
    for (int k = 0; k < 10; k++) v_buf[m++] = static_cast<double>(d_arho3m(j,k));
    v_buf[m++] = static_cast<double>(d_arho3mb(j,0));
    v_buf[m++] = static_cast<double>(d_arho3mb(j,1));
    v_buf[m++] = static_cast<double>(d_arho3mb(j,2));
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::unpack_forward_comm_kokkos(int n, int first_in, DAT::tdual_double_1d &buf)
{
  first = first_in;
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairMEAMUnpackForwardComm>(0,n),*this);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMUnpackForwardComm, const int &i) const{
  //int m = i*38;
  int m = i*comm_forward;

    d_rho0[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_rho1[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_rho2[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_rho3[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_frhop[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_gamma[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_dgamma1[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_dgamma2[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_dgamma3[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2b[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,3) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,4) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2(i+first,5) = static_cast<KK_FLOAT>(v_buf[m++]);
    for (int k = 0; k < 10; k++) d_arho3(i+first,k) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3b(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3b(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3b(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_t_ave(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_t_ave(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_t_ave(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_tsq_ave(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_tsq_ave(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
    d_tsq_ave(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    if (msmeamflag) {
      d_arho2mb[i+first] = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho1m(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho1m(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho1m(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,3) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,4) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho2m(i+first,5) = static_cast<KK_FLOAT>(v_buf[m++]);
      for (int k = 0; k < 10; k++) d_arho3m(i+first,k) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho3mb(i+first,0) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho3mb(i+first,1) = static_cast<KK_FLOAT>(v_buf[m++]);
      d_arho3mb(i+first,2) = static_cast<KK_FLOAT>(v_buf[m++]);
    }
 }

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMEAMKokkos<DeviceType>::pack_forward_comm(int n, int *list, double *buf,
                               int /*pbc_flag*/, int * /*pbc*/)
{
  meam_inst_kk->k_rho0.sync_host();
  meam_inst_kk->k_rho1.sync_host();
  meam_inst_kk->k_rho2.sync_host();
  meam_inst_kk->k_rho3.sync_host();
  meam_inst_kk->k_frhop.sync_host();
  meam_inst_kk->k_gamma.sync_host();
  meam_inst_kk->k_dgamma1.sync_host();
  meam_inst_kk->k_dgamma2.sync_host();
  meam_inst_kk->k_dgamma3.sync_host();
  meam_inst_kk->k_arho2b.sync_host();
  meam_inst_kk->k_arho1.sync_host();
  meam_inst_kk->k_arho2.sync_host();
  meam_inst_kk->k_arho3.sync_host();
  meam_inst_kk->k_arho3b.sync_host();
  meam_inst_kk->k_t_ave.sync_host();
  meam_inst_kk->k_tsq_ave.sync_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.sync_host();
    meam_inst_kk->k_arho1m.sync_host();
    meam_inst_kk->k_arho2m.sync_host();
    meam_inst_kk->k_arho3m.sync_host();
    meam_inst_kk->k_arho3mb.sync_host();
  }

  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    buf[m++] = static_cast<double>(meam_inst_kk->h_rho0[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_rho1[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_rho2[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_rho3[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_frhop[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_gamma[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_dgamma1[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_dgamma2[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_dgamma3[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2b[j]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(j,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(j,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(j,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,3));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,4));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(j,5));
    for (int k = 0; k < 10; k++) buf[m++] = static_cast<double>(meam_inst_kk->h_arho3(j,k));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(j,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(j,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(j,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(j,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(j,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(j,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(j,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(j,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(j,2));
    if (msmeamflag) {
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2mb[j]);
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(j,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(j,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(j,2));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,2));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,3));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,4));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(j,5));
      for (int k = 0; k < 10; k++) buf[m++] = static_cast<double>(meam_inst_kk->h_arho3m(j,k));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(j,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(j,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(j,2));
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::unpack_forward_comm(int n, int first, double *buf)
{
  meam_inst_kk->k_rho0.sync_host();
  meam_inst_kk->k_rho1.sync_host();
  meam_inst_kk->k_rho2.sync_host();
  meam_inst_kk->k_rho3.sync_host();
  meam_inst_kk->k_frhop.sync_host();
  meam_inst_kk->k_gamma.sync_host();
  meam_inst_kk->k_dgamma1.sync_host();
  meam_inst_kk->k_dgamma2.sync_host();
  meam_inst_kk->k_dgamma3.sync_host();
  meam_inst_kk->k_arho2b.sync_host();
  meam_inst_kk->k_arho1.sync_host();
  meam_inst_kk->k_arho2.sync_host();
  meam_inst_kk->k_arho3.sync_host();
  meam_inst_kk->k_arho3b.sync_host();
  meam_inst_kk->k_t_ave.sync_host();
  meam_inst_kk->k_tsq_ave.sync_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.sync_host();
    meam_inst_kk->k_arho1m.sync_host();
    meam_inst_kk->k_arho2m.sync_host();
    meam_inst_kk->k_arho3m.sync_host();
    meam_inst_kk->k_arho3mb.sync_host();
  }

  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    meam_inst_kk->h_rho0[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_rho1[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_rho2[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_rho3[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_frhop[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_gamma[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_dgamma1[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_dgamma2[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_dgamma3[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2b[i] = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(i,0) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(i,1) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,0) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,1) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,3) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,4) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(i,5) = static_cast<KK_FLOAT>(buf[m++]);
    for (int k = 0; k < 10; k++) meam_inst_kk->h_arho3(i,k) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(i,0) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(i,1) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(i,0) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(i,1) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(i,0) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(i,1) = static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    if (msmeamflag) {
      meam_inst_kk->h_arho2mb[i] = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(i,0) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(i,1) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(i,2) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,0) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,1) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,2) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,3) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,4) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(i,5) = static_cast<KK_FLOAT>(buf[m++]);
      for (int k = 0; k < 10; k++) meam_inst_kk->h_arho3m(i,k) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(i,0) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(i,1) = static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(i,2) = static_cast<KK_FLOAT>(buf[m++]);
    }
  }

  meam_inst_kk->k_rho0.modify_host();
  meam_inst_kk->k_rho1.modify_host();
  meam_inst_kk->k_rho2.modify_host();
  meam_inst_kk->k_rho3.modify_host();
  meam_inst_kk->k_frhop.modify_host();
  meam_inst_kk->k_gamma.modify_host();
  meam_inst_kk->k_dgamma1.modify_host();
  meam_inst_kk->k_dgamma2.modify_host();
  meam_inst_kk->k_dgamma3.modify_host();
  meam_inst_kk->k_arho2b.modify_host();
  meam_inst_kk->k_arho1.modify_host();
  meam_inst_kk->k_arho2.modify_host();
  meam_inst_kk->k_arho3.modify_host();
  meam_inst_kk->k_arho3b.modify_host();
  meam_inst_kk->k_t_ave.modify_host();
  meam_inst_kk->k_tsq_ave.modify_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.modify_host();
    meam_inst_kk->k_arho1m.modify_host();
    meam_inst_kk->k_arho2m.modify_host();
    meam_inst_kk->k_arho3m.modify_host();
    meam_inst_kk->k_arho3mb.modify_host();
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMEAMKokkos<DeviceType>::pack_reverse_comm_kokkos(int n, int first_in, DAT::tdual_double_1d &buf)
{
  first = first_in;
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairMEAMPackReverseComm>(0,n),*this);
  //return n*30;
  return n*comm_reverse;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMPackReverseComm, const int &i) const {
  //int m = i*30;
  int m = i*comm_reverse;

  v_buf[m++] = static_cast<double>(d_rho0[i+first]);
  v_buf[m++] = static_cast<double>(d_arho2b[i+first]);
  v_buf[m++] = static_cast<double>(d_arho1(i+first,0));
  v_buf[m++] = static_cast<double>(d_arho1(i+first,1));
  v_buf[m++] = static_cast<double>(d_arho1(i+first,2));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,0));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,1));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,2));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,3));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,4));
  v_buf[m++] = static_cast<double>(d_arho2(i+first,5));
  for (int k = 0; k < 10; k++) v_buf[m++] = static_cast<double>(d_arho3(i+first,k));
  v_buf[m++] = static_cast<double>(d_arho3b(i+first,0));
  v_buf[m++] = static_cast<double>(d_arho3b(i+first,1));
  v_buf[m++] = static_cast<double>(d_arho3b(i+first,2));
  v_buf[m++] = static_cast<double>(d_t_ave(i+first,0));
  v_buf[m++] = static_cast<double>(d_t_ave(i+first,1));
  v_buf[m++] = static_cast<double>(d_t_ave(i+first,2));
  v_buf[m++] = static_cast<double>(d_tsq_ave(i+first,0));
  v_buf[m++] = static_cast<double>(d_tsq_ave(i+first,1));
  v_buf[m++] = static_cast<double>(d_tsq_ave(i+first,2));
  if (msmeamflag) {
    v_buf[m++] = static_cast<double>(d_arho2mb[i+first]);
    v_buf[m++] = static_cast<double>(d_arho1m(i+first,0));
    v_buf[m++] = static_cast<double>(d_arho1m(i+first,1));
    v_buf[m++] = static_cast<double>(d_arho1m(i+first,2));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,0));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,1));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,2));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,3));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,4));
    v_buf[m++] = static_cast<double>(d_arho2m(i+first,5));
    for (int k = 0; k < 10; k++) v_buf[m++] = static_cast<double>(d_arho3m(i+first,k));
    v_buf[m++] = static_cast<double>(d_arho3mb(i+first,0));
    v_buf[m++] = static_cast<double>(d_arho3mb(i+first,1));
    v_buf[m++] = static_cast<double>(d_arho3mb(i+first,2));
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int PairMEAMKokkos<DeviceType>::pack_reverse_comm(int n, int first, double *buf)
{
  meam_inst_kk->k_rho0.sync_host();
  meam_inst_kk->k_arho2b.sync_host();
  meam_inst_kk->k_arho1.sync_host();
  meam_inst_kk->k_arho2.sync_host();
  meam_inst_kk->k_arho3.sync_host();
  meam_inst_kk->k_arho3b.sync_host();
  meam_inst_kk->k_t_ave.sync_host();
  meam_inst_kk->k_tsq_ave.sync_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.sync_host();
    meam_inst_kk->k_arho1m.sync_host();
    meam_inst_kk->k_arho2m.sync_host();
    meam_inst_kk->k_arho3m.sync_host();
    meam_inst_kk->k_arho3mb.sync_host();
  }

  int m = 0;
  const int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = static_cast<double>(meam_inst_kk->h_rho0[i]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2b[i]);
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(i,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(i,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho1(i,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,3));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,4));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho2(i,5));
    for (int k = 0; k < 10; k++) buf[m++] = static_cast<double>(meam_inst_kk->h_arho3(i,k));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(i,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(i,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_arho3b(i,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(i,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(i,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_t_ave(i,2));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(i,0));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(i,1));
    buf[m++] = static_cast<double>(meam_inst_kk->h_tsq_ave(i,2));
    if (msmeamflag) {
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2mb[i]);
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(i,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(i,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho1m(i,2));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,2));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,3));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,4));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho2m(i,5));
      for (int k = 0; k < 10; k++) buf[m++] = static_cast<double>(meam_inst_kk->h_arho3m(i,k));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(i,0));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(i,1));
      buf[m++] = static_cast<double>(meam_inst_kk->h_arho3mb(i,2));
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::unpack_reverse_comm_kokkos(int n, DAT::tdual_int_1d k_sendlist, DAT::tdual_double_1d &buf)
{
  d_sendlist = k_sendlist.view<DeviceType>();
  v_buf = buf.view<DeviceType>();
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagPairMEAMUnpackReverseComm>(0,n),*this);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMUnpackReverseComm, const int &i) const {
  int j = d_sendlist(i);
  //int m = i*30;
  int m = i*comm_reverse;

  d_rho0[j] += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2b[j] += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho1(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho1(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho1(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,3) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,4) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho2(j,5) += static_cast<KK_FLOAT>(v_buf[m++]);
  for (int k = 0; k < 10; k++) d_arho3(j,k) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho3b(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho3b(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_arho3b(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_t_ave(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_t_ave(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_t_ave(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_tsq_ave(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_tsq_ave(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
  d_tsq_ave(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  if (msmeamflag) {
    d_arho2mb[j] += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1m(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1m(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho1m(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,3) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,4) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho2m(j,5) += static_cast<KK_FLOAT>(v_buf[m++]);
    for (int k = 0; k < 10; k++) d_arho3m(j,k) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3mb(j,0) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3mb(j,1) += static_cast<KK_FLOAT>(v_buf[m++]);
    d_arho3mb(j,2) += static_cast<KK_FLOAT>(v_buf[m++]);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::unpack_reverse_comm(int n, int *list, double *buf)
{
  meam_inst_kk->k_rho0.sync_host();
  meam_inst_kk->k_arho2b.sync_host();
  meam_inst_kk->k_arho1.sync_host();
  meam_inst_kk->k_arho2.sync_host();
  meam_inst_kk->k_arho3.sync_host();
  meam_inst_kk->k_arho3b.sync_host();
  meam_inst_kk->k_t_ave.sync_host();
  meam_inst_kk->k_tsq_ave.sync_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.sync_host();
    meam_inst_kk->k_arho1m.sync_host();
    meam_inst_kk->k_arho2m.sync_host();
    meam_inst_kk->k_arho3m.sync_host();
    meam_inst_kk->k_arho3mb.sync_host();
  }

  int m = 0;
  for (int i = 0; i < n; i++) {
    const int j = list[i];
    meam_inst_kk->h_rho0[j] += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2b[j] += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(j,0) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(j,1) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho1(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,0) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,1) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,3) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,4) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho2(j,5) += static_cast<KK_FLOAT>(buf[m++]);
    for (int k = 0; k < 10; k++) meam_inst_kk->h_arho3(j,k) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(j,0) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(j,1) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_arho3b(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(j,0) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(j,1) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_t_ave(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(j,0) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(j,1) += static_cast<KK_FLOAT>(buf[m++]);
    meam_inst_kk->h_tsq_ave(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    if (msmeamflag) {
      meam_inst_kk->h_arho2mb[j] += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(j,0) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(j,1) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho1m(j,2) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,0) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,1) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,2) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,3) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,4) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho2m(j,5) += static_cast<KK_FLOAT>(buf[m++]);
      for (int k = 0; k < 10; k++) meam_inst_kk->h_arho3m(j,k) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(j,0) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(j,1) += static_cast<KK_FLOAT>(buf[m++]);
      meam_inst_kk->h_arho3mb(j,2) += static_cast<KK_FLOAT>(buf[m++]);
    }
  }

  meam_inst_kk->k_rho0.modify_host();
  meam_inst_kk->k_arho2b.modify_host();
  meam_inst_kk->k_arho1.modify_host();
  meam_inst_kk->k_arho2.modify_host();
  meam_inst_kk->k_arho3.modify_host();
  meam_inst_kk->k_arho3b.modify_host();
  meam_inst_kk->k_t_ave.modify_host();
  meam_inst_kk->k_tsq_ave.modify_host();
  if (msmeamflag) {
    meam_inst_kk->k_arho2mb.modify_host();
    meam_inst_kk->k_arho1m.modify_host();
    meam_inst_kk->k_arho2m.modify_host();
    meam_inst_kk->k_arho3m.modify_host();
    meam_inst_kk->k_arho3mb.modify_host();
  }
}

/* ----------------------------------------------------------------------
   strip special bond flags from neighbor list entries
   are not used with MEAM
   need to do here so Fortran lib doesn't see them
   done once per reneighbor so that neigh_f2c and neigh_c2f don't see them
------------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMNeighStrip, const int &ii) const {

  const int i = d_ilist_half[ii];
  const int jnum_half = d_numneigh_half[i];
  const int jnum_full = d_numneigh_full[i];
  for (int jj = 0; jj < jnum_half; jj++)
    d_neighbors_half(i,jj) &= NEIGHMASK;
  for (int jj = 0; jj < jnum_full; jj++)
    d_neighbors_full(i,jj) &= NEIGHMASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairMEAMKokkos<DeviceType>::operator()(TagPairMEAMOffsets, const int ii, int &n) const {
  const int i = d_ilist_half[ii];
  n += d_numneigh_half[i];
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairMEAMKokkos<DeviceType>::update_meam_views()
{
  d_rho0 = meam_inst_kk->d_rho0;
  d_rho1 = meam_inst_kk->d_rho1;
  d_rho2 = meam_inst_kk->d_rho2;
  d_rho3 = meam_inst_kk->d_rho3;
  d_frhop = meam_inst_kk->d_frhop;
  d_gamma = meam_inst_kk->d_gamma;
  d_dgamma1 = meam_inst_kk->d_dgamma1;
  d_dgamma2 = meam_inst_kk->d_dgamma2;
  d_dgamma3 = meam_inst_kk->d_dgamma3;
  d_arho1 = meam_inst_kk->d_arho1;
  d_arho2 = meam_inst_kk->d_arho2;
  d_arho3 = meam_inst_kk->d_arho3;
  d_arho2b = meam_inst_kk->d_arho2b;
  d_arho3b = meam_inst_kk->d_arho3b;
  d_t_ave = meam_inst_kk->d_t_ave;
  d_tsq_ave = meam_inst_kk->d_tsq_ave;
  // msmeam
  d_arho1m = meam_inst_kk->d_arho1m;
  d_arho2m = meam_inst_kk->d_arho2m;
  d_arho3m = meam_inst_kk->d_arho3m;
  d_arho2mb = meam_inst_kk->d_arho2mb;
  d_arho3mb = meam_inst_kk->d_arho3mb;
}

/* ----------------------------------------------------------------------
   local energy changes for atom type swaps (device owner-computes)
------------------------------------------------------------------------- */

template <class DeviceType> PairSwapLocal *PairMEAMKokkos<DeviceType>::swap_local()
{
  if (typeid(*this) != typeid(PairMEAMKokkos<DeviceType>)) return nullptr;
  return this;
}

template <class DeviceType>
SwapLocalCaps PairMEAMKokkos<DeviceType>::swap_local_caps(std::string &reason) const
{
  SwapLocalCaps caps;
  caps.supported = 1;
  caps.supports_charge = 1;
  caps.dependency_radius = 0.0;
  caps.conflict_radius = 0.0;

  if (typeid(*this) != typeid(PairMEAMKokkos<DeviceType>)) {
    caps.supported = 0;
    reason = "derived pair style has not been validated for local swap energies";
    return caps;
  }
  if (msmeamflag) {
    caps.supported = 0;
    reason = "MS-MEAM is not supported";
    return caps;
  }
  if (!allocated) {
    caps.supported = 0;
    reason = "pair coefficients are not set";
    return caps;
  }
  for (int itype = 1; itype <= atom->ntypes; itype++) {
    if (map[itype] < 0) {
      caps.supported = 0;
      reason = "atom types mapped to NULL are not supported";
      return caps;
    }
  }

  const double rc = meam_inst->swap_cutoff();
  const double ebmax = meam_inst->swap_ebound_max();
  caps.dependency_radius = sqrt(ebmax) * rc;
  caps.conflict_radius = 2.0 * caps.dependency_radius;

  const double rlist = cutmax + neighbor->skin;
  if (!(ebmax * rc * rc * (1.0 + 1.0e-10) < rlist * rlist)) {
    caps.supported = 0;
    reason = fmt::format("screening atoms up to a distance of {:.8g} are not all inside the "
                         "neighbor list cutoff {:.8g}, increase the neighbor skin",
                         caps.dependency_radius, rlist);
  }
  return caps;
}

template <class DeviceType> void PairMEAMKokkos<DeviceType>::swap_ensure_dens(int nlocal_need) const
{
  constexpr int NDENS = MeamSwapParams::NDENS;
  if ((int) d_swap_dens.extent(0) < nlocal_need || (int) d_swap_dens.extent(1) < NDENS) {
    const int cap = nlocal_need + nlocal_need / 4 + 1;
    d_swap_dens = decltype(d_swap_dens)("meam:swap dens", cap, NDENS);
    d_swap_dens_valid = typename AT::t_int_1d("meam:swap dens valid", cap);
  } else if ((int) d_swap_dens_valid.extent(0) < nlocal_need) {
    d_swap_dens_valid =
        typename AT::t_int_1d("meam:swap dens valid", nlocal_need + nlocal_need / 4 + 1);
  }
}

template <class DeviceType> void PairMEAMKokkos<DeviceType>::swap_local_begin(NeighList *full_list)
{
  swap_eval_valid = 0;
  swap_eval_n = 0;
  swap_list_kk = nullptr;
  // A host fix can query this interface too.  Decline a list from another
  // memory space so the caller can fall back to full energy evaluations.
  if (!full_list || !full_list->kokkos || (full_list->execution_space != execution_space)) return;
  swap_list_kk = static_cast<NeighListKokkos<DeviceType> *>(full_list);
  if (swap_centers_kk.setup(atom, swap_list_kk)) swap_list_kk = nullptr;

  const double rc = meam_inst->swap_cutoff();
  swap_rdepsq = meam_inst->swap_ebound_max() * rc * rc * (1.0 + 1.0e-10);
  if (!d_swap_params.data())
    d_swap_params = Kokkos::View<MeamSwapParams, DeviceType>(
        Kokkos::view_alloc("meam:swap params", Kokkos::WithoutInitializing));
  auto h_params = Kokkos::create_mirror_view(d_swap_params);
  meam_inst_kk->fill_swap_params(h_params());
  Kokkos::deep_copy(d_swap_params, h_params);

  const int nlocal_need = atom->nlocal;
  swap_ensure_dens(nlocal_need);
  if (nlocal_need > 0) {
    Kokkos::deep_copy(Kokkos::subview(d_swap_dens_valid, Kokkos::pair<int, int>(0, nlocal_need)),
                      0);
  }
  if (!d_swap_error.data()) d_swap_error = decltype(d_swap_error)("meam:swap error");

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | TAG_MASK);
}

template <class DeviceType> void PairMEAMKokkos<DeviceType>::swap_local_end()
{
  swap_list_kk = nullptr;
  swap_eval_valid = 0;
  swap_eval_n = 0;
}

template <class DeviceType>
int PairMEAMKokkos<DeviceType>::swap_local_delta_energy_batch(int n, const SwapProposal *proposals,
                                                              double *dE) const
{
  if (!swap_list_kk) return 1;
  if (!n) return 0;

  swap_images_kk.stage(atom, n, proposals);
  swap_eval_n = n;
  swap_eval_valid = 0;

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | TAG_MASK);
  const auto sx = atomKK->k_x.template view<DeviceType>();
  if (swap_affected_kk.build(n, swap_images_kk, swap_centers_kk, swap_list_kk, sx, atom->nlocal,
                             swap_rdepsq))
    return 1;

  constexpr int NDENS = MeamSwapParams::NDENS;
  const auto stype = atomKK->k_type.template view<DeviceType>();
  const auto stag = atomKK->k_tag.template view<DeviceType>();
  const auto smap = d_map;
  const auto sscale = d_scale;
  const auto num = swap_list_kk->d_numneigh;
  const auto neighbors = swap_list_kk->d_neighbors;
  const auto phirar = meam_inst_kk->d_phirar;
  const auto phirar1 = meam_inst_kk->d_phirar1;
  const auto phirar2 = meam_inst_kk->d_phirar2;
  const auto phirar3 = meam_inst_kk->d_phirar3;
  const Kokkos::View<const MeamSwapParams, DeviceType> d_params = d_swap_params;

  const int na = swap_affected_kk.size;
  if ((int) d_swap_ddens.extent(0) < na || (int) d_swap_ddens.extent(1) < NDENS) {
    const int cap = Kokkos::max(na + na / 4 + 1, 1);
    d_swap_ddens = decltype(d_swap_ddens)("meam:swap ddens", cap, NDENS);
  }
  if ((int) d_swap_energy.extent(0) < n) {
    const int cap = n + n / 4 + 1;
    d_swap_energy = decltype(d_swap_energy)("meam:swap energy", cap);
    h_swap_energy = decltype(h_swap_energy)("meam:host swap energy", cap);
  }
  if (!d_swap_error.data()) d_swap_error = decltype(d_swap_error)("meam:swap error");
  Kokkos::deep_copy(d_swap_energy, 0.0);
  Kokkos::deep_copy(d_swap_error, 0);

  const auto dens = d_swap_dens;
  const auto dens_valid = d_swap_dens_valid;
  const auto ddens = d_swap_ddens;
  const auto energy_out = d_swap_energy;
  const auto err = d_swap_error;
  const auto centers = swap_affected_kk.d_atom;
  const auto which = swap_affected_kk.d_proposal;
  const auto props = swap_images_kk.d_proposals;
  const auto img_offset = swap_images_kk.d_img_offset;
  const auto img_atom = swap_images_kk.d_img_atom;

  Kokkos::parallel_for(
      "meam:swap delta energies", Kokkos::RangePolicy<DeviceType>(0, na), KOKKOS_LAMBDA(int s) {
        const MeamSwapParams &params = d_params();
        const int c = centers(s);
        const int k = which(s);
        const SwapProposal p = props(k);
        const int wc = swap_kk_which(p, stag(c));
        const int tcold = (wc < 0) ? stype(c) : p.oldtype[wc];
        const int tcnew = (wc < 0) ? stype(c) : p.newtype[wc];
        const int ecold = smap(tcold);
        const int ecnew = smap(tcnew);
        const int nfull = num(c);

        if (!dens_valid(c)) {
          double tmp[NDENS];
          for (int m = 0; m < NDENS; m++) tmp[m] = 0.0;
          for (int jn = 0; jn < nfull; jn++) {
            const int j = neighbors(c, jn) & NEIGHMASK;
            double delij[3] = {sx(j, 0) - sx(c, 0), sx(j, 1) - sx(c, 1), sx(j, 2) - sx(c, 2)};
            const double rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
            if (!(rij2 < params.cutforcesq)) continue;
            const int ej = smap(swap_kk_oldtype(p, stag(j), stype(j)));
            double scr, fc, epair;
            meam_swap_screen(params, c, j, ecold, ej, sx, nfull, neighbors, smap, stype, stag, p, 0,
                             scr, fc);
            meam_swap_pair(params, scr, fc, ecold, ej, delij, rij2, 1.0, 1.0, epair, tmp, phirar,
                           phirar1, phirar2, phirar3);
          }
          for (int m = 0; m < NDENS; m++) dens(c, m) = tmp[m];
          dens_valid(c) = 1;
        }

        double dc[NDENS];
        for (int m = 0; m < NDENS; m++) dc[m] = 0.0;
        double epair_sum = 0.0;

        for (int jn = 0; jn < nfull; jn++) {
          const int j = neighbors(c, jn) & NEIGHMASK;
          double delij[3] = {sx(j, 0) - sx(c, 0), sx(j, 1) - sx(c, 1), sx(j, 2) - sx(c, 2)};
          const double rij2 = delij[0] * delij[0] + delij[1] * delij[1] + delij[2] * delij[2];
          if (!(rij2 < params.cutforcesq)) continue;

          const int wj = swap_kk_which(p, stag(j));
          const int tjold = (wj < 0) ? stype(j) : p.oldtype[wj];
          const int tjnew = (wj < 0) ? stype(j) : p.newtype[wj];
          const int ejold = smap(tjold);
          const int ejnew = smap(tjnew);

          if ((wc < 0) && (wj < 0)) {
            const double rbound = params.ebound_meam[ecold][ejold] * rij2;
            int changed = 0;
            for (int im = img_offset(k); im < img_offset(k + 1); ++im) {
              const int ka = img_atom(im);
              const double dxjk = sx(ka, 0) - sx(j, 0);
              const double dyjk = sx(ka, 1) - sx(j, 1);
              const double dzjk = sx(ka, 2) - sx(j, 2);
              if (dxjk * dxjk + dyjk * dyjk + dzjk * dzjk > rbound) continue;
              const double dxik = sx(ka, 0) - sx(c, 0);
              const double dyik = sx(ka, 1) - sx(c, 1);
              const double dzik = sx(ka, 2) - sx(c, 2);
              if (dxik * dxik + dyik * dyik + dzik * dzik > rbound) continue;
              changed = 1;
              break;
            }
            if (!changed) continue;
          }

          double scr_old, scr_new, fc, eold, enew;
          meam_swap_screen_both(params, c, j, ecold, ejold, ecnew, ejnew, sx, nfull, neighbors,
                                smap, stype, stag, p, scr_old, scr_new, fc);
          meam_swap_pair(params, scr_old, fc, ecold, ejold, delij, rij2,
                         static_cast<double>(sscale(tcold, tjold)), -1.0, eold, dc, phirar, phirar1,
                         phirar2, phirar3);
          meam_swap_pair(params, scr_new, fc, ecnew, ejnew, delij, rij2,
                         static_cast<double>(sscale(tcnew, tjnew)), 1.0, enew, dc, phirar, phirar1,
                         phirar2, phirar3);
          epair_sum += 0.5 * (enew - eold);
        }

        for (int m = 0; m < NDENS; m++) ddens(s, m) = dc[m];

        double dnew[NDENS];
        double dold_buf[NDENS];
        for (int m = 0; m < NDENS; m++) {
          dold_buf[m] = dens(c, m);
          dnew[m] = dens(c, m) + dc[m];
        }
        int errnew = 0, errold = 0;
        const double fnew = meam_swap_embedding(params, dnew, ecnew,
                                                static_cast<double>(sscale(tcnew, tcnew)), errnew);
        const double fold = meam_swap_embedding(params, dold_buf, ecold,
                                                static_cast<double>(sscale(tcold, tcold)), errold);
        if (errnew || errold) {
          Kokkos::atomic_exchange(&err(), 1);
          return;
        }
        Kokkos::atomic_add(&energy_out(k), epair_sum + fnew - fold);
      });

  int host_err = 0;
  Kokkos::deep_copy(host_err, d_swap_error);
  if (host_err) {
    swap_eval_valid = 0;
    return 1;
  }

  Kokkos::deep_copy(Kokkos::subview(h_swap_energy, Kokkos::pair<int, int>(0, n)),
                    Kokkos::subview(d_swap_energy, Kokkos::pair<int, int>(0, n)));
  for (int k = 0; k < n; ++k) dE[k] = h_swap_energy(k);
  swap_eval_valid = 1;
  swap_eval_n = n;
  if ((int) swap_eval_proposals.extent(0) < n)
    swap_eval_proposals = decltype(swap_eval_proposals)("meam:cached proposals", n + n / 4 + 1);
  for (int k = 0; k < n; ++k) swap_eval_proposals(k) = proposals[k];
  return 0;
}

template <class DeviceType>
void PairMEAMKokkos<DeviceType>::swap_local_commit_batch(int n, const SwapProposal *proposals,
                                                         const int *accepted)
{
  if (!swap_list_kk)
    error->one(FLERR, Error::NOLASTLINE, "Pair style {} could not update local swap data", myname);
  if (!n) return;

  constexpr int NDENS = MeamSwapParams::NDENS;
  int any = 0;
  for (int k = 0; k < n; k++)
    if (accepted[k]) any = 1;
  if (!any) {
    swap_eval_valid = 0;
    return;
  }

  // Cached density deltas belong to the exact batch, including its order.
  bool matching = swap_eval_valid && (swap_eval_n == n);
  for (int k = 0; matching && (k < n); ++k)
    matching = swap_local_same_types(proposals[k], swap_eval_proposals(k));
  if (!matching) {
    std::vector<double> unused(n);
    if (swap_local_delta_energy_batch(n, proposals, unused.data()))
      error->one(FLERR, Error::NOLASTLINE, "Pair style {} could not update local swap data",
                 myname);
  }

  if ((int) d_swap_accepted.extent(0) < n) {
    const int cap = n + n / 4 + 1;
    d_swap_accepted = decltype(d_swap_accepted)("meam:swap accepted", cap);
    h_swap_accepted = decltype(h_swap_accepted)("meam:host swap accepted", cap);
  }
  for (int k = 0; k < n; k++) h_swap_accepted(k) = accepted[k];
  Kokkos::deep_copy(Kokkos::subview(d_swap_accepted, Kokkos::pair<int, int>(0, n)),
                    Kokkos::subview(h_swap_accepted, Kokkos::pair<int, int>(0, n)));

  const int na = swap_affected_kk.size;
  const auto dens = d_swap_dens;
  const auto dens_valid = d_swap_dens_valid;
  const auto ddens = d_swap_ddens;
  const auto centers = swap_affected_kk.d_atom;
  const auto which = swap_affected_kk.d_proposal;
  const auto acc = d_swap_accepted;

  Kokkos::parallel_for(
      "meam:swap commit apply", Kokkos::RangePolicy<DeviceType>(0, na), KOKKOS_LAMBDA(int s) {
        const int k = which(s);
        if (!acc(k)) return;
        const int c = centers(s);
        if (!dens_valid(c)) return;
        for (int m = 0; m < NDENS; m++) dens(c, m) += ddens(s, m);
      });

  swap_eval_valid = 0;
  swap_eval_n = 0;
}

template <class DeviceType>
void PairMEAMKokkos<DeviceType>::swap_local_commit(const SwapProposal &accepted)
{
  int one = 1;
  swap_local_commit_batch(1, &accepted, &one);
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairMEAMKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairMEAMKokkos<LMPHostType>;
#endif
}

