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

#ifndef LMP_PAIR_SWAP_LOCAL_KOKKOS_H
#define LMP_PAIR_SWAP_LOCAL_KOKKOS_H

#include "atom.h"
#include "neigh_list_kokkos.h"
#include "pair_swap_local.h"

namespace LAMMPS_NS {

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
int swap_kk_which(const SwapProposal &p, tagint tag)
{
  if (p.tag[0] == tag) return 0;
  if ((p.count > 1) && (p.tag[1] == tag)) return 1;
  return -1;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
int swap_kk_oldtype(const SwapProposal &p, tagint tag, int type)
{
  const int m = swap_kk_which(p, tag);
  return (m < 0) ? type : p.oldtype[m];
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
int swap_kk_newtype(const SwapProposal &p, tagint tag, int type)
{
  const int m = swap_kk_which(p, tag);
  return (m < 0) ? type : p.newtype[m];
}

// Reverse incidence lists contain only owned centers of ghost neighbors.
// Count, scan and scatter all run in the list's execution space.

template <class DeviceType> class SwapLocalCentersKokkos {
 public:
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d d_gfirst, d_gcenter;

  // Named functors keep CUDA kernel identity stable across translation
  // units.  Extended lambdas in a shared header can select a host copier
  // from a different object file when the linker merges template instances.
  struct CountGhosts {
    typename AT::t_int_1d ilist, num, count;
    typename AT::t_neighbors_2d neighbors;
    int nlocal;

    // NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(int ii) const
    {
      const int c = ilist(ii);
      for (int jj = 0; jj < num(c); ++jj) {
        const int j = neighbors(c, jj) & NEIGHMASK;
        if (j >= nlocal) Kokkos::atomic_inc(&count(j - nlocal));
      }
    }
  };

  struct ScanGhosts {
    typename AT::t_int_1d count, first;
    int nghost;

    // NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(int g, int &offset, bool final) const
    {
      if (final) first(g) = offset;
      if (g < nghost) offset += count(g);
    }
  };

  struct ScatterGhosts {
    typename AT::t_int_1d ilist, num, first, fill, center;
    typename AT::t_neighbors_2d neighbors;
    int nlocal;

    // NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(int ii) const
    {
      const int c = ilist(ii);
      for (int jj = 0; jj < num(c); ++jj) {
        const int j = neighbors(c, jj) & NEIGHMASK;
        if (j < nlocal) continue;
        const int g = j - nlocal;
        center(first(g) + Kokkos::atomic_fetch_add(&fill(g), 1)) = c;
      }
    }
  };

  int setup(Atom *atom, NeighListKokkos<DeviceType> *list)
  {
    const int nlocal = atom->nlocal;
    const int nghost = atom->nghost;
    if (!list || (list->inum != nlocal)) return 1;
    if ((int) d_gfirst.extent(0) < nghost + 1) {
      const int cap = nghost + nghost / 4 + 1;
      d_gfirst = typename AT::t_int_1d("swap:gfirst", cap);
      d_count = typename AT::t_int_1d("swap:gcount", cap);
      d_fill = typename AT::t_int_1d("swap:gfill", cap);
    }
    Kokkos::deep_copy(d_count, 0);
    Kokkos::parallel_for(
        "swap:count ghosts", Kokkos::RangePolicy<DeviceType>(0, nlocal),
        CountGhosts{list->d_ilist, list->d_numneigh, d_count, list->d_neighbors, nlocal});
    int total = 0;
    Kokkos::parallel_scan("swap:ghost offsets", Kokkos::RangePolicy<DeviceType>(0, nghost + 1),
                          ScanGhosts{d_count, d_gfirst, nghost}, total);
    if ((int) d_gcenter.extent(0) < total)
      d_gcenter = typename AT::t_int_1d("swap:gcenter", total + total / 4 + 1);
    Kokkos::deep_copy(d_fill, 0);
    Kokkos::parallel_for("swap:scatter ghosts", Kokkos::RangePolicy<DeviceType>(0, nlocal),
                         ScatterGhosts{list->d_ilist, list->d_numneigh, d_gfirst, d_fill, d_gcenter,
                                       list->d_neighbors, nlocal});
    return 0;
  }

 private:
  typename AT::t_int_1d d_count, d_fill;
};

// The host map is used only for the few edited tags, including every image.
// Transfer subviews keep staging proportional to this batch, not capacity.

template <class DeviceType> class SwapBatchImagesKokkos {
 public:
  using AT = ArrayTypes<DeviceType>;
  Kokkos::View<SwapProposal *, DeviceType> d_proposals;
  typename AT::t_int_1d d_img_offset, d_img_atom, d_img_slot;

  void stage(Atom *atom, int n, const SwapProposal *proposals)
  {
    if ((int) d_proposals.extent(0) < n) {
      const int cap = n + n / 4 + 1;
      d_proposals = decltype(d_proposals)("swap:proposals", cap);
      h_proposals = decltype(h_proposals)("swap:host proposals", cap);
      d_img_offset = typename AT::t_int_1d("swap:image offsets", cap + 1);
      h_offset = decltype(h_offset)("swap:host offsets", cap + 1);
    }
    int nimage = 0;
    for (int k = 0; k < n; ++k)
      for (int m = 0; m < proposals[k].count; ++m)
        for (int a = atom->map(proposals[k].tag[m]); a >= 0; a = atom->sametag[a])
          if (a < atom->nlocal + atom->nghost) ++nimage;
    if ((int) d_img_atom.extent(0) < nimage) {
      const int cap = nimage + nimage / 4 + 1;
      d_img_atom = typename AT::t_int_1d("swap:image atoms", cap);
      d_img_slot = typename AT::t_int_1d("swap:image slots", cap);
      h_atom = decltype(h_atom)("swap:host images", cap);
      h_slot = decltype(h_slot)("swap:host slots", cap);
    }
    int pos = 0;
    for (int k = 0; k < n; ++k) {
      h_proposals(k) = proposals[k];
      h_offset(k) = pos;
      for (int m = 0; m < proposals[k].count; ++m)
        for (int a = atom->map(proposals[k].tag[m]); a >= 0; a = atom->sametag[a]) {
          if (a >= atom->nlocal + atom->nghost) continue;
          h_atom(pos) = a;
          h_slot(pos++) = m;
        }
    }
    if (!n) return;
    h_offset(n) = pos;
    Kokkos::deep_copy(Kokkos::subview(d_proposals, Kokkos::pair<int, int>(0, n)),
                      Kokkos::subview(h_proposals, Kokkos::pair<int, int>(0, n)));
    Kokkos::deep_copy(Kokkos::subview(d_img_offset, Kokkos::pair<int, int>(0, n + 1)),
                      Kokkos::subview(h_offset, Kokkos::pair<int, int>(0, n + 1)));
    if (pos) {
      Kokkos::deep_copy(Kokkos::subview(d_img_atom, Kokkos::pair<int, int>(0, pos)),
                        Kokkos::subview(h_atom, Kokkos::pair<int, int>(0, pos)));
      Kokkos::deep_copy(Kokkos::subview(d_img_slot, Kokkos::pair<int, int>(0, pos)),
                        Kokkos::subview(h_slot, Kokkos::pair<int, int>(0, pos)));
    }
  }

 private:
  Kokkos::View<SwapProposal *, LMPPinnedHostType> h_proposals;
  Kokkos::View<int *, LMPPinnedHostType> h_offset, h_atom, h_slot;
};

// Independent proposals have disjoint affected owned sites.  A compact queue
// therefore needs at most nlocal entries; repeated periodic images are stamped
// once per batch.  No per-proposal full-box walk or device atom map is needed.

template <class DeviceType> class SwapLocalAffectedKokkos {
 public:
  using AT = ArrayTypes<DeviceType>;
  typename AT::t_int_1d d_atom, d_proposal;
  int size = 0;

  template <class XView> struct BuildAffected {
    XView x;
    typename AT::t_int_1d offset, img, first, gcenter, num, seen, atoms, props;
    typename AT::t_neighbors_2d neighbors;
    Kokkos::View<int, DeviceType> count, overflow;
    int nlocal;
    double rdepsq;

    // NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(int k) const
    {
      for (int im = offset(k); im < offset(k + 1); ++im) {
        const int a = img(im);
        const int start = (a < nlocal) ? -1 : first(a - nlocal);
        const int end = (a < nlocal) ? num(a) : first(a - nlocal + 1);
        for (int jj = start; jj < end; ++jj) {
          const int c = (a < nlocal) ? ((jj < 0) ? a : (neighbors(a, jj) & NEIGHMASK))
                                     : gcenter(jj);
          if (c >= nlocal) continue;
          const double dx = x(c, 0) - x(a, 0);
          const double dy = x(c, 1) - x(a, 1);
          const double dz = x(c, 2) - x(a, 2);
          if (dx * dx + dy * dy + dz * dz > rdepsq) continue;
          if (Kokkos::atomic_exchange(&seen(c), 1) != 0) continue;
          const int s = Kokkos::atomic_fetch_add(&count(), 1);
          if (s >= static_cast<int>(atoms.extent(0))) {
            Kokkos::atomic_exchange(&overflow(), 1);
            continue;
          }
          atoms(s) = c;
          props(s) = k;
        }
      }
    }
  };

  template <class XView>
  int build(int n, const SwapBatchImagesKokkos<DeviceType> &images,
            const SwapLocalCentersKokkos<DeviceType> &centers, NeighListKokkos<DeviceType> *list,
            XView x, int nlocal, double rdepsq)
  {
    size = 0;
    if (!nlocal || !n) return 0;
    if ((int) d_atom.extent(0) < nlocal) {
      const int cap = nlocal + nlocal / 4 + 1;
      d_atom = typename AT::t_int_1d("swap:affected atoms", cap);
      d_proposal = typename AT::t_int_1d("swap:affected proposals", cap);
      d_seen = typename AT::t_int_1d("swap:affected seen", cap);
    }
    if (!d_size.data()) d_size = decltype(d_size)("swap:affected count");
    if (!d_overflow.data()) d_overflow = decltype(d_overflow)("swap:affected overflow");
    Kokkos::deep_copy(d_overflow, 0);
    Kokkos::deep_copy(d_seen, 0);
    Kokkos::deep_copy(d_size, 0);
    BuildAffected<XView> f;
    f.x = x;
    f.offset = images.d_img_offset;
    f.img = images.d_img_atom;
    f.first = centers.d_gfirst;
    f.gcenter = centers.d_gcenter;
    f.num = list->d_numneigh;
    f.neighbors = list->d_neighbors;
    f.seen = d_seen;
    f.atoms = d_atom;
    f.props = d_proposal;
    f.count = d_size;
    f.overflow = d_overflow;
    f.nlocal = nlocal;
    f.rdepsq = rdepsq;
    Kokkos::parallel_for("swap:affected owned sites", Kokkos::RangePolicy<DeviceType>(0, n), f);
    int overflow = 0;
    Kokkos::deep_copy(overflow, d_overflow);
    if (overflow) return 1;
    Kokkos::deep_copy(size, d_size);
    return 0;
  }

 private:
  typename AT::t_int_1d d_seen;
  Kokkos::View<int, DeviceType> d_size, d_overflow;
};

}    // namespace LAMMPS_NS

#endif
