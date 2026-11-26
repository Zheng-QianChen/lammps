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

#ifndef LMP_PAIR_SWAP_LOCAL_H
#define LMP_PAIR_SWAP_LOCAL_H

#include "lmptype.h"

#include <limits>
#include <string>
#include <vector>

namespace LAMMPS_NS {

/* ----------------------------------------------------------------------
   optional interface of a pair style for exact local energy changes of
   atom type swaps, used by fix atom/swap.  a pair style returns a pointer
   to it from Pair::swap_local().

   energy decomposition (owner computes):
     E_pair = sum over all atoms c of the site energy E_c, which depends
     only on atoms within the dependency radius R_dep of c.  each MPI rank
     evaluates the change of E_c for the atoms c it owns, the caller sums
     these partial changes over all ranks with MPI_Allreduce(MPI_SUM).

   requirements met by the caller between swap_local_begin() and
   swap_local_end():
   - coordinates and ghost atoms are current and ghost atoms are acquired
     out to at least max(R_dep, Pair::cutforce)
   - the atom map (Atom::map() and Atom::sametag) is valid for all owned
     and ghost atoms
   - "full_list" is a full neighbor list of all owned atoms with a uniform
     cutoff >= max(R_dep, Pair::cutforce) for all pairs of atom types.
     neighbors include ghost atoms.  ghost atoms are not list centers,
     owned atoms near a ghost atom are found from the owned centers.
   - every proposal is known on all MPI ranks.  atom IDs and old and new
     types are valid on all ranks, idx[] is the local index of an edited
     atom on the rank owning it and -1 on all other ranks.
   - atom types and charges change only through accepted proposals.  every
     accepted proposal is reported with swap_local_commit() (or
     swap_local_commit_batch()) on all MPI ranks, also on ranks that do not
     own an edited atom.  the caller writes the new types and charges to
     all owned and ghost images on all ranks, either before or after the
     commit call.
   - proposals of one batch are independent: their edited atoms are more
     than R_conflict apart, so their energy changes are additive.
------------------------------------------------------------------------- */

struct SwapProposal {
  int count;         // 1 = semi-grand (single atom), 2 = canonical swap
  int idx[2];        // local index of owned atom on this rank, -1 if not owned here
  tagint tag[2];     // atom ID (valid on all ranks)
  int oldtype[2];    // type before the proposal
  int newtype[2];    // type after the proposal
  double oldq[2];    // charge before the proposal (if qflag)
  double newq[2];    // charge after the proposal (if qflag)
  int qflag;         // 1 if charges change
};

struct SwapLocalCaps {
  int supported;               // 1 if the current pair style setup supports local changes
  int supports_charge;         // 1 if charge changes are supported
  double dependency_radius;    // R_dep: max distance of an affected site from an edited atom
  double conflict_radius;      // R_conflict: min distance of independent concurrent edits
};

class PairSwapLocal {
 public:
  virtual ~PairSwapLocal() noexcept(false) {}    // match Pointers::~Pointers()

  // report whether the current pair style setup supports local energy changes.
  // if supported == 0, "reason" explains why.

  virtual SwapLocalCaps swap_local_caps(std::string &reason) const = 0;

  // set up per-invocation data for the current state

  virtual void swap_local_begin(class NeighList *full_list) = 0;

  // owner computes partial energy changes of n independent proposals:
  // dE[k] = sum over owned atoms c of (E_c after proposal k - E_c now).
  // each proposal is relative to the current state.  the caller sums
  // dE[0..n-1] over all MPI ranks with MPI_Allreduce(MPI_SUM).
  // must not modify atom->type or atom->q.  return 0 if all values were
  // computed exactly, non-zero otherwise (the caller then falls back to
  // full energy evaluations).  must be called on all ranks, also on
  // ranks without owned atoms.

  virtual int swap_local_delta_energy_batch(int n, const SwapProposal *proposals,
                                            double *dE) const = 0;

  // update internal data after "accepted" was accepted.  old and new types
  // and charges of the edited atoms must be taken from the proposal.

  virtual void swap_local_commit(const SwapProposal &accepted) = 0;

  // update internal data for all proposals k with accepted[k] != 0

  virtual void swap_local_commit_batch(int n, const SwapProposal *proposals, const int *accepted)
  {
    for (int k = 0; k < n; k++)
      if (accepted[k]) swap_local_commit(proposals[k]);
  }

  // release per-invocation resources

  virtual void swap_local_end() {}
};

/* ----------------------------------------------------------------------
   helpers for type lookups in implementations
------------------------------------------------------------------------- */

// return index m (0 or 1) of the edited atom with ID "itag", or -1

inline int swap_local_which(const SwapProposal &p, tagint itag)
{
  if (p.tag[0] == itag) return 0;
  if ((p.count > 1) && (p.tag[1] == itag)) return 1;
  return -1;
}

// return type of atom with ID "itag" and current type "itype" after the proposal

inline int swap_local_newtype(const SwapProposal &p, tagint itag, int itype)
{
  if (p.tag[0] == itag) return p.newtype[0];
  if ((p.count > 1) && (p.tag[1] == itag)) return p.newtype[1];
  return itype;
}

// return type of atom with ID "itag" and current type "itype" before the proposal

inline int swap_local_oldtype(const SwapProposal &p, tagint itag, int itype)
{
  if (p.tag[0] == itag) return p.oldtype[0];
  if ((p.count > 1) && (p.tag[1] == itag)) return p.oldtype[1];
  return itype;
}

// return true if two proposals edit the same atoms with the same old and new types.
// local indices are not compared, since they differ between MPI ranks.  charges are
// not compared either, implementations whose energy depends on charges must check them.

inline bool swap_local_same_types(const SwapProposal &a, const SwapProposal &b)
{
  if (a.count != b.count) return false;
  for (int m = 0; m < a.count; m++)
    if ((a.tag[m] != b.tag[m]) || (a.oldtype[m] != b.oldtype[m]) || (a.newtype[m] != b.newtype[m]))
      return false;
  return true;
}

/* ----------------------------------------------------------------------
   owned list centers near an owned or ghost atom for owner computes
   implementations.  for an owned atom a these are the owned neighbors in
   the (symmetric) full list of a, for a ghost atom a the owned atoms that
   have a in their full list.  the latter are stored in reverse lists.
------------------------------------------------------------------------- */

class SwapLocalCenters {
 public:
  // set up reverse lists for the current full neighbor list.
  // return 0 on success, 1 if not all owned atoms are list centers

  int setup(class Atom *, class NeighList *);
  void clear();

  // call func(c) for every owned atom c != a that has atom a in its full list

  template <typename F> void for_each(int a, F &&func) const
  {
    if (a < nlocal) {
      const int *jlist = firstneigh[a];
      const int jnum = numneigh[a];
      for (int jj = 0; jj < jnum; jj++) {
        const int c = jlist[jj] & NEIGHMASK;
        if (c < nlocal) func(c);
      }
    } else {
      const int g = a - nlocal;
      for (int n = gfirst[g]; n < gfirst[g + 1]; n++) func(gcenter[n]);
    }
  }

 private:
  int nlocal = 0;
  int *numneigh = nullptr;
  int **firstneigh = nullptr;
  std::vector<int> gfirst, gcenter;    // reverse lists of ghost atoms (CSR)
};

/* ----------------------------------------------------------------------
   set of local atom indices with O(1) insertion and O(1) reset
------------------------------------------------------------------------- */

class SwapLocalStamp {
 public:
  void resize(int n)
  {
    if ((int) stamp.size() < n) stamp.resize(n, 0);
  }

  // start a new empty set

  void reset()
  {
    if (current == std::numeric_limits<int>::max()) {
      for (auto &s : stamp) s = 0;
      current = 0;
    }
    ++current;
  }

  // insert i, return true if i was not in the set

  bool insert(int i)
  {
    if (stamp[i] == current) return false;
    stamp[i] = current;
    return true;
  }

  bool contains(int i) const { return stamp[i] == current; }

  void release()
  {
    stamp.clear();
    stamp.shrink_to_fit();
    current = 0;
  }

 private:
  std::vector<int> stamp;
  int current = 0;
};

}    // namespace LAMMPS_NS

#endif
