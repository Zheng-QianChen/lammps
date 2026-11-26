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
   Contributing authors: Paul Crozier (SNL)
                         Alexander Stukowski
------------------------------------------------------------------------- */

#include "fix_atom_swap.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "compute.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "graphics.h"
#include "group.h"
#include "improper.h"
#include "kspace.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "pair_hybrid.h"
#include "random_park.h"
#include "region.h"
#include "suffix.h"
#include "update.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {
// run a cleanup function when leaving a scope, also on errors (exceptions)
template <typename F> class ScopeExit {
 public:
  explicit ScopeExit(F f) : func(f) {}
  ~ScopeExit() { func(); }
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;

 private:
  F func;
};

// streams of the counter based random numbers of checkerboard sweeps
enum { STREAM_SHIFT = 1, STREAM_ORDER, STREAM_EDGE, STREAM_THIN, STREAM_ACCEPT };

// SplitMix64 finalizer
inline uint64_t mix64(uint64_t z)
{
  z += UINT64_C(0x9e3779b97f4a7c15);
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

// lazy factor for cells with a single selectable edge: with probability 1/2
// such a cell makes a null move, so the chain cannot alternate with period 2
inline double lazy_factor(bigint m)
{
  return (m == 1) ? 2.0 : 1.0;
}

// Metropolis-Hastings acceptance test u < exp(arg) without overflow
inline bool accept_test(double u, double arg)
{
  if (arg >= 0.0) return true;
  return u < exp(arg);
}

// true if word is a keyword of fix atom/swap, so that type labels (which
// start with a letter) can be told apart from the next keyword
bool is_swap_keyword(const char *word)
{
  static const char *const keywords[] = {"region", "ke",       "semi-grand", "types", "mu",
                                         "rmax",   "proposal", "parallel",   "energy"};
  for (const char *keyword : keywords)
    if (strcmp(word, keyword) == 0) return true;
  return false;
}
}    // namespace

/* ---------------------------------------------------------------------- */

FixAtomSwap::FixAtomSwap(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), region(nullptr), idregion(nullptr), type_list(nullptr), mu(nullptr),
    qtype(nullptr), mtype(nullptr), sqrt_mass_ratio(nullptr), local_swap_iatom_list(nullptr),
    local_swap_jatom_list(nullptr), local_swap_atom_list(nullptr), random_equal(nullptr),
    random_unequal(nullptr), c_pe(nullptr), imgobjs(nullptr), imgparms(nullptr), weight(nullptr),
    type2swap(nullptr), pair_local(nullptr), swap_list(nullptr)
{
  if (narg < 10) utils::missing_cmd_args(FLERR, "fix atom/swap", error);

  dynamic_group_allow = 1;

  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;
  restart_global = 1;
  time_depend = 1;

  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE) {
    vizsteps = 0;
  } else {
    vizsteps = 1000;
  }

  // required args

  nevery = utils::inumeric(FLERR, arg[3], false, lmp);
  ncycles = utils::inumeric(FLERR, arg[4], false, lmp);
  seed = utils::inumeric(FLERR, arg[5], false, lmp);
  double temperature = utils::numeric(FLERR, arg[6], false, lmp);

  if (nevery <= 0) error->all(FLERR, 3, "Illegal fix atom/swap command nevery value");
  if (ncycles < 0) error->all(FLERR, 4, "Illegal fix atom/swap command ncycles value");
  if (seed <= 0) error->all(FLERR, 5, "Illegal fix atom/swap command random seed");
  if (temperature <= 0.0) error->all(FLERR, 6, "Illegal fix atom/swap command temperature value");

  beta = 1.0 / (force->boltz * temperature);

  memory->create(type_list, atom->ntypes, "atom/swap:type_list");
  memory->create(mu, atom->ntypes + 1, "atom/swap:mu");
  for (int i = 0; i <= atom->ntypes; i++) mu[i] = 0.0;

  // defaults for extension keywords

  new_syntax = 0;
  proposal_style = PROPOSAL_LEGACY;
  proposal_flag = 0;
  energy_mode = ENERGY_UNSET;
  parallel_mode = PARALLEL_SERIAL;
  rmax = 0.0;
  wmax = 1.0;
  use_local = local_now = 0;
  local_msg = -1;
  dynamic_msg = checker_msg = eval_msg = 0;
  cb_msg_dims[0] = cb_msg_dims[1] = cb_msg_dims[2] = 0;
  cb_dims[0] = cb_dims[1] = cb_dims[2] = 0;
  caps.supported = caps.supports_charge = 0;
  caps.dependency_radius = caps.conflict_radius = 0.0;
  comm_mode = COMM_TYPEQ;
  map_temporary = 0;
  local_active = 0;
  swap_hooks_active = 0;
  hooks_invoked = 0;
  energy_valid = 0;
  nswap_null = 0.0;
  verify_maxdev = 0.0;
  gcount = 0;
  ecount = ebefore = 0;

  // read options from end of input line

  options(narg - 7, &arg[7]);

  // without any extension keyword the behavior is identical to the original fix

  if (energy_mode == ENERGY_UNSET) energy_mode = new_syntax ? ENERGY_AUTO : ENERGY_FULL;
  if (!proposal_flag && (rmax > 0.0)) proposal_style = PROPOSAL_HETERO;
  if (semi_grand_flag && (proposal_flag || (rmax > 0.0)))
    error->all(FLERR, "Fix atom/swap keywords proposal and rmax require semi-grand no");
  if (!weight_args_value.empty() && (proposal_style != PROPOSAL_GEOMETRIC))
    error->all(FLERR, "Fix atom/swap proposal weights require proposal geometric");
  if (new_syntax) size_vector = 4;

  memory->create(weight, atom->ntypes + 1, atom->ntypes + 1, "atom/swap:weight");
  memory->create(type2swap, atom->ntypes + 1, "atom/swap:type2swap");

  // random number generator, same for all procs

  random_equal = new RanPark(lmp, seed);

  // random number generator, not the same for all procs

  random_unequal = new RanPark(lmp, seed);
  random_unequal->reset_rank(seed, comm->me);

  // set up reneighboring

  force_reneighbor = 1;
  next_reneighbor = update->ntimestep + 1;

  // zero out counters

  mc_active = 0;

  nswap_attempts = 0.0;
  nswap_successes = 0.0;

  atom_swap_nmax = 0;
  local_swap_atom_list = nullptr;
  local_swap_iatom_list = nullptr;
  local_swap_jatom_list = nullptr;

  // set comm size needed by this Fix

  if (atom->q_flag)
    comm_forward = 2;
  else
    comm_forward = 1;
}

/* ---------------------------------------------------------------------- */

FixAtomSwap::~FixAtomSwap()
{
  if (copymode) return;
  memory->destroy(type_list);
  memory->destroy(mu);
  memory->destroy(qtype);
  memory->destroy(mtype);
  memory->destroy(sqrt_mass_ratio);
  memory->destroy(local_swap_atom_list);
  memory->destroy(local_swap_iatom_list);
  memory->destroy(local_swap_jatom_list);
  memory->destroy(weight);
  memory->destroy(type2swap);
  delete[] idregion;
  delete random_equal;
  delete random_unequal;
  memory->destroy(imgobjs);
  memory->destroy(imgparms);
}

/* ----------------------------------------------------------------------
   parse optional parameters at end of input line
------------------------------------------------------------------------- */

void FixAtomSwap::options(int narg, char **arg)
{
  if (narg < 0) error->all(FLERR, "Illegal fix atom/swap command");

  ke_flag = 1;
  semi_grand_flag = 0;
  nswaptypes = 0;
  nmutypes = 0;

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "region") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      region = domain->get_region_by_id(arg[iarg + 1]);
      if (!region) error->all(FLERR, "Region {} for fix atom/swap does not exist", arg[iarg + 1]);
      delete[] idregion;
      idregion = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      ke_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "semi-grand") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      semi_grand_flag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "types") == 0) {
      if (iarg + 3 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        if (nswaptypes >= atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        type_list[nswaptypes] = utils::expand_type_int(FLERR, arg[iarg], Atom::ATOM, lmp);
        nswaptypes++;
        iarg++;
      }
    } else if (strcmp(arg[iarg], "mu") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix atom/swap command");
      iarg++;
      while (iarg < narg) {
        if (isalpha(arg[iarg][0])) break;
        nmutypes++;
        if (nmutypes > atom->ntypes) error->all(FLERR, "Illegal fix atom/swap command");
        mu[nmutypes] = utils::numeric(FLERR, arg[iarg], false, lmp);
        iarg++;
      }
    } else if (strcmp(arg[iarg], "rmax") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix atom/swap rmax", error);
      rmax = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (!std::isfinite(rmax) || (rmax < 0.0))
        error->all(FLERR, "Fix atom/swap rmax must be a finite number >= 0.0");
      new_syntax = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "proposal") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix atom/swap proposal", error);
      if (strcmp(arg[iarg + 1], "hetero") == 0)
        proposal_style = PROPOSAL_HETERO;
      else if (strcmp(arg[iarg + 1], "geometric") == 0)
        proposal_style = PROPOSAL_GEOMETRIC;
      else
        error->all(FLERR, "Unknown fix atom/swap proposal style {}", arg[iarg + 1]);
      proposal_flag = 1;
      new_syntax = 1;
      iarg += 2;
      if ((iarg < narg) && (strcmp(arg[iarg], "weights") == 0)) {
        iarg++;
        while ((iarg + 3 <= narg) && !is_swap_keyword(arg[iarg])) {
          weight_args_type.push_back(utils::expand_type_int(FLERR, arg[iarg], Atom::ATOM, lmp));
          weight_args_type.push_back(utils::expand_type_int(FLERR, arg[iarg + 1], Atom::ATOM, lmp));
          double w = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
          if (!std::isfinite(w) || (w < 0.0))
            error->all(FLERR, "Fix atom/swap proposal weights must be finite numbers >= 0.0");
          weight_args_value.push_back(w);
          iarg += 3;
        }
        if (weight_args_value.empty())
          error->all(FLERR, "Fix atom/swap proposal weights requires at least one triplet");
      }
    } else if (strcmp(arg[iarg], "parallel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix atom/swap parallel", error);
      if (strcmp(arg[iarg + 1], "serial") == 0)
        parallel_mode = PARALLEL_SERIAL;
      else if (strcmp(arg[iarg + 1], "checkerboard") == 0)
        parallel_mode = PARALLEL_CHECKERBOARD;
      else
        error->all(FLERR, "Unknown fix atom/swap parallel setting {}", arg[iarg + 1]);
      new_syntax = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "energy") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix atom/swap energy", error);
      if (strcmp(arg[iarg + 1], "auto") == 0)
        energy_mode = ENERGY_AUTO;
      else if (strcmp(arg[iarg + 1], "local") == 0)
        energy_mode = ENERGY_LOCAL;
      else if (strcmp(arg[iarg + 1], "full") == 0)
        energy_mode = ENERGY_FULL;
      else if (strcmp(arg[iarg + 1], "verify") == 0)
        energy_mode = ENERGY_VERIFY;
      else
        error->all(FLERR, "Unknown fix atom/swap energy setting {}", arg[iarg + 1]);
      new_syntax = 1;
      iarg += 2;
    } else
      error->all(FLERR, "Illegal fix atom/swap command");
  }
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "vizsteps") == 0) {
    if (narg < 2) utils::missing_cmd_args(FLERR, "fix_modify atom/swap", error);
    vizsteps = utils::inumeric(FLERR, arg[1], false, lmp);
    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::setmask()
{
  int mask = 0;
  mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::init()
{
  if ((atom->mass != nullptr) && (atom->rmass != nullptr) && (comm->me == 0))
    error->warning(FLERR, "Fix atom/swap will use per-atom masses for velocity rescaling");

  c_pe = modify->get_compute_by_id("thermo_pe");

  int *type = atom->type;

  if (nswaptypes < 2)
    error->all(FLERR, Error::NOLASTLINE,
               "Must specify at least 2 atom types in fix atom/swap command");

  if (semi_grand_flag) {
    if (nswaptypes != nmutypes)
      error->all(FLERR, Error::NOLASTLINE, "Need nswaptypes mu values in fix atom/swap command");
  } else {
    if ((nswaptypes != 2) && (proposal_style == PROPOSAL_LEGACY))
      error->all(FLERR, Error::NOLASTLINE,
                 "Exactly 2 atom types must be used without semi-grand keyword in fix atom/swap "
                 "unless keyword proposal or rmax is used");
    if (nmutypes != 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "Mu not allowed when not using semi-grand in fix atom/swap command");
  }

  // must have a pair style and not use INTEL package

  if (!force->pair) error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap requires a pair style");
  if (force->pair && (force->pair->suffix_flag & Suffix::INTEL))
    error->all(FLERR, Error::NOLASTLINE, "Fix {} is not compatible with /intel pair styles", style);

  // check if constraints for hybrid pair styles are fulfilled

  if (utils::strmatch(force->pair_style, "^hybrid")) {
    auto *hybrid = dynamic_cast<PairHybrid *>(force->pair);
    if (hybrid) {
      for (int i = 0; i < nswaptypes - 1; ++i) {
        int type1 = type_list[i];
        for (int j = i + 1; j < nswaptypes; ++j) {
          int type2 = type_list[j];
          if (hybrid->nmap[type1][type1] != hybrid->nmap[type2][type2])
            error->all(FLERR, Error::NOLASTLINE,
                       "Pair {} substyles for atom types {} and {} are not compatible",
                       force->pair_style, type1, type2);
          for (int k = 0; k < hybrid->nmap[type1][type1]; ++k) {
            if (hybrid->map[type1][type1][k] != hybrid->map[type2][type2][k])
              error->all(FLERR, Error::NOLASTLINE,
                         "Pair {} substyles for atom types {} and {} are not compatible",
                         force->pair_style, type1, type2);
          }
        }
      }
    }
  }

  // set index and check validity of region

  if (idregion) {
    region = domain->get_region_by_id(idregion);
    if (!region)
      error->all(FLERR, Error::NOLASTLINE, "Region {} for fix atom/swap does not exist", idregion);
  }

  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    if (type_list[iswaptype] <= 0 || type_list[iswaptype] > atom->ntypes)
      error->all(FLERR, "Invalid atom type in fix atom/swap command");

  // this is only required for non-semi-grand

  if (atom->q_flag && !semi_grand_flag) {
    double qmax, qmin;
    int firstall, first;
    memory->destroy(qtype);
    memory->create(qtype, nswaptypes, "atom/swap:qtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first) {
              qtype[iswaptype] = atom->q[i];
              first = 0;
            } else if (qtype[iswaptype] != atom->q[i])
              error->one(FLERR, "All atoms of a swapped type must have the same charge.");
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall)
        error->all(FLERR,
                   "At least one atom of each swapped type must be present to define charges.");
      if (first) qtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) qtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&qtype[iswaptype], &qmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (qmax != qmin) error->all(FLERR, "All atoms of a swapped type must have same charge.");
      qtype[iswaptype] = qmax;
    }
  }

  // if we have per-atom masses, check that rmass is consistent with type,
  // and set per-type mass to that value
  if ((atom->rmass != nullptr) && !semi_grand_flag) {
    double mmax, mmin;
    int firstall, first;
    memory->destroy(mtype);
    memory->create(mtype, nswaptypes, "atom/swap:mtype");
    for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
      first = 1;
      for (int i = 0; i < atom->nlocal; i++) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[iswaptype]) {
            if (first > 0) {
              mtype[iswaptype] = atom->rmass[i];
              first = 0;
            } else if (mtype[iswaptype] != atom->rmass[i])
              first = -1;
          }
        }
      }
      MPI_Allreduce(&first, &firstall, 1, MPI_INT, MPI_MIN, world);
      if (firstall < 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "All atoms of a swapped type must have the same per-atom mass");
      if (firstall > 0)
        error->all(FLERR, Error::NOLASTLINE,
                   "At least one atom of each swapped type must be present to define masses");
      if (first) mtype[iswaptype] = -DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmax, 1, MPI_DOUBLE, MPI_MAX, world);
      if (first) mtype[iswaptype] = DBL_MAX;
      MPI_Allreduce(&mtype[iswaptype], &mmin, 1, MPI_DOUBLE, MPI_MIN, world);
      if (mmax != mmin)
        error->all(FLERR, Error::NOLASTLINE, "All atoms of a swapped type must have same mass.");
      mtype[iswaptype] = mmax;
    }
  }

  memory->destroy(sqrt_mass_ratio);
  memory->create(sqrt_mass_ratio, atom->ntypes + 1, atom->ntypes + 1, "atom/swap:sqrt_mass_ratio");
  if (atom->rmass != nullptr) {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++) sqrt_mass_ratio[itype][jtype] = 1.0;
    if (mtype) {
      for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
        int itype = type_list[iswaptype];
        for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++) {
          int jtype = type_list[jswaptype];
          sqrt_mass_ratio[itype][jtype] = sqrt(mtype[iswaptype] / mtype[jswaptype]);
        }
      }
    }
  } else {
    for (int itype = 1; itype <= atom->ntypes; itype++)
      for (int jtype = 1; jtype <= atom->ntypes; jtype++)
        sqrt_mass_ratio[itype][jtype] = sqrt(atom->mass[itype] / atom->mass[jtype]);
  }

  // check to see if itype and jtype cutoffs are the same
  // if not, reneighboring will be needed between swaps

  double **cutsq = force->pair->cutsq;
  unequal_cutoffs = false;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      for (int ktype = 1; ktype <= atom->ntypes; ktype++)
        if (cutsq[type_list[iswaptype]][ktype] != cutsq[type_list[jswaptype]][ktype])
          unequal_cutoffs = true;

  // no further setup without extension keywords

  if (!new_syntax) return;

  if (!atom->tag_enable)
    error->all(FLERR, Error::NOLASTLINE,
               "Fix atom/swap keywords energy, rmax, proposal and parallel require atom IDs");

  // map from atom type to swap type index

  for (int itype = 0; itype <= atom->ntypes; itype++) type2swap[itype] = -1;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++) {
    if (type2swap[type_list[iswaptype]] >= 0)
      error->all(FLERR, Error::NOLASTLINE,
                 "Atom type {} is listed more than once in fix atom/swap command",
                 type_list[iswaptype]);
    type2swap[type_list[iswaptype]] = iswaptype;
  }

  // proposal weights: default 1.0 for all pairs of different swap types

  for (int itype = 0; itype <= atom->ntypes; itype++)
    for (int jtype = 0; jtype <= atom->ntypes; jtype++) weight[itype][jtype] = 0.0;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      if (iswaptype != jswaptype) weight[type_list[iswaptype]][type_list[jswaptype]] = 1.0;
  for (std::size_t n = 0; n < weight_args_value.size(); n++) {
    int itype = weight_args_type[2 * n];
    int jtype = weight_args_type[2 * n + 1];
    if ((itype < 1) || (itype > atom->ntypes) || (jtype < 1) || (jtype > atom->ntypes) ||
        (type2swap[itype] < 0) || (type2swap[jtype] < 0))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap proposal weights must refer to swap types");
    if (itype == jtype)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap proposal weights must refer to two different types");
    weight[itype][jtype] = weight[jtype][itype] = weight_args_value[n];
  }
  wmax = 0.0;
  for (int iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
    for (int jswaptype = 0; jswaptype < nswaptypes; jswaptype++)
      wmax = MAX(wmax, weight[type_list[iswaptype]][type_list[jswaptype]]);
  if ((proposal_style == PROPOSAL_GEOMETRIC) && (wmax <= 0.0))
    error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap proposal weights must not all be zero");

  // communication cutoff: neighbor->cutneighmax is not yet updated here

  const double cutcomm = MAX(comm->cutghostuser, force->pair->cutforce + neighbor->skin);

  // checks for distance limited proposals

  if (rmax > 0.0) {
    if (neighbor->style == Neighbor::MULTI)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap rmax requires neighbor style 'bin' or 'nsq'");
    if (neighbor->includegroup)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap rmax is not compatible with "
                 "neigh_modify include");
    // the neighbor list for pairs within rmax follows the exclusion settings
    // and would silently miss some of the pairs
    if ((neighbor->nex_type > 0) || (neighbor->nex_group > 0) || (neighbor->nex_mol > 0))
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap rmax is not compatible with neigh_modify exclude");
    if (rmax > cutcomm)
      error->all(FLERR, Error::NOLASTLINE,
                 "Fix atom/swap rmax {} exceeds the communication cutoff {}; "
                 "use comm_modify cutoff to increase it",
                 rmax, cutcomm);
  }

  // decide whether local energy changes from the pair style can be used

  use_local = 0;
  pair_local = nullptr;
  int msg = 0;
  std::string reason;
  if (energy_mode != ENERGY_FULL) {
    if (local_energy_check(reason)) {
      use_local = 1;
      msg = 1;
    } else if (energy_mode == ENERGY_AUTO) {
      msg = 2;
    } else {
      error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap energy {} is not possible: {}",
                 (energy_mode == ENERGY_LOCAL) ? "local" : "verify", reason);
    }
  }
  if ((comm->me == 0) && (msg != local_msg)) {
    if (msg == 1)
      utils::logmesg(lmp,
                     "Fix atom/swap: using local energy changes from pair style {} "
                     "(dependency radius {:.8g}, conflict radius {:.8g}){}\n",
                     force->pair_style, caps.dependency_radius, caps.conflict_radius,
                     (energy_mode == ENERGY_VERIFY) ? " verified against full energies" : "");
    else if (msg == 2)
      error->warning(FLERR, "Fix atom/swap: using full energy evaluation: {}", reason);
  }
  local_msg = msg;
  dynamic_msg = 0;
  checker_msg = 0;
  eval_msg = 0;

  // request an occasional full neighbor list for proposals and local energies.
  // the cutoff is uniform for all pairs of atom types, so it covers the
  // dependency radius also for unequal per-type pair cutoffs.  a small
  // margin keeps pairs exactly at the cutoff of the pair style or at rmax,
  // but the cutoff must not exceed the communication cutoff.

  if (use_local || (rmax > 0.0)) {
    double cut = rmax;
    if (use_local) cut = MAX(cut, MAX(caps.dependency_radius, force->pair->cutforce));
    cut = MIN(cut * (1.0 + 1.0e-9), MAX(cut, cutcomm));
    request_swap_list(cut);
  }
}

/* ----------------------------------------------------------------------
   request the occasional full neighbor list with a uniform cutoff
   may be replaced by accelerated variants to set more request flags
------------------------------------------------------------------------- */

void FixAtomSwap::request_swap_list(double cut)
{
  auto *req = neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
  req->set_cutoff_fixed(cut);
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::init_list(int /*id*/, NeighList *ptr)
{
  swap_list = ptr;
}

/* ----------------------------------------------------------------------
   hooks for local energy changes, may be replaced by accelerated variants
------------------------------------------------------------------------- */

PairSwapLocal *FixAtomSwap::local_pair_interface()
{
  return force->pair->swap_local();
}

int FixAtomSwap::swap_eval_batch(int n, const SwapProposal *proposals, double *dE)
{
  return pair_local->swap_local_delta_energy_batch(n, proposals, dE);
}

void FixAtomSwap::swap_commit_batch(int n, const SwapProposal *proposals, const int *accepted)
{
  pair_local->swap_local_commit_batch(n, proposals, accepted);
}

void FixAtomSwap::swap_forward_comm()
{
  comm->forward_comm(this);
}

/* ----------------------------------------------------------------------
   apply accepted proposals: new types and charges for all owned and ghost
   images on this proc, velocity rescaling and visualization data for
   owned atoms.  the proposals are known on all procs.
------------------------------------------------------------------------- */

void FixAtomSwap::swap_apply_accepted(int n, const SwapProposal *proposals, const int *accepted)
{
  int *type = atom->type;
  double *q = atom->q;
  double **v = atom->v;
  const int nlocal = atom->nlocal;

  for (int k = 0; k < n; k++) {
    if (!accepted[k]) continue;
    const SwapProposal &p = proposals[k];
    for (int m = 0; m < p.count; m++) {
      for (int a = atom->map(p.tag[m]); a >= 0; a = atom->sametag[a]) {
        type[a] = p.newtype[m];
        if (atom->q_flag && p.qflag) q[a] = p.newq[m];
      }
      const int i = atom->map(p.tag[m]);
      if ((i < 0) || (i >= nlocal)) continue;
      if (ke_flag) {
        const double factor = sqrt_mass_ratio[p.oldtype[m]][p.newtype[m]];
        v[i][0] *= factor;
        v[i][1] *= factor;
        v[i][2] *= factor;
      }
      if (vizsteps > 0) vizatoms[p.tag[m]] = std::make_pair(vizsteps, p.oldtype[m]);
    }
  }
}

/* ----------------------------------------------------------------------
   check if exact local energy changes from the pair style can replace the
   full energy evaluation.  only checks of settings that do not change
   during a run.  return 1 if yes, else 0 and the reason.
------------------------------------------------------------------------- */

int FixAtomSwap::local_energy_check(std::string &reason)
{
  pair_local = nullptr;
  if (!atom->tag_enable) {
    reason = "atom IDs are required";
    return 0;
  }
  if (atom->rmass_flag) {
    reason = "per-atom masses are not supported";
    return 0;
  }
  if (force->kspace) {
    reason = "a kspace style is defined";
    return 0;
  }
  if (force->bond || force->angle || force->dihedral || force->improper) {
    reason = "bonded interactions are defined";
    return 0;
  }
  if (atom->molecular != Atom::ATOMIC) {
    for (int m = 1; m < 4; m++) {
      if ((force->special_lj[m] != 1.0) || (force->special_coul[m] != 1.0)) {
        reason = "special bond factors other than 1.0 are used";
        return 0;
      }
    }
  }
  for (const auto &ifix : modify->get_fix_list()) {
    if (ifix->energy_global_flag && ifix->thermo_energy) {
      reason = fmt::format("fix {} contributes to the potential energy", ifix->id);
      return 0;
    }
  }

  // check the exclusion settings directly: neighbor->exclude is only
  // updated in Neighbor::init(), which runs after Modify::init()

  if ((neighbor->nex_type > 0) || (neighbor->nex_group > 0) || (neighbor->nex_mol > 0)) {
    reason = "neigh_modify exclude is used";
    return 0;
  }
  if (neighbor->includegroup) {
    reason = "neigh_modify include is used";
    return 0;
  }
  if (neighbor->style == Neighbor::MULTI) {
    reason = "neighbor style multi is used";
    return 0;
  }
  if (comm->style != Comm::BRICK) {
    reason = "local energy requires comm_style brick";
    return 0;
  }
  if (comm->mode == Comm::MULTI) {
    reason = "comm_modify mode multi is used";
    return 0;
  }
  if (strstr(update->integrate_style, "respa") != nullptr) {
    reason = "run_style respa is used";
    return 0;
  }
  if (force->pair->tail_flag) {
    reason = "pair_modify tail yes is used";
    return 0;
  }
  if (!force->pair->compute_flag) {
    reason = "pair_modify compute no is used";
    return 0;
  }

  PairSwapLocal *plocal = local_pair_interface();
  if (!plocal) {
    reason = fmt::format("pair style {} does not support local swap energies", force->pair_style);
    return 0;
  }
  std::string why;
  caps = plocal->swap_local_caps(why);
  if (!caps.supported) {
    reason = fmt::format("pair style {}: {}", force->pair_style, why);
    return 0;
  }
  if (!caps.supports_charge && atom->q_flag && !semi_grand_flag && qtype) {
    for (int iswaptype = 1; iswaptype < nswaptypes; iswaptype++) {
      if (qtype[iswaptype] != qtype[0]) {
        reason = fmt::format("pair style {} does not support charge changes", force->pair_style);
        return 0;
      }
    }
  }
  const double cutcomm = MAX(comm->cutghostuser, force->pair->cutforce + neighbor->skin);
  if (caps.dependency_radius > cutcomm) {
    reason = fmt::format("the dependency radius {:.8g} of the pair style exceeds the "
                         "communication cutoff {:.8g} (use comm_modify cutoff)",
                         caps.dependency_radius, cutcomm);
    return 0;
  }
  pair_local = plocal;
  return 1;
}

/* ----------------------------------------------------------------------
   check settings that may change during a run, called after ghost atoms
   were acquired: ghost atoms must cover max(dependency radius, rmax).
   return 1 if ok, else 0 and the reason (same result on all procs).
------------------------------------------------------------------------- */

int FixAtomSwap::local_dynamic_check(std::string &reason)
{
  const double rneed = MAX(caps.dependency_radius, rmax);
  if (ghost_cutoff() * (1.0 + 1.0e-10) >= rneed) return 1;
  reason = fmt::format("local energy requires comm->cutghost >= {:.8g} (use comm_modify "
                       "cutoff {:.8g} or larger)",
                       rneed, rneed);
  return 0;
}

/* ----------------------------------------------------------------------
   smallest distance up to which ghost atoms are acquired in any dimension
   on any proc, in distance units also for triclinic boxes
------------------------------------------------------------------------- */

double FixAtomSwap::ghost_cutoff()
{
  const double *h_inv = domain->h_inv;
  double length[3];
  length[0] = sqrt(h_inv[0] * h_inv[0] + h_inv[5] * h_inv[5] + h_inv[4] * h_inv[4]);
  length[1] = sqrt(h_inv[1] * h_inv[1] + h_inv[3] * h_inv[3]);
  length[2] = h_inv[2];

  double have = DBL_MAX;
  for (int d = 0; d < domain->dimension; d++) {
    double cut = comm->cutghost[d];
    if (domain->triclinic) cut /= length[d];
    have = MIN(have, cut);
  }
  double haveall = have;
  MPI_Allreduce(&have, &haveall, 1, MPI_DOUBLE, MPI_MIN, world);
  return haveall;
}

/* ----------------------------------------------------------------------
   the pair style could not compute local energy changes on some proc.
   must be called on all procs.  stop with energy local and verify, else
   end the use of local energy changes for the rest of this invocation
   and continue with full energies.
------------------------------------------------------------------------- */

void FixAtomSwap::local_eval_failed()
{
  if (energy_mode != ENERGY_AUTO)
    error->all(FLERR, Error::NOLASTLINE,
               "Fix atom/swap energy {}: pair style {} could not compute local energy changes",
               (energy_mode == ENERGY_LOCAL) ? "local" : "verify", force->pair_style);
  if (local_active) {
    local_active = 0;
    pair_local->swap_local_end();
  }
  if (swap_hooks_active) {
    swap_hooks_active = 0;
    swap_end_hooks();
  }
  local_now = 0;

  // neighbor lists were not updated after swaps with local energy changes

  rebuild_lists_if_needed();
  energy_stored = energy_full();
  energy_valid = 1;
  if ((comm->me == 0) && !eval_msg)
    error->warning(FLERR,
                   "Fix atom/swap: pair style {} could not compute local energy changes; "
                   "using full energy evaluation for the rest of this step",
                   force->pair_style);
  eval_msg = 1;
}

/* ----------------------------------------------------------------------
   check if a checkerboard sweep is possible for the current box and set
   up the grid: an even number >= 2 of cells in each dimension, each wider
   than max(rmax, conflict radius).  return 1 if yes, else 0 and the reason.
------------------------------------------------------------------------- */

int FixAtomSwap::checkerboard_check(std::string &reason)
{
  if (!local_now) {
    reason = "checkerboard sweeps require local energy changes";
    return 0;
  }
  if (semi_grand_flag) {
    reason = "checkerboard sweeps require semi-grand no";
    return 0;
  }
  if (rmax <= 0.0) {
    reason = "checkerboard sweeps require rmax > 0.0";
    return 0;
  }
  if (domain->dimension != 3) {
    reason = "checkerboard sweeps require a 3d system";
    return 0;
  }
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic) {
    reason = "checkerboard sweeps require a fully periodic box";
    return 0;
  }

  // perpendicular box heights, also for triclinic boxes

  const double *h_inv = domain->h_inv;
  double height[3];
  height[0] = 1.0 / sqrt(h_inv[0] * h_inv[0] + h_inv[5] * h_inv[5] + h_inv[4] * h_inv[4]);
  height[1] = 1.0 / sqrt(h_inv[1] * h_inv[1] + h_inv[3] * h_inv[3]);
  height[2] = 1.0 / h_inv[2];

  const double wmin = MAX(rmax, caps.conflict_radius) * (1.0 + 1.0e-10);
  for (int d = 0; d < 3; d++) {
    const double ncell = std::floor(height[d] / wmin);
    if (!(ncell >= 2.0)) {
      reason = fmt::format("box height {:.8g} in dimension {} is less than two cells of width "
                           "{:.8g}",
                           height[d], d + 1, wmin);
      return 0;
    }
    if (ncell > (double) MAXSMALLINT) {
      reason = "too many checkerboard cells";
      return 0;
    }
    int nd = static_cast<int>(ncell);
    nd -= (nd & 1);
    cb_dims[d] = nd;
  }

  // the cell IDs must fit into a bigint

  const double ncells = (double) cb_dims[0] * (double) cb_dims[1] * (double) cb_dims[2];
  if (ncells > (double) MAXBIGINT / 4.0) {
    reason = "too many checkerboard cells";
    return 0;
  }
  return 1;
}

/* ----------------------------------------------------------------------
   attempt Monte Carlo swaps
------------------------------------------------------------------------- */

void FixAtomSwap::pre_exchange()
{
  // just return if should not be called on this timestep

  if (next_reneighbor != update->ntimestep) return;

  // mc_active and all per-invocation resources are reset on any exit path

  mc_active = 1;
  bool restore_shrink_box = false;
  double saved_boxlo[3], saved_boxhi[3];
  auto cleanup = [this, &restore_shrink_box, &saved_boxlo, &saved_boxhi]() {
    end_mc_invocation();
    if (restore_shrink_box) {
      for (int d = 0; d < 3; d++) {
        domain->boxlo[d] = saved_boxlo[d];
        domain->boxhi[d] = saved_boxhi[d];
      }
      domain->set_global_box();
      domain->set_local_box();
      restore_shrink_box = false;
    }
  };
  ScopeExit<decltype(cleanup)> guard(cleanup);
  hooks_invoked = 1;
  pre_exchange_begin_hook();

  // ensure current system is ready to compute energy
  // Refresh box-dependent communication and bins before exchanging atoms.
  // For a triclinic shrink-wrapped box, save the bounds used as input to
  // reset_box() so Verlet can recompute the bounds from the same geometry.

  if (domain->triclinic) domain->x2lamda(atom->nlocal);
  domain->pbc();
  if (domain->box_change) {
    if (domain->nonperiodic == 2 && domain->triclinic) {
      for (int d = 0; d < 3; d++) {
        saved_boxlo[d] = domain->boxlo[d];
        saved_boxhi[d] = domain->boxhi[d];
      }
      restore_shrink_box = true;
    }
    domain->reset_box();
    comm->setup();
    if (neighbor->style) neighbor->setup_bins();
  }
  comm->exchange();
  comm->borders();
  if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  if (modify->has_fixes(FixConst::PRE_NEIGHBOR)) modify->pre_neighbor();
  neighbor->build(1);

  if (!new_syntax) {

    // energy_stored = energy of current state
    // will be updated after accepted swaps

    energy_stored = energy_full();

    // attempt Ncycle atom swaps

    int nsuccess = 0;
    if (semi_grand_flag) {
      update_semi_grand_atoms_list();
      for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
    } else {
      update_swap_atoms_list();
      for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
    }

    // update MC stats

    nswap_attempts += ncycles;
    nswap_successes += nsuccess;

  } else {

    // the pairs within rmax must be inside the ghost atom cutoff, which
    // may change with the box during a run

    if (rmax > 0.0) {
      const double have = ghost_cutoff();
      if (have * (1.0 + 1.0e-10) < rmax)
        error->all(FLERR, Error::NOLASTLINE,
                   "Fix atom/swap rmax {:.8g} exceeds the ghost atom cutoff {:.8g}; "
                   "use comm_modify cutoff to increase it",
                   rmax, have);
    }

    // decide if local energy changes can be used in this invocation

    local_now = use_local;
    if (local_now) {
      std::string why;
      if (!local_dynamic_check(why)) {
        if (energy_mode != ENERGY_AUTO)
          error->all(FLERR, Error::NOLASTLINE, "Fix atom/swap energy {} is not possible: {}",
                     (energy_mode == ENERGY_LOCAL) ? "local" : "verify", why);
        if ((comm->me == 0) && !dynamic_msg)
          error->warning(FLERR, "Fix atom/swap: using full energy evaluation: {}", why);
        dynamic_msg = 1;
        local_now = 0;
      }
    }

    // atom map (temporary if none exists), neighbor list and pair style caches

    if (atom->map_style == Atom::MAP_NONE) {
      map_temporary = 1;
      atom->map_init();
      atom->map_set();
    }
    if (local_now || (rmax > 0.0)) neighbor->build_one(swap_list);
    if (local_now) {
      swap_begin_hooks();
      swap_hooks_active = 1;
      pair_local->swap_local_begin(swap_list);
      local_active = 1;
    }

    // energy of current state, not needed when only local energy changes are used

    energy_valid = 0;
    if (!local_now || (energy_mode == ENERGY_VERIFY)) {
      energy_stored = energy_full();
      energy_valid = 1;
    }

    // attempt swaps

    int checker = 0;
    if (parallel_mode == PARALLEL_CHECKERBOARD) {
      std::string why;
      if (checkerboard_check(why))
        checker = 1;
      else {
        if ((comm->me == 0) && !checker_msg)
          error->warning(FLERR, "Fix atom/swap: using serial sweeps: {}", why);
        checker_msg = 1;
      }
    }

    bigint nattempt = ncycles;
    bigint nsuccess = 0;
    if (checker)
      nsuccess = checkerboard_sweep(nattempt);
    else
      nsuccess = serial_sweep();

    nswap_attempts += (double) nattempt;
    nswap_successes += (double) nsuccess;
  }

  next_reneighbor = update->ntimestep + nevery;

  cleanup();

  // if visualization support is enabled, age vizatoms and remove expired ones
  if (vizsteps > 0) {
    std::vector<tagint> eraseme;
    for (const auto &[key, data] : vizatoms) {
      int idx = atom->map(key);
      if ((idx < 0) || (data.first < 0)) {
        eraseme.push_back(key);
        continue;
      }
      vizatoms[key] = std::make_pair(data.first - nevery, data.second);
    }
    for (const auto &key : eraseme) vizatoms.erase(key);
  }
}

/* ----------------------------------------------------------------------
   release per-invocation resources and clear mc_active
   called on every exit path of pre_exchange(), may be called twice
------------------------------------------------------------------------- */

void FixAtomSwap::end_mc_invocation()
{
  if (local_active) {
    local_active = 0;
    pair_local->swap_local_end();
  }
  if (swap_hooks_active) {
    swap_hooks_active = 0;
    swap_end_hooks();
  }
  if (map_temporary) {
    atom->map_delete();
    atom->map_style = Atom::MAP_NONE;
    map_temporary = 0;
  }

  if (hooks_invoked) {
    hooks_invoked = 0;
    pre_exchange_end_hook();
  }
  mc_active = 0;
}

/* ----------------------------------------------------------------------
   serial sweep of ncycles swap attempts with the extension keywords
------------------------------------------------------------------------- */

bigint FixAtomSwap::serial_sweep()
{
  bigint nsuccess = 0;
  if (semi_grand_flag) {
    update_semi_grand_atoms_list();
    if (local_now)
      for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand_local();
    else
      for (int i = 0; i < ncycles; i++) nsuccess += attempt_semi_grand();
  } else if ((proposal_style == PROPOSAL_LEGACY) && !local_now) {
    update_swap_atoms_list();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap();
  } else {
    if (proposal_style == PROPOSAL_LEGACY)
      update_swap_atoms_list();
    else if (rmax > 0.0)
      build_graph();
    else
      update_type_lists();
    for (int i = 0; i < ncycles; i++) nsuccess += attempt_swap_general();
  }
  return nsuccess;
}

/* ----------------------------------------------------------------------
   attempt a semd-grand swap of a single atom
   compare before/after energy and accept/reject the swap
   NOTE: atom charges are assumed equal and so are not updated
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_semi_grand()
{
  if (nswap == 0) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  // pick a random atom and perform swap

  int itype, jtype, jswaptype;
  int i = pick_semi_grand_atom();
  if (i >= 0) {
    jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
    jtype = type_list[jswaptype];
    itype = atom->type[i];
    while (itype == jtype) {
      jswaptype = static_cast<int>(nswaptypes * random_unequal->uniform());
      jtype = type_list[jswaptype];
    }
    atom->type[i] = jtype;
  }

  // if unequal_cutoffs or type exclusions, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs || (neighbor->nex_type > 0)) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->has_fixes(FixConst::PRE_NEIGHBOR)) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy

  if (force->kspace) force->kspace->qsum_qsq();
  double energy_after = energy_full();

  int success = 0;
  if (i >= 0)
    if (random_unequal->uniform() <
        exp(beta * (energy_before - energy_after + mu[jtype] - mu[itype])))
      success = 1;

  int success_all = 0;
  MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  // swap accepted, return 1

  if (success_all) {
    update_semi_grand_atoms_list();
    energy_stored = energy_after;
    if (i >= 0) {
      if (ke_flag) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
      // record atom for which the type was swapped and store the old type
      if (vizsteps > 0) vizatoms[atom->tag[i]] = std::make_pair(vizsteps, itype);
    }
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped atom
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  if (i >= 0) atom->type[i] = itype;
  if (force->kspace) force->kspace->qsum_qsq();

  return 0;
}

/* ----------------------------------------------------------------------
   attempt a swap of a pair of atoms
   compare before/after energy and accept/reject the swap
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_swap()
{
  if ((niswap == 0) || (njswap == 0)) return 0;

  // pre-swap energy

  double energy_before = energy_stored;

  // pick a random pair of atoms
  // swap their properties

  int i = pick_i_swap_atom();
  int j = pick_j_swap_atom();
  int itype = type_list[0];
  int jtype = type_list[1];

  if (i >= 0) {
    atom->type[i] = jtype;
    if (atom->q_flag) atom->q[i] = qtype[1];
    if (atom->rmass != nullptr) atom->rmass[i] = mtype[1];
  }
  if (j >= 0) {
    atom->type[j] = itype;
    if (atom->q_flag) atom->q[j] = qtype[0];
    if (atom->rmass != nullptr) atom->rmass[j] = mtype[0];
  }

  // if unequal_cutoffs or type exclusions, call comm->borders() and rebuild neighbor list
  // else communicate ghost atoms
  // call to comm->exchange() is a no-op but clears ghost atoms

  if (unequal_cutoffs || (neighbor->nex_type > 0)) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    domain->pbc();
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
    if (modify->has_fixes(FixConst::PRE_NEIGHBOR)) modify->pre_neighbor();
    neighbor->build(1);
  } else {
    comm->forward_comm(this);
  }

  // post-swap energy

  double energy_after = energy_full();

  // swap accepted, return 1
  // if ke_flag, rescale atom velocities

  if (random_equal->uniform() < exp(beta * (energy_before - energy_after))) {
    update_swap_atoms_list();
    if (ke_flag) {
      if (i >= 0) {
        atom->v[i][0] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][1] *= sqrt_mass_ratio[itype][jtype];
        atom->v[i][2] *= sqrt_mass_ratio[itype][jtype];
      }
      if (j >= 0) {
        atom->v[j][0] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][1] *= sqrt_mass_ratio[jtype][itype];
        atom->v[j][2] *= sqrt_mass_ratio[jtype][itype];
      }
    }
    // record atoms for which the type was swapped and store the old types
    if (vizsteps > 0) {
      if (i >= 0) vizatoms[atom->tag[i]] = std::make_pair(vizsteps, itype);
      if (j >= 0) vizatoms[atom->tag[j]] = std::make_pair(vizsteps, jtype);
    }
    energy_stored = energy_after;
    return 1;
  }

  // swap not accepted, return 0
  // restore the swapped itype & jtype atoms
  // do not need to re-call comm->borders() and rebuild neighbor list
  //   since will be done on next cycle or in Verlet when this fix finishes

  if (i >= 0) {
    atom->type[i] = type_list[0];
    if (atom->q_flag) atom->q[i] = qtype[0];
    if (atom->rmass != nullptr) atom->rmass[i] = mtype[0];
  }
  if (j >= 0) {
    atom->type[j] = type_list[1];
    if (atom->q_flag) atom->q[j] = qtype[1];
    if (atom->rmass != nullptr) atom->rmass[j] = mtype[1];
  }

  return 0;
}

/* ----------------------------------------------------------------------
   compute system potential energy
------------------------------------------------------------------------- */

double FixAtomSwap::energy_full()
{
  // flag that we only need to compute the global energy
  int eflag = ENERGY_GLOBAL | ENERGY_ONLY;
  int vflag = VIRIAL_NONE;

  if (modify->has_fixes(FixConst::PRE_FORCE)) modify->pre_force(vflag);

  if (force->pair) force->pair->compute(eflag, vflag);

  if (atom->molecular != Atom::ATOMIC) {
    if (force->bond) force->bond->compute(eflag, vflag);
    if (force->angle) force->angle->compute(eflag, vflag);
    if (force->dihedral) force->dihedral->compute(eflag, vflag);
    if (force->improper) force->improper->compute(eflag, vflag);
  }

  if (force->kspace) force->kspace->compute(eflag, vflag);

  if (modify->has_fixes(FixConst::POST_FORCE_ANY)) modify->post_force(vflag);

  update->eflag_global = update->ntimestep;
  return c_pe->compute_scalar();
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_semi_grand_atom()
{
  int i = -1;
  int iwhichglobal = static_cast<int>(nswap * random_equal->uniform());
  if ((iwhichglobal >= nswap_before) && (iwhichglobal < nswap_before + nswap_local)) {
    int iwhichlocal = iwhichglobal - nswap_before;
    i = local_swap_atom_list[iwhichlocal];
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_i_swap_atom(int *pos)
{
  int i = -1;
  int iwhichglobal = static_cast<int>(niswap * random_equal->uniform());
  if (pos) *pos = -1;
  if ((iwhichglobal >= niswap_before) && (iwhichglobal < niswap_before + niswap_local)) {
    int iwhichlocal = iwhichglobal - niswap_before;
    i = local_swap_iatom_list[iwhichlocal];
    if (pos) *pos = iwhichlocal;
  }

  return i;
}

/* ----------------------------------------------------------------------
------------------------------------------------------------------------- */

int FixAtomSwap::pick_j_swap_atom(int *pos)
{
  int j = -1;
  int jwhichglobal = static_cast<int>(njswap * random_equal->uniform());
  if (pos) *pos = -1;
  if ((jwhichglobal >= njswap_before) && (jwhichglobal < njswap_before + njswap_local)) {
    int jwhichlocal = jwhichglobal - njswap_before;
    j = local_swap_jatom_list[jwhichlocal];
    if (pos) *pos = jwhichlocal;
  }

  return j;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_semi_grand_atoms_list()
{
  int nlocal = atom->nlocal;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_atom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_atom_list, atom_swap_nmax, "MCSWAP:local_swap_atom_list");
  }

  nswap_local = 0;

  if (region) {
    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          int itype = atom->type[i];
          int iswaptype;
          for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
            if (itype == type_list[iswaptype]) break;
          if (iswaptype == nswaptypes) continue;
          local_swap_atom_list[nswap_local] = i;
          nswap_local++;
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        int itype = atom->type[i];
        int iswaptype;
        for (iswaptype = 0; iswaptype < nswaptypes; iswaptype++)
          if (itype == type_list[iswaptype]) break;
        if (iswaptype == nswaptypes) continue;
        local_swap_atom_list[nswap_local] = i;
        nswap_local++;
      }
    }
  }

  MPI_Allreduce(&nswap_local, &nswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&nswap_local, &nswap_before, 1, MPI_INT, MPI_SUM, world);
  nswap_before -= nswap_local;
}

/* ----------------------------------------------------------------------
   update the list of gas atoms
------------------------------------------------------------------------- */

void FixAtomSwap::update_swap_atoms_list()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;
  double **x = atom->x;

  if (atom->nmax > atom_swap_nmax) {
    memory->destroy(local_swap_iatom_list);
    memory->destroy(local_swap_jatom_list);
    atom_swap_nmax = atom->nmax;
    memory->create(local_swap_iatom_list, atom_swap_nmax, "MCSWAP:local_swap_iatom_list");
    memory->create(local_swap_jatom_list, atom_swap_nmax, "MCSWAP:local_swap_jatom_list");
  }

  niswap_local = 0;
  njswap_local = 0;

  if (region) {

    for (int i = 0; i < nlocal; i++) {
      if (region->match(x[i][0], x[i][1], x[i][2]) == 1) {
        if (atom->mask[i] & groupbit) {
          if (type[i] == type_list[0]) {
            local_swap_iatom_list[niswap_local] = i;
            niswap_local++;
          } else if (type[i] == type_list[1]) {
            local_swap_jatom_list[njswap_local] = i;
            njswap_local++;
          }
        }
      }
    }

  } else {
    for (int i = 0; i < nlocal; i++) {
      if (atom->mask[i] & groupbit) {
        if (type[i] == type_list[0]) {
          local_swap_iatom_list[niswap_local] = i;
          niswap_local++;
        } else if (type[i] == type_list[1]) {
          local_swap_jatom_list[njswap_local] = i;
          njswap_local++;
        }
      }
    }
  }

  update_swap_atoms_counts();
}

/* ----------------------------------------------------------------------
   update global counts and offsets of the i,j swap atom lists
------------------------------------------------------------------------- */

void FixAtomSwap::update_swap_atoms_counts()
{
  MPI_Allreduce(&niswap_local, &niswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&niswap_local, &niswap_before, 1, MPI_INT, MPI_SUM, world);
  niswap_before -= niswap_local;

  MPI_Allreduce(&njswap_local, &njswap, 1, MPI_INT, MPI_SUM, world);
  MPI_Scan(&njswap_local, &njswap_before, 1, MPI_INT, MPI_SUM, world);
  njswap_before -= njswap_local;
}

/* ---------------------------------------------------------------------- */

int FixAtomSwap::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  int i, j, m;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;

  if (comm_mode == COMM_FLAG) {
    for (i = 0; i < n; i++) buf[m++] = eligible[list[i]];
    return m;
  }

  if (atom->q_flag) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
      buf[m++] = q[j];
    }
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = type[j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixAtomSwap::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m, last;

  int *type = atom->type;
  double *q = atom->q;

  m = 0;
  last = first + n;

  if (comm_mode == COMM_FLAG) {
    for (i = first; i < last; i++) eligible[i] = static_cast<int>(buf[m++]);
    return;
  }

  if (atom->q_flag) {
    for (i = first; i < last; i++) {
      type[i] = static_cast<int>(buf[m++]);
      q[i] = buf[m++];
    }
  } else {
    for (i = first; i < last; i++) type[i] = static_cast<int>(buf[m++]);
  }
}

/* ----------------------------------------------------------------------
  return acceptance ratio
------------------------------------------------------------------------- */

double FixAtomSwap::compute_vector(int n)
{
  if (n == 0) return nswap_attempts;
  if (n == 1) return nswap_successes;
  if (n == 2) return nswap_null;
  if (n == 3) return verify_maxdev;
  return 0.0;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixAtomSwap::memory_usage()
{
  double bytes = (double) atom_swap_nmax * sizeof(int);
  for (const auto &l : tlist) bytes += (double) l.capacity() * sizeof(int);
  bytes += (double) (elist.capacity() + eligible.capacity() + edge_i.capacity() +
                     edge_j.capacity() + edge_jimg.capacity() + adj_first.capacity() +
                     adj_edge.capacity() + het_list.capacity() + het_pos.capacity()) *
      sizeof(int);
  bytes += (double) edge_cell.capacity() * sizeof(bigint);
  for (const auto &cc : color_cells)
    bytes += (double) (cc.first.capacity() + cc.edges.capacity()) * sizeof(int) +
        (double) (cc.cell.capacity() + cc.mlocal.capacity()) * sizeof(bigint) +
        (double) cc.wlocal.capacity() * sizeof(double);
  return bytes;
}

/* ----------------------------------------------------------------------
   pack entire state of Fix into one write
------------------------------------------------------------------------- */

void FixAtomSwap::write_restart(FILE *fp)
{
  int nglobalvals = 6;
  int nlocalvals = comm->nprocs;
  int ntotalvals = nglobalvals + nlocalvals;
  std::vector<double> list(ntotalvals);
  int n = 0;
  list[n++] = random_equal->state();
  list[n++] = ubuf(next_reneighbor).d;
  list[n++] = nswap_attempts;
  list[n++] = nswap_successes;
  list[n++] = ubuf(update->ntimestep).d;
  list[n++] = nlocalvals;

  double state = random_unequal->state();
  MPI_Gather(&state, 1, MPI_DOUBLE, &list[n], 1, MPI_DOUBLE, 0, world);

  if (comm->me == 0) {
    int size = ntotalvals * sizeof(double);
    fwrite(&size, sizeof(int), 1, fp);
    fwrite(list.data(), sizeof(double), ntotalvals, fp);
  }
}

/* ----------------------------------------------------------------------
   use state info from restart file to restart the Fix
------------------------------------------------------------------------- */

void FixAtomSwap::restart(char *buf)
{
  int n = 0;
  auto *list = (double *) buf;

  // the seed of the fix command must not change: checkerboard sweeps use it
  // on all procs for their counter based random numbers

  int rseed = static_cast<int>(list[n++]);
  random_equal->reset(rseed);

  next_reneighbor = (bigint) ubuf(list[n++]).i;

  nswap_attempts = list[n++];
  nswap_successes = list[n++];

  bigint ntimestep_restart = (bigint) ubuf(list[n++]).i;
  if (ntimestep_restart != update->ntimestep)
    error->all(FLERR, "Must not reset timestep when restarting fix atom/swap");

  // read stored state of RNG unique to this process
  // if nprocs changed, derive a unique state from the state of proc 0

  auto nlocalvals = (int) list[n++];
  if (nlocalvals == comm->nprocs) {
    rseed = static_cast<int>(list[n + comm->me]);
    random_unequal->reset(rseed);
  } else {
    rseed = static_cast<int>(list[n]);
    random_unequal->reset_rank(rseed, comm->me);
  }
}

/* ----------------------------------------------------------------------
   extract variable which stores whether MC is active or not
     active = MC moves are taking place
     not active = normal MD is taking place
------------------------------------------------------------------------- */

void *FixAtomSwap::extract(const char *name, int &dim)
{
  if (strcmp(name, "mc_active") == 0) {
    dim = 0;
    return (void *) &mc_active;
  }
  return nullptr;
}

/* ----------------------------------------------------------------------
   provide graphics information to dump image to render spheres
   at the location of atoms that were involved in a reaction
------------------------------------------------------------------------- */

int FixAtomSwap::image(int *&objs, double **&parms)
{
  // no visualization without an atom map
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, Error::NOLASTLINE,
               "Cannot use fix atom/swap in dump image without an atom map");

  memory->destroy(imgobjs);
  memory->destroy(imgparms);

  int numobjs = vizatoms.size();
  int n = 0;
  if (numobjs > 0) {
    memory->create(imgobjs, numobjs, "atom/swap:imgobjs");
    memory->create(imgparms, numobjs, 5, "atom/swap:imgparms");

    int idx;
    const auto *const *const x = atom->x;
    for (const auto &[key, data] : vizatoms) {
      idx = atom->map(key);
      if (idx < 0) continue;
      imgobjs[n] = Graphics::SPHERE;
      imgparms[n][0] = data.second;    // use stored pre-swap atom type
      imgparms[n][1] = x[idx][0];
      imgparms[n][2] = x[idx][1];
      imgparms[n][3] = x[idx][2];
      imgparms[n][4] = 0.0;    // radius is set with fflag2 in dump image
      ++n;
    }
  }
  objs = imgobjs;
  parms = imgparms;
  return n;
}

/* ----------------------------------------------------------------------
   return 1 if owned atom i may take part in swaps (group, region, type)
------------------------------------------------------------------------- */

int FixAtomSwap::is_eligible(int i)
{
  if (!(atom->mask[i] & groupbit)) return 0;
  if (type2swap[atom->type[i]] < 0) return 0;
  if (region) {
    double **x = atom->x;
    if (region->match(x[i][0], x[i][1], x[i][2]) != 1) return 0;
  }
  return 1;
}

/* ----------------------------------------------------------------------
   selection weight of an edge between atoms of type itype and jtype:
   hetero: weight of the type pair, 0.0 for equal types
   geometric: 1.0 for all edges (weights are applied by thinning)
------------------------------------------------------------------------- */

double FixAtomSwap::selection_weight(int itype, int jtype) const
{
  if (proposal_style == PROPOSAL_HETERO) return (itype != jtype) ? weight[itype][jtype] : 0.0;
  return 1.0;
}

/* ----------------------------------------------------------------------
   set types and charges of all owned and ghost images of the edited atoms
   of a proposal on this proc to the new (flag = 1) or old (flag = 0) values
------------------------------------------------------------------------- */

void FixAtomSwap::set_proposal_types(const SwapProposal &p, int flag)
{
  int *type = atom->type;
  double *q = atom->q;
  for (int m = 0; m < p.count; m++) {
    const int t = flag ? p.newtype[m] : p.oldtype[m];
    const double qm = flag ? p.newq[m] : p.oldq[m];
    for (int a = atom->map(p.tag[m]); a >= 0; a = atom->sametag[a]) {
      type[a] = t;
      if (atom->q_flag && p.qflag) q[a] = qm;
    }
  }
  swap_apply_trial_hook(p, flag);
}

/* ----------------------------------------------------------------------
   sum of local energy changes over all procs.  return non-zero if the
   pair style could not compute them on any proc.
------------------------------------------------------------------------- */

int FixAtomSwap::local_delta(int n, const SwapProposal *proposals, double *dE)
{
  std::vector<double> buf(n + 1, 0.0);
  int fail = swap_eval_batch(n, proposals, buf.data());
  buf[n] = fail ? 1.0 : 0.0;
  if (comm->nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, buf.data(), n + 1, MPI_DOUBLE, MPI_SUM, world);
  for (int k = 0; k < n; k++) dE[k] = buf[k];
  return (buf[n] > 0.0) ? 1 : 0;
}

/* ----------------------------------------------------------------------
   energy changes of independent proposals from full energy evaluations,
   each relative to the current state
------------------------------------------------------------------------- */

void FixAtomSwap::full_delta_batch(int n, const SwapProposal *proposals, double *dE)
{
  rebuild_lists_if_needed();
  const double ebefore = energy_full();
  for (int k = 0; k < n; k++) {
    set_proposal_types(proposals[k], 1);
    rebuild_lists_if_needed();
    dE[k] = energy_full() - ebefore;
    set_proposal_types(proposals[k], 0);
  }
  rebuild_lists_if_needed();
}

/* ----------------------------------------------------------------------
   compare local energy changes of a batch of proposals with full energy
   evaluations, each proposal alone and all accepted ones together
------------------------------------------------------------------------- */

void FixAtomSwap::verify_batch(int n, const SwapProposal *proposals, const double *dE,
                               const int *accepted)
{
  std::vector<double> dfull(n);
  full_delta_batch(n, proposals, dfull.data());
  for (int k = 0; k < n; k++) verify_maxdev = MAX(verify_maxdev, fabs(dfull[k] - dE[k]));

  int nacc = 0;
  double joint = 0.0;
  for (int k = 0; k < n; k++) {
    if (!accepted[k]) continue;
    nacc++;
    joint += dE[k];
  }
  if (nacc < 2) return;

  const double ebefore = energy_full();
  for (int k = 0; k < n; k++)
    if (accepted[k]) set_proposal_types(proposals[k], 1);
  rebuild_lists_if_needed();
  verify_maxdev = MAX(verify_maxdev, fabs((energy_full() - ebefore) - joint));
  for (int k = 0; k < n; k++)
    if (accepted[k]) set_proposal_types(proposals[k], 0);
  rebuild_lists_if_needed();
}

/* ----------------------------------------------------------------------
   attempt a semi-grand change of the type of a single atom using local
   energy changes from the pair style.  uses the same random numbers as
   attempt_semi_grand(): the owner of the atom selects the new type and
   decides on acceptance.
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_semi_grand_local()
{
  if (nswap == 0) {
    nswap_null += 1.0;
    return 0;
  }

  int *type = atom->type;
  const int i = pick_semi_grand_atom();

  tagint buf[3] = {0, 0, 0};
  if (i >= 0) {
    const int itype = type[i];
    int jtype = type_list[static_cast<int>(nswaptypes * random_unequal->uniform())];
    while (itype == jtype)
      jtype = type_list[static_cast<int>(nswaptypes * random_unequal->uniform())];
    buf[0] = atom->tag[i];
    buf[1] = itype;
    buf[2] = jtype;
  }
  if (comm->nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, buf, 3, MPI_LMP_TAGINT, MPI_MAX, world);
  const int itype = static_cast<int>(buf[1]);
  const int jtype = static_cast<int>(buf[2]);

  SwapProposal prop;
  prop.count = 1;
  prop.idx[0] = i;
  prop.idx[1] = -1;
  prop.tag[0] = buf[0];
  prop.tag[1] = 0;
  prop.oldtype[0] = itype;
  prop.newtype[0] = jtype;
  prop.oldtype[1] = prop.newtype[1] = 0;
  prop.oldq[0] = prop.newq[0] = (atom->q_flag && (i >= 0)) ? atom->q[i] : 0.0;
  prop.oldq[1] = prop.newq[1] = 0.0;
  prop.qflag = 0;

  double de = 0.0;
  int fail = 1;
  if (local_now) {
    fail = local_delta(1, &prop, &de);
    if (fail) local_eval_failed();
  }

  // full energy evaluation as fallback or for verification

  double energy_after = 0.0;
  int applied = 0;
  if (fail || (energy_mode == ENERGY_VERIFY)) {
    if (!energy_valid) {
      energy_stored = energy_full();
      energy_valid = 1;
    }
    set_proposal_types(prop, 1);
    rebuild_lists_if_needed();
    energy_after = energy_full();
    const double defull = energy_after - energy_stored;
    if (!fail) verify_maxdev = MAX(verify_maxdev, fabs(de - defull));
    de = defull;
    applied = 1;
  }

  int success = 0;
  if (i >= 0)
    if (random_unequal->uniform() < exp(beta * (-de + mu[jtype] - mu[itype]))) success = 1;
  int success_all = success;
  if (comm->nprocs > 1) MPI_Allreduce(&success, &success_all, 1, MPI_INT, MPI_MAX, world);

  if (success_all) {
    const int one = 1;
    swap_apply_accepted(1, &prop, &one);
    swap_forward_comm();
    if (local_now) swap_commit_batch(1, &prop, &one);
    if (applied)
      energy_stored = energy_after;
    else if (energy_valid)
      energy_stored += de;
    return 1;
  }

  if (applied) {
    set_proposal_types(prop, 0);
    rebuild_lists_if_needed();
  }
  return 0;
}

/* ----------------------------------------------------------------------
   attempt a swap of two atoms of different type with the selected
   proposal kernel and energy evaluation.  return 1 if accepted.
------------------------------------------------------------------------- */

int FixAtomSwap::attempt_swap_general()
{
  SwapTrial trial;
  for (int m = 0; m < 2; m++) {
    trial.idx[m] = trial.pos[m] = trial.list[m] = -1;
    trial.tag[m] = 0;
    trial.oldtype[m] = 0;
  }

  // select a proposal, identical on all procs. 0 = null move

  int valid;
  if (proposal_style == PROPOSAL_LEGACY)
    valid = select_legacy(trial);
  else if (rmax > 0.0)
    valid = select_graph(trial);
  else
    valid = select_global(trial);
  if (!valid) {
    nswap_null += 1.0;
    return 0;
  }

  // Hastings factor for hetero edge proposals: M(x) / M(x')

  double hastings = 1.0;
  bigint mnew = gcount;
  if ((proposal_style == PROPOSAL_HETERO) && (rmax > 0.0)) {
    mnew = gcount + delta_hetero(trial);
    if (mnew <= 0)
      error->all(FLERR, "Fix atom/swap internal error: hetero edge count {} after swap", mnew);
    hastings = (double) gcount / (double) mnew;
  }

  // energy change

  SwapProposal prop;
  make_proposal(trial, prop);
  double de = 0.0, energy_after = 0.0;
  int applied = 0;
  int fail = 1;
  if (local_now) {
    fail = local_delta(1, &prop, &de);
    if (fail) local_eval_failed();
  }
  if (fail || (energy_mode == ENERGY_VERIFY)) {
    if (!energy_valid) {
      energy_stored = energy_full();
      energy_valid = 1;
    }
    apply_trial(trial, 1);
    energy_after = energy_full();
    const double defull = energy_after - energy_stored;
    if (local_now && !fail) verify_maxdev = MAX(verify_maxdev, fabs(de - defull));
    de = defull;
    applied = 1;
  }

  // accept or reject

  const double u = random_equal->uniform();
  if (accept_test(u, log(hastings) - beta * de)) {
    const int one = 1;
    swap_apply_accepted(1, &prop, &one);
    if (local_now) {
      swap_forward_comm();
      swap_commit_batch(1, &prop, &one);
    }

    if (applied)
      energy_stored = energy_after;
    else if (energy_valid)
      energy_stored += de;

    // update proposal data structures

    if ((proposal_style == PROPOSAL_HETERO) && (rmax > 0.0)) {
      update_hetero(trial);
      if (gcount != mnew)
        error->all(FLERR, "Fix atom/swap internal error: hetero edge count {} != {}", gcount, mnew);
    } else
      update_lists_after_swap(trial);
    return 1;
  }

  if (applied) apply_trial(trial, 0);
  return 0;
}

/* ----------------------------------------------------------------------
   update the per type atom lists after an accepted swap: each proc moves
   the atoms it owns to the list of their new type, then the global counts
   and offsets are updated
------------------------------------------------------------------------- */

void FixAtomSwap::update_lists_after_swap(const SwapTrial &trial)
{
  const int here0 = (trial.pos[0] >= 0);
  const int here1 = (trial.pos[1] >= 0);

  if (proposal_style == PROPOSAL_LEGACY) {
    if (here0 && here1) {
      local_swap_iatom_list[trial.pos[0]] = trial.idx[1];
      local_swap_jatom_list[trial.pos[1]] = trial.idx[0];
    } else if (here0) {
      local_swap_iatom_list[trial.pos[0]] = local_swap_iatom_list[--niswap_local];
      local_swap_jatom_list[njswap_local++] = trial.idx[0];
    } else if (here1) {
      local_swap_jatom_list[trial.pos[1]] = local_swap_jatom_list[--njswap_local];
      local_swap_iatom_list[niswap_local++] = trial.idx[1];
    }
    update_swap_atoms_counts();
  } else if ((proposal_style == PROPOSAL_HETERO) && (rmax <= 0.0)) {
    const int a = trial.list[0];
    const int b = trial.list[1];
    if (here0 && here1) {
      tlist[a][trial.pos[0]] = trial.idx[1];
      tlist[b][trial.pos[1]] = trial.idx[0];
    } else if (here0) {
      tlist[a][trial.pos[0]] = tlist[a].back();
      tlist[a].pop_back();
      tlist[b].push_back(trial.idx[0]);
    } else if (here1) {
      tlist[b][trial.pos[1]] = tlist[b].back();
      tlist[b].pop_back();
      tlist[a].push_back(trial.idx[1]);
    }
    update_type_counts();
  }

  // proposal geometric: the eligible atoms do not change
}

/* ----------------------------------------------------------------------
   make atom IDs of the selected owned atoms known on all procs
------------------------------------------------------------------------- */

void FixAtomSwap::share_trial(SwapTrial &trial)
{
  tagint buf[2];
  for (int m = 0; m < 2; m++) buf[m] = (trial.idx[m] >= 0) ? atom->tag[trial.idx[m]] : 0;
  if (comm->nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_LMP_TAGINT, MPI_MAX, world);
  for (int m = 0; m < 2; m++) trial.tag[m] = buf[m];
}

/* ----------------------------------------------------------------------
   proposal of the original fix: random atom of type_list[0] and
   random atom of type_list[1]
------------------------------------------------------------------------- */

int FixAtomSwap::select_legacy(SwapTrial &trial)
{
  if ((niswap == 0) || (njswap == 0)) return 0;
  trial.idx[0] = pick_i_swap_atom(&trial.pos[0]);
  trial.idx[1] = pick_j_swap_atom(&trial.pos[1]);
  trial.oldtype[0] = type_list[0];
  trial.oldtype[1] = type_list[1];
  share_trial(trial);
  return 1;
}

/* ----------------------------------------------------------------------
   global proposals without distance limit
   hetero: unordered pair of swap types (a,b) with probability ~ n_a*n_b,
     then a random atom of each type. uniform over all hetero atom pairs,
     their number is invariant under swaps, so the Hastings factor is 1.
   geometric: uniform random pair of distinct eligible atoms, pairs of
     equal type are null moves, then thinning with weight / wmax.
------------------------------------------------------------------------- */

int FixAtomSwap::select_global(SwapTrial &trial)
{
  if (proposal_style == PROPOSAL_HETERO) {
    double wsum = 0.0;
    int npos = 0, alast = -1, blast = -1;
    for (int a = 0; a < nswaptypes; a++)
      for (int b = a + 1; b < nswaptypes; b++) {
        double w = (double) tcount[a] * (double) tcount[b];
        if (w > 0.0) {
          wsum += w;
          npos++;
          alast = a;
          blast = b;
        }
      }
    if (npos == 0) return 0;

    // only draw a type pair when there is a choice

    int a = alast, b = blast;
    if (npos > 1) {
      double r = wsum * random_equal->uniform();
      double cum = 0.0;
      int found = 0;
      for (int ia = 0; (ia < nswaptypes) && !found; ia++)
        for (int ib = ia + 1; ib < nswaptypes; ib++) {
          double w = (double) tcount[ia] * (double) tcount[ib];
          if (w <= 0.0) continue;
          cum += w;
          if (r < cum) {
            a = ia;
            b = ib;
            found = 1;
            break;
          }
        }
    }
    trial.idx[0] = pick_from_list(a, &trial.pos[0]);
    trial.idx[1] = pick_from_list(b, &trial.pos[1]);
    trial.list[0] = a;
    trial.list[1] = b;
    trial.oldtype[0] = type_list[a];
    trial.oldtype[1] = type_list[b];
    share_trial(trial);
    return 1;
  }

  // proposal geometric

  if (ecount < 2) return 0;
  auto iglobal = static_cast<bigint>((double) ecount * random_equal->uniform());
  auto jglobal = static_cast<bigint>((double) (ecount - 1) * random_equal->uniform());
  if (iglobal >= ecount) iglobal = ecount - 1;
  if (jglobal >= ecount - 1) jglobal = ecount - 2;
  if (jglobal >= iglobal) jglobal++;
  const bigint iwhich[2] = {iglobal, jglobal};
  const auto nelist = (bigint) elist.size();
  tagint buf[4] = {0, 0, 0, 0};
  for (int m = 0; m < 2; m++) {
    int k = -1;
    if ((iwhich[m] >= ebefore) && (iwhich[m] < ebefore + nelist)) k = elist[iwhich[m] - ebefore];
    trial.idx[m] = k;
    if (k >= 0) {
      buf[m] = atom->tag[k];
      buf[2 + m] = atom->type[k];
    }
  }
  if (comm->nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, buf, 4, MPI_LMP_TAGINT, MPI_MAX, world);
  for (int m = 0; m < 2; m++) {
    trial.tag[m] = buf[m];
    trial.oldtype[m] = static_cast<int>(buf[2 + m]);
  }
  if (trial.oldtype[0] == trial.oldtype[1]) return 0;
  double w = weight[trial.oldtype[0]][trial.oldtype[1]];
  if (w < wmax)
    if (random_equal->uniform() * wmax >= w) return 0;
  return 1;
}

/* ----------------------------------------------------------------------
   proposals on the fixed geometric graph of eligible pairs within rmax
   hetero: uniform random edge between atoms of different type
   geometric: uniform random edge, equal types are null moves,
     then thinning with weight / wmax
------------------------------------------------------------------------- */

int FixAtomSwap::select_graph(SwapTrial &trial)
{
  if (gcount <= 0) return 0;
  auto r = static_cast<bigint>((double) gcount * random_equal->uniform());
  if (r >= gcount) r = gcount - 1;

  bigint offset;
  int owner = find_owner(r, offset);
  tagint buf[4] = {0, 0, 0, 0};
  if (comm->me == owner) {
    auto n = static_cast<int>(r - offset);
    int e = (proposal_style == PROPOSAL_HETERO) ? het_list[n] : n;
    int i = edge_i[e];
    int j = edge_j[e];
    buf[0] = atom->tag[i];
    buf[1] = atom->tag[j];
    buf[2] = atom->type[i];
    buf[3] = atom->type[j];
  }
  if (comm->nprocs > 1) MPI_Bcast(buf, 4, MPI_LMP_TAGINT, owner, world);

  for (int m = 0; m < 2; m++) {
    trial.tag[m] = buf[m];
    trial.oldtype[m] = static_cast<int>(buf[2 + m]);
    int k = atom->map(trial.tag[m]);
    trial.idx[m] = ((k >= 0) && (k < atom->nlocal)) ? k : -1;
  }

  if (trial.oldtype[0] == trial.oldtype[1]) {
    if (proposal_style == PROPOSAL_HETERO)
      error->all(FLERR, "Fix atom/swap internal error: selected hetero edge has equal types");
    return 0;
  }
  if (proposal_style == PROPOSAL_GEOMETRIC) {
    double w = weight[trial.oldtype[0]][trial.oldtype[1]];
    if (w < wmax)
      if (random_equal->uniform() * wmax >= w) return 0;
  }
  return 1;
}

/* ----------------------------------------------------------------------
   pick a random atom of swap type index t, return local index or -1
------------------------------------------------------------------------- */

int FixAtomSwap::pick_from_list(int t, int *pos)
{
  int i = -1;
  *pos = -1;
  auto iwhichglobal = static_cast<bigint>((double) tcount[t] * random_equal->uniform());
  const auto nlocal = (bigint) tlist[t].size();
  if ((iwhichglobal >= tbefore[t]) && (iwhichglobal < tbefore[t] + nlocal)) {
    *pos = static_cast<int>(iwhichglobal - tbefore[t]);
    i = tlist[t][*pos];
  }
  return i;
}

/* ----------------------------------------------------------------------
   find proc owning global selection index r, also return its offset
------------------------------------------------------------------------- */

int FixAtomSwap::find_owner(bigint r, bigint &offset)
{
  offset = 0;
  for (int iproc = 0; iproc < comm->nprocs; iproc++) {
    if (r < offset + proc_count[iproc]) return iproc;
    offset += proc_count[iproc];
  }
  error->all(FLERR, "Fix atom/swap internal error: invalid selection index");
  return -1;
}

/* ----------------------------------------------------------------------
   update per swap type lists of eligible atoms for global proposals
------------------------------------------------------------------------- */

void FixAtomSwap::update_type_lists()
{
  int nlocal = atom->nlocal;
  int *type = atom->type;

  tlist.resize(nswaptypes);
  for (auto &l : tlist) l.clear();
  elist.clear();
  for (int i = 0; i < nlocal; i++) {
    if (!is_eligible(i)) continue;
    tlist[type2swap[type[i]]].push_back(i);
    elist.push_back(i);
  }
  update_type_counts();
}

/* ----------------------------------------------------------------------
   update global counts and offsets of the per swap type lists
------------------------------------------------------------------------- */

void FixAtomSwap::update_type_counts()
{
  std::vector<bigint> nlocal_type(nswaptypes + 1);
  for (int t = 0; t < nswaptypes; t++) nlocal_type[t] = tlist[t].size();
  nlocal_type[nswaptypes] = elist.size();
  std::vector<bigint> nall(nswaptypes + 1), nbefore(nswaptypes + 1);
  MPI_Allreduce(nlocal_type.data(), nall.data(), nswaptypes + 1, MPI_LMP_BIGINT, MPI_SUM, world);
  MPI_Scan(nlocal_type.data(), nbefore.data(), nswaptypes + 1, MPI_LMP_BIGINT, MPI_SUM, world);

  tcount.resize(nswaptypes);
  tbefore.resize(nswaptypes);
  for (int t = 0; t < nswaptypes; t++) {
    tcount[t] = nall[t];
    tbefore[t] = nbefore[t] - nlocal_type[t];
  }
  ecount = nall[nswaptypes];
  ebefore = nbefore[nswaptypes] - nlocal_type[nswaptypes];
}

/* ----------------------------------------------------------------------
   build the fixed geometric graph of eligible atom pairs with 0 < r <= rmax.
   the graph does not depend on atom types, since swaps only exchange
   types among the swap types. each edge is stored once, on the proc
   owning the atom with the smaller atom ID. multiple periodic images of
   a pair within rmax are merged into a single edge.
------------------------------------------------------------------------- */

void FixAtomSwap::build_graph()
{
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int *type = atom->type;
  tagint *tag = atom->tag;
  double **x = atom->x;

  // eligibility of owned atoms, communicated to ghost atoms

  eligible.assign(nall, 0);
  for (int i = 0; i < nlocal; i++) eligible[i] = is_eligible(i);
  comm_mode = COMM_FLAG;
  swap_forward_comm();
  comm_mode = COMM_TYPEQ;

  // edges from the occasional full neighbor list

  edge_i.clear();
  edge_j.clear();
  edge_jimg.clear();
  const double rmaxsq = rmax * rmax;
  std::vector<std::pair<tagint, std::pair<int, int>>> cand;
  int inum = swap_list->inum;
  int *ilist = swap_list->ilist;
  int *numneigh = swap_list->numneigh;
  int **firstneigh = swap_list->firstneigh;

  // visit owned atoms in tag order and sort each row by the larger tag,
  // so a random edge index is independent of the neighbor list order

  std::vector<int> centers;
  if (inum) centers.assign(ilist, ilist + inum);
  std::sort(centers.begin(), centers.end(), [tag](int i, int j) {
    return tag[i] < tag[j];
  });
  for (int i : centers) {
    if (!eligible[i]) continue;
    cand.clear();
    int *jlist = firstneigh[i];
    int jnum = numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      if (!eligible[j] || (tag[j] <= tag[i])) continue;
      double delx = x[i][0] - x[j][0];
      double dely = x[i][1] - x[j][1];
      double delz = x[i][2] - x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if ((rsq > rmaxsq) || (rsq <= 0.0)) continue;
      int jlocal = atom->map(tag[j]);
      if ((jlocal < 0) || (jlocal >= nlocal)) jlocal = j;
      cand.emplace_back(tag[j], std::make_pair(jlocal, j));
    }
    std::sort(cand.begin(), cand.end());
    for (std::size_t n = 0; n < cand.size(); n++) {
      if ((n > 0) && (cand[n].first == cand[n - 1].first)) continue;
      edge_i.push_back(i);
      edge_j.push_back(cand[n].second.first);
      edge_jimg.push_back(cand[n].second.second);
    }
  }
  int nedge = edge_i.size();

  // edges incident to each local index

  adj_first.assign(nall + 1, 0);
  for (int e = 0; e < nedge; e++) {
    adj_first[edge_i[e] + 1]++;
    adj_first[edge_j[e] + 1]++;
  }
  for (int k = 0; k < nall; k++) adj_first[k + 1] += adj_first[k];
  adj_edge.resize(2 * nedge);
  std::vector<int> fill(adj_first.begin(), adj_first.end() - 1);
  for (int e = 0; e < nedge; e++) {
    adj_edge[fill[edge_i[e]]++] = e;
    adj_edge[fill[edge_j[e]]++] = e;
  }

  // set of hetero edges with reverse index

  het_pos.assign(nedge, -1);
  het_list.clear();
  if (proposal_style == PROPOSAL_HETERO) {
    for (int e = 0; e < nedge; e++) {
      if (type[edge_i[e]] != type[edge_j[e]]) {
        het_pos[e] = het_list.size();
        het_list.push_back(e);
      }
    }
  }

  update_graph_counts();
}

/* ----------------------------------------------------------------------
   gather # of selectable edges of all procs
------------------------------------------------------------------------- */

void FixAtomSwap::update_graph_counts()
{
  bigint nmine = (proposal_style == PROPOSAL_HETERO) ? (bigint) het_list.size()
                                                     : (bigint) edge_i.size();
  proc_count.resize(comm->nprocs);
  MPI_Allgather(&nmine, 1, MPI_LMP_BIGINT, proc_count.data(), 1, MPI_LMP_BIGINT, world);
  gcount = 0;
  for (const auto &n : proc_count) gcount += n;
}

/* ----------------------------------------------------------------------
   change of the global number of hetero edges for a swap of the two atoms
   only edges incident to the two atoms can change. the edge between them
   stays hetero. must be called before the types are changed.
------------------------------------------------------------------------- */

bigint FixAtomSwap::delta_hetero(const SwapTrial &trial)
{
  int *type = atom->type;
  tagint *tag = atom->tag;
  bigint dm = 0;

  for (int m = 0; m < 2; m++) {
    const tagint other = trial.tag[1 - m];
    const int told = trial.oldtype[m];
    const int tnew = trial.oldtype[1 - m];
    for (int k = atom->map(trial.tag[m]); k >= 0; k = atom->sametag[k]) {
      for (int n = adj_first[k]; n < adj_first[k + 1]; n++) {
        int e = adj_edge[n];
        int w = (edge_i[e] == k) ? edge_j[e] : edge_i[e];
        if (tag[w] == other) continue;
        int hold = (told != type[w]) ? 1 : 0;
        int hnew = (tnew != type[w]) ? 1 : 0;
        dm += hnew - hold;
      }
    }
  }

  bigint dmall = dm;
  if (comm->nprocs > 1) MPI_Allreduce(&dm, &dmall, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  return dmall;
}

/* ----------------------------------------------------------------------
   update hetero edge set after an accepted swap (types already updated
   for owned and ghost atoms)
------------------------------------------------------------------------- */

void FixAtomSwap::update_hetero(const SwapTrial &trial)
{
  int *type = atom->type;
  for (int m = 0; m < 2; m++) {
    for (int k = atom->map(trial.tag[m]); k >= 0; k = atom->sametag[k]) {
      for (int n = adj_first[k]; n < adj_first[k + 1]; n++) {
        int e = adj_edge[n];
        int het = (type[edge_i[e]] != type[edge_j[e]]) ? 1 : 0;
        if (het && (het_pos[e] < 0)) {
          het_pos[e] = het_list.size();
          het_list.push_back(e);
        } else if (!het && (het_pos[e] >= 0)) {
          int last = het_list.back();
          het_list[het_pos[e]] = last;
          het_pos[last] = het_pos[e];
          het_list.pop_back();
          het_pos[e] = -1;
        }
      }
    }
  }
  update_graph_counts();
}

/* ----------------------------------------------------------------------
   fill proposal for the local energy interface of the pair style
------------------------------------------------------------------------- */

void FixAtomSwap::make_proposal(const SwapTrial &trial, SwapProposal &prop)
{
  prop.count = 2;
  prop.qflag = 0;
  for (int m = 0; m < 2; m++) {
    const int k = trial.idx[m];
    prop.idx[m] = k;
    prop.tag[m] = trial.tag[m];
    prop.oldtype[m] = trial.oldtype[m];
    prop.newtype[m] = trial.oldtype[1 - m];
    if (atom->q_flag && qtype) {
      prop.oldq[m] = qtype[type2swap[trial.oldtype[m]]];
      prop.newq[m] = qtype[type2swap[trial.oldtype[1 - m]]];
      if (prop.oldq[m] != prop.newq[m]) prop.qflag = 1;
    } else {
      prop.oldq[m] = prop.newq[m] = (atom->q_flag && (k >= 0)) ? atom->q[k] : 0.0;
    }
  }
}

/* ----------------------------------------------------------------------
   apply (flag = 1) or revert (flag = 0) a swap trial for a full energy
   evaluation on all owned and ghost images.  rebuild neighbor lists if
   the swap types have different cutoffs.
------------------------------------------------------------------------- */

void FixAtomSwap::apply_trial(const SwapTrial &trial, int flag)
{
  SwapProposal prop;
  make_proposal(trial, prop);
  set_proposal_types(prop, flag);
  if (atom->rmass && mtype) {
    for (int m = 0; m < 2; m++) {
      const int k = trial.idx[m];
      if (k < 0) continue;
      const int t = flag ? trial.oldtype[1 - m] : trial.oldtype[m];
      atom->rmass[k] = mtype[type2swap[t]];
    }
  }
  rebuild_lists_if_needed();
}

/* ----------------------------------------------------------------------
   rebuild neighbor lists when the types of swapped atoms have different
   cutoffs or when neighbor list exclusions by type are used.  ghost atoms
   are unchanged, so no borders() call is needed, except with comm_modify
   mode multi, where the ghost atom cutoff depends on the atom type.
   comm_modify mode multi requires neighbor style multi, so rmax = 0.0 and
   only owned atoms are stored in the lists of swap candidates.
------------------------------------------------------------------------- */

void FixAtomSwap::rebuild_lists_if_needed()
{
  if (!unequal_cutoffs && (neighbor->nex_type <= 0)) return;
  if (unequal_cutoffs && (comm->mode == Comm::MULTI)) {
    if (domain->triclinic) domain->x2lamda(atom->nlocal);
    comm->exchange();
    comm->borders();
    if (domain->triclinic) domain->lamda2x(atom->nlocal + atom->nghost);
  }
  if (modify->has_fixes(FixConst::PRE_NEIGHBOR)) modify->pre_neighbor();
  neighbor->build(1);
}

/* ----------------------------------------------------------------------
   counter based uniform random number in [0,1) for checkerboard sweeps.
   depends only on the seed, the timestep and the arguments, so it is the
   same on all procs and independent of the domain decomposition.
------------------------------------------------------------------------- */

double FixAtomSwap::counter_uniform(int stream, int color, int round, bigint cell, int slot) const
{
  uint64_t h = mix64((uint64_t) seed);
  h = mix64(h ^ (uint64_t) update->ntimestep);
  h = mix64(h ^ (uint64_t) stream);
  h = mix64(h ^ (uint64_t) color);
  h = mix64(h ^ (uint64_t) round);
  h = mix64(h ^ (uint64_t) cell);
  h = mix64(h ^ (uint64_t) slot);
  return (double) (h >> 11) * (1.0 / 9007199254740992.0);
}

/* ----------------------------------------------------------------------
   assign the edges of the geometric graph to cells of the shifted
   checkerboard grid.  an edge belongs to a cell if both end points (the
   owned atom i and the image of j within rmax) are inside the cell.  the
   decision is made by the proc storing the edge, so it is unique.
------------------------------------------------------------------------- */

void FixAtomSwap::setup_checkerboard_cells(const double *shift)
{
  double **x = atom->x;
  const int nedge = edge_i.size();
  edge_cell.assign(nedge, -1);
  std::vector<std::pair<bigint, int>> sorted[8];

  for (int e = 0; e < nedge; e++) {
    double lami[3], lamj[3];
    domain->x2lamda(x[edge_i[e]], lami);
    domain->x2lamda(x[edge_jimg[e]], lamj);
    int ci[3];
    int inside = 1;
    for (int d = 0; d < 3; d++) {
      double s = lami[d] - shift[d];
      s -= std::floor(s);
      if (s >= 1.0) s = 0.0;
      ci[d] = MIN(cb_dims[d] - 1, static_cast<int>(s * cb_dims[d]));
      const double sj = s + (lamj[d] - lami[d]);
      if (static_cast<int>(std::floor(sj * cb_dims[d])) != ci[d]) {
        inside = 0;
        break;
      }
    }
    if (!inside) continue;
    const bigint cell = ((bigint) ci[0] * cb_dims[1] + ci[1]) * cb_dims[2] + ci[2];
    const int color = (ci[0] & 1) | ((ci[1] & 1) << 1) | ((ci[2] & 1) << 2);
    edge_cell[e] = cell;
    sorted[color].emplace_back(cell, e);
  }

  for (int c = 0; c < 8; c++) {
    ColorCells &cc = color_cells[c];
    std::sort(sorted[c].begin(), sorted[c].end());
    cc.cell.clear();
    cc.first.clear();
    cc.edges.clear();
    for (const auto &ce : sorted[c]) {
      if (cc.cell.empty() || (cc.cell.back() != ce.first)) {
        cc.cell.push_back(ce.first);
        cc.first.push_back(cc.edges.size());
      }
      cc.edges.push_back(ce.second);
    }
    cc.first.push_back(cc.edges.size());
    cc.mlocal.assign(cc.cell.size(), 0);
    cc.wlocal.assign(cc.cell.size(), 0.0);
  }
}

/* ----------------------------------------------------------------------
   parallel sweep with an 8-color checkerboard of cells wider than
   max(rmax, conflict radius).  in each round of a color, every cell of
   that color proposes at most one swap of an edge inside the cell.
   proposals of different cells are independent, so they are evaluated
   and accepted together.  the selection of the edge of a cell, which may
   have edges on several procs, uses the cell totals of all procs and
   counter based random numbers, so all procs make the same decisions.
   return # of accepted swaps, nattempt is set to the # of attempts.
------------------------------------------------------------------------- */

bigint FixAtomSwap::checkerboard_sweep(bigint &nattempt)
{
  const int me = comm->me;
  const int nprocs = comm->nprocs;
  int *type = atom->type;
  tagint *tag = atom->tag;
  const int hetero = (proposal_style == PROPOSAL_HETERO);

  build_graph();

  double shift[3];
  for (int d = 0; d < 3; d++) shift[d] = counter_uniform(STREAM_SHIFT, 0, 0, 0, d);
  setup_checkerboard_cells(shift);

  if ((me == 0) &&
      ((cb_dims[0] != cb_msg_dims[0]) || (cb_dims[1] != cb_msg_dims[1]) ||
       (cb_dims[2] != cb_msg_dims[2])))
    utils::logmesg(lmp, "Fix atom/swap: using checkerboard sweeps with {}x{}x{} cells\n",
                   cb_dims[0], cb_dims[1], cb_dims[2]);
  for (int d = 0; d < 3; d++) cb_msg_dims[d] = cb_dims[d];

  const bigint ncells = (bigint) cb_dims[0] * cb_dims[1] * cb_dims[2];
  const bigint cells_per_color = ncells / 8;
  nattempt = ncells * ncycles;
  const int reverse = (counter_uniform(STREAM_ORDER, 0, 0, 0, 0) < 0.5) ? 1 : 0;

  bigint nsuccess = 0;
  std::vector<int> counts(nprocs), displs(nprocs), counts2(nprocs), displs2(nprocs);
  std::vector<bigint> sendcm, recvcm;
  std::vector<double> sendw, recvw;
  std::vector<int> recrank;

  // global cell table of the current color: active cells with totals and
  // the records of all procs for the cell in ascending proc order

  std::vector<bigint> gcell, gm;
  std::vector<double> gw;
  std::vector<int> gfirst, grec;

  std::vector<bigint> sendp, recvp;
  std::vector<SwapProposal> props;
  std::vector<bigint> pcell;
  std::vector<int> pslot;
  std::vector<double> dbuf, ldm, ldw;
  std::vector<int> accepted;

  // valid send buffers for procs without records

  bigint dummy_bi = 0;
  double dummy_d = 0.0;

  for (int pass = 0; pass < 8; pass++) {
    const int color = reverse ? 7 - pass : pass;
    ColorCells &cc = color_cells[color];
    const int nlcell = cc.cell.size();

    // local # and weight of selectable edges per cell

    for (int n = 0; n < nlcell; n++) {
      bigint m = 0;
      double w = 0.0;
      for (int k = cc.first[n]; k < cc.first[n + 1]; k++) {
        const int e = cc.edges[k];
        const double we = selection_weight(type[edge_i[e]], type[edge_jimg[e]]);
        if (we > 0.0) {
          m++;
          w += we;
        }
      }
      cc.mlocal[n] = m;
      cc.wlocal[n] = w;
    }

    int need_gather = 1;
    for (int round = 0; round < ncycles; round++) {

      // gather cells with selectable edges from all procs and merge them
      // by cell ID. geometric selection does not depend on types, so its
      // table is gathered only once per color.

      if (need_gather) {
        sendcm.clear();
        sendw.clear();
        for (int n = 0; n < nlcell; n++) {
          if (cc.mlocal[n] <= 0) continue;
          sendcm.push_back(cc.cell[n]);
          sendcm.push_back(cc.mlocal[n]);
          sendw.push_back(cc.wlocal[n]);
        }
        int nsend = sendw.size();
        MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
        int nrec = 0;
        for (int p = 0; p < nprocs; p++) {
          displs[p] = nrec;
          counts2[p] = 2 * counts[p];
          displs2[p] = 2 * nrec;
          nrec += counts[p];
        }
        recvcm.resize(2 * (std::size_t) nrec + 1);
        recvw.resize((std::size_t) nrec + 1);
        const bigint *scm = nsend ? sendcm.data() : &dummy_bi;
        const double *sw = nsend ? sendw.data() : &dummy_d;
        MPI_Allgatherv(scm, 2 * nsend, MPI_LMP_BIGINT, recvcm.data(), counts2.data(),
                       displs2.data(), MPI_LMP_BIGINT, world);
        MPI_Allgatherv(sw, nsend, MPI_DOUBLE, recvw.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, world);
        recrank.resize(nrec);
        for (int p = 0; p < nprocs; p++)
          for (int r = displs[p]; r < displs[p] + counts[p]; r++) recrank[r] = p;

        // records are in proc order, a stable sort by cell keeps that order

        grec.resize(nrec);
        for (int r = 0; r < nrec; r++) grec[r] = r;
        std::stable_sort(grec.begin(), grec.end(), [&recvcm](int a, int b) {
          return recvcm[2 * a] < recvcm[2 * b];
        });
        gcell.clear();
        gm.clear();
        gw.clear();
        gfirst.clear();
        for (int k = 0; k < nrec; k++) {
          const int r = grec[k];
          if (gcell.empty() || (gcell.back() != recvcm[2 * r])) {
            gcell.push_back(recvcm[2 * r]);
            gm.push_back(0);
            gw.push_back(0.0);
            gfirst.push_back(k);
          }
          gm.back() += recvcm[2 * r + 1];
          gw.back() += recvw[r];
        }
        gfirst.push_back(nrec);
        need_gather = 0;
      }
      const int nactive = gcell.size();

      // select at most one edge per active cell

      bigint nnull = cells_per_color - nactive;
      sendp.clear();
      for (int g = 0; g < nactive; g++) {
        const bigint cell = gcell[g];
        const double u = counter_uniform(STREAM_EDGE, color, round, cell, 0);
        const double target = u * lazy_factor(gm[g]) * gw[g];
        if (target >= gw[g]) {
          nnull++;
          continue;
        }

        // proc owning the selected edge and the residual target

        double cum = 0.0;
        int owner = recrank[grec[gfirst[g + 1] - 1]];
        double rem = target;
        for (int k = gfirst[g]; k < gfirst[g + 1]; k++) {
          const int r = grec[k];
          if (target < cum + recvw[r]) {
            owner = recrank[r];
            rem = target - cum;
            break;
          }
          cum += recvw[r];
          rem = target - cum;
        }
        if (owner != me) continue;

        // selected local edge of the cell

        const int n = std::lower_bound(cc.cell.begin(), cc.cell.end(), cell) - cc.cell.begin();
        int chosen = -1, last = -1;
        double lcum = 0.0;
        for (int k = cc.first[n]; k < cc.first[n + 1]; k++) {
          const int e = cc.edges[k];
          const double we = selection_weight(type[edge_i[e]], type[edge_jimg[e]]);
          if (we <= 0.0) continue;
          last = e;
          if (rem < lcum + we) {
            chosen = e;
            break;
          }
          lcum += we;
        }
        if (chosen < 0) chosen = last;
        const int i = edge_i[chosen];
        const int j = edge_jimg[chosen];
        const int ti = type[i];
        const int tj = type[j];

        // equal types (geometric only) and thinning are null moves

        bigint flag = 1;
        if (ti == tj)
          flag = 0;
        else if (!hetero) {
          const double w = weight[ti][tj];
          if ((w < wmax) && (counter_uniform(STREAM_THIN, color, round, cell, 2) * wmax >= w))
            flag = 0;
        }
        sendp.push_back(cell);
        sendp.push_back(flag);
        sendp.push_back(tag[i]);
        sendp.push_back(tag[j]);
        sendp.push_back(ti);
        sendp.push_back(tj);
      }

      // gather the proposals of all procs, all procs must participate

      int nsend = sendp.size();
      MPI_Allgather(&nsend, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
      int ntotal = 0;
      for (int p = 0; p < nprocs; p++) {
        displs[p] = ntotal;
        ntotal += counts[p];
      }
      recvp.resize((std::size_t) ntotal + 1);
      const bigint *sp = nsend ? sendp.data() : &dummy_bi;
      MPI_Allgatherv(sp, nsend, MPI_LMP_BIGINT, recvp.data(), counts.data(), displs.data(),
                     MPI_LMP_BIGINT, world);

      props.clear();
      pcell.clear();
      pslot.clear();
      for (int r = 0; r < ntotal / 6; r++) {
        const bigint *rec = &recvp[6 * (std::size_t) r];
        if (!rec[1]) {
          nnull++;
          continue;
        }
        SwapTrial trial;
        for (int m = 0; m < 2; m++) {
          trial.tag[m] = (tagint) rec[2 + m];
          trial.oldtype[m] = (int) rec[4 + m];
          const int k = atom->map(trial.tag[m]);
          trial.idx[m] = ((k >= 0) && (k < atom->nlocal)) ? k : -1;
          trial.pos[m] = trial.list[m] = -1;
        }
        SwapProposal prop;
        make_proposal(trial, prop);
        props.push_back(prop);
        pcell.push_back(rec[0]);
        pslot.push_back(std::lower_bound(gcell.begin(), gcell.end(), rec[0]) - gcell.begin());
      }
      nswap_null += (double) nnull;
      const int nprop = props.size();
      if (nprop == 0) continue;

      // owner computes energy changes and, for hetero selection, the change
      // of the cell totals from the edges stored on this proc

      dbuf.assign(3 * nprop + 1, 0.0);
      ldm.assign(nprop, 0.0);
      ldw.assign(nprop, 0.0);
      const int fail = local_now ? swap_eval_batch(nprop, props.data(), dbuf.data()) : 0;
      if (hetero) {
        for (int k = 0; k < nprop; k++) {
          const SwapProposal &p = props[k];
          for (int m = 0; m < 2; m++) {
            const tagint other = p.tag[1 - m];
            const int told = p.oldtype[m];
            const int tnew = p.newtype[m];
            for (int a = atom->map(p.tag[m]); a >= 0; a = atom->sametag[a]) {
              for (int n = adj_first[a]; n < adj_first[a + 1]; n++) {
                const int e = adj_edge[n];
                if (edge_cell[e] != pcell[k]) continue;
                const int w = (edge_i[e] == a) ? edge_j[e] : edge_i[e];
                if (tag[w] == other) continue;
                const double wold = selection_weight(told, type[w]);
                const double wnew = selection_weight(tnew, type[w]);
                ldm[k] += ((wnew > 0.0) ? 1.0 : 0.0) - ((wold > 0.0) ? 1.0 : 0.0);
                ldw[k] += wnew - wold;
              }
            }
          }
          dbuf[nprop + k] = ldm[k];
          dbuf[2 * nprop + k] = ldw[k];
        }
      }
      dbuf[3 * nprop] = fail ? 1.0 : 0.0;
      if (nprocs > 1)
        MPI_Allreduce(MPI_IN_PLACE, dbuf.data(), 3 * nprop + 1, MPI_DOUBLE, MPI_SUM, world);
      const int failall = (dbuf[3 * nprop] > 0.0) ? 1 : 0;

      // the swaps of different cells are independent, so after a failed
      // local evaluation full energy changes can be used instead

      if (failall) local_eval_failed();
      if (!local_now) full_delta_batch(nprop, props.data(), dbuf.data());

      // identical Metropolis-Hastings decisions on all procs

      accepted.assign(nprop, 0);
      int nacc = 0;
      for (int k = 0; k < nprop; k++) {
        const int g = pslot[k];
        double hastings = 1.0;
        if (hetero) {
          const double mbefore = (double) gm[g];
          const double wbefore = gw[g];
          const double mafter = mbefore + dbuf[nprop + k];
          const double wafter = wbefore + dbuf[2 * nprop + k];
          if ((mafter < 1.0) || (wafter <= 0.0))
            error->all(FLERR, Error::NOLASTLINE,
                       "Fix atom/swap internal error: no hetero edge left in a cell after swap");
          hastings =
              (lazy_factor(gm[g]) * wbefore) / (lazy_factor((bigint) (mafter + 0.5)) * wafter);
        }
        const double u = counter_uniform(STREAM_ACCEPT, color, round, pcell[k], 1);
        if (accept_test(u, log(hastings) - beta * dbuf[k])) {
          accepted[k] = 1;
          nacc++;
        }
      }

      if ((energy_mode == ENERGY_VERIFY) && local_now)
        verify_batch(nprop, props.data(), dbuf.data(), accepted.data());

      if (nacc == 0) continue;

      // apply on all procs, refresh ghost atoms, update pair style data

      swap_apply_accepted(nprop, props.data(), accepted.data());
      swap_forward_comm();
      if (local_now) swap_commit_batch(nprop, props.data(), accepted.data());
      nsuccess += nacc;

      // hetero selection depends on the types: gather the cell totals again

      need_gather = hetero;

      // update the local and global cell totals

      if (hetero) {
        for (int k = 0; k < nprop; k++) {
          if (!accepted[k]) continue;
          const int g = pslot[k];
          gm[g] += (bigint) std::lround(dbuf[nprop + k]);
          gw[g] += dbuf[2 * nprop + k];
          auto it = std::lower_bound(cc.cell.begin(), cc.cell.end(), pcell[k]);
          if ((it != cc.cell.end()) && (*it == pcell[k])) {
            const int n = it - cc.cell.begin();
            cc.mlocal[n] += (bigint) std::lround(ldm[k]);
            cc.wlocal[n] += ldw[k];
          }
        }
      }
    }
  }

  energy_valid = 0;
  return nsuccess;
}
