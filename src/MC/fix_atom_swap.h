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
FixStyle(atom/swap,FixAtomSwap);
// clang-format on
#else

#ifndef LMP_FIX_ATOM_SWAP_H
#define LMP_FIX_ATOM_SWAP_H

#include "fix.h"
#include "pair_swap_local.h"

#include <unordered_map>
#include <utility>

namespace LAMMPS_NS {

class FixAtomSwap : public Fix {
 public:
  FixAtomSwap(class LAMMPS *, int, char **);
  ~FixAtomSwap() override;
  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void pre_exchange() override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  double compute_vector(int) override;
  int modify_param(int, char **) override;
  double memory_usage() override;
  void write_restart(FILE *) override;
  void restart(char *) override;
  void *extract(const char *, int &) override;

  int image(int *&, double **&) override;

 protected:
  int nevery, seed;
  int ke_flag;            // yes = conserve ke, no = do not conserve ke
  int semi_grand_flag;    // yes = semi-grand canonical, no = constant composition
  int ncycles;
  int niswap, njswap;                  // # of i,j swap atoms on all procs
  int niswap_local, njswap_local;      // # of swap atoms on this proc
  int niswap_before, njswap_before;    // # of swap atoms on procs < this proc
  int nswap;                           // # of swap atoms on all procs
  int nswap_local;                     // # of swap atoms on this proc
  int nswap_before;                    // # of swap atoms on procs < this proc
  class Region *region;                // swap region
  char *idregion;                      // swap region id

  int mc_active;    // 1 during MC trials, otherwise 0

  int nswaptypes, nmutypes;
  int *type_list;
  double *mu;

  double nswap_attempts;
  double nswap_successes;

  bool unequal_cutoffs;

  int atom_swap_nmax;
  double beta;
  double *qtype, *mtype;
  double energy_stored;
  double **sqrt_mass_ratio;
  int *local_swap_iatom_list;
  int *local_swap_jatom_list;
  int *local_swap_atom_list;

  class RanPark *random_equal;
  class RanPark *random_unequal;

  class Compute *c_pe;

  // arrays for dump image rendering

  int *imgobjs;
  double **imgparms;
  // maps atom IDs to number of steps they have been highlighted
  std::unordered_map<tagint, std::pair<int, int>> vizatoms;
  int vizsteps;    // number of steps to highlight atoms in reactions

  void options(int, char **);
  int attempt_semi_grand();
  int attempt_swap();
  double energy_full();
  int pick_semi_grand_atom();
  int pick_i_swap_atom(int *pos = nullptr);
  int pick_j_swap_atom(int *pos = nullptr);
  void update_semi_grand_atoms_list();
  void update_swap_atoms_list();
  void update_swap_atoms_counts();

  // ------------------------------------------------------------------
  // optional extensions: proposal kernels, local energy changes and
  // parallel checkerboard sweeps

  enum { PROPOSAL_LEGACY, PROPOSAL_HETERO, PROPOSAL_GEOMETRIC };
  enum { ENERGY_UNSET = -1, ENERGY_FULL, ENERGY_AUTO, ENERGY_LOCAL, ENERGY_VERIFY };
  enum { PARALLEL_SERIAL, PARALLEL_CHECKERBOARD };
  enum { COMM_TYPEQ, COMM_FLAG };

  int new_syntax;        // 1 if any of the extension keywords was used
  int proposal_style;    // PROPOSAL_LEGACY, PROPOSAL_HETERO, PROPOSAL_GEOMETRIC
  int proposal_flag;     // 1 if keyword proposal was used
  int energy_mode;       // ENERGY_FULL, ENERGY_AUTO, ENERGY_LOCAL, ENERGY_VERIFY
  int parallel_mode;     // PARALLEL_SERIAL or PARALLEL_CHECKERBOARD
  double rmax;           // max distance of swap partners, 0.0 = no limit

  std::vector<int> weight_args_type;    // user weights: type pairs and values
  std::vector<double> weight_args_value;
  double **weight;    // symmetric proposal weight for each type pair
  double wmax;        // max weight of pairs of different swap types
  int *type2swap;     // swap type index for each atom type, -1 if none

  int use_local;                      // 1 if local energy changes pass the static checks
  int local_now;                      // 1 if local energy changes are used in this invocation
  int local_msg;                      // last reported energy evaluation decision
  int dynamic_msg;                    // 1 if a fallback to full energies was reported
  int eval_msg;                       // 1 if a failed local energy evaluation was reported
  int checker_msg;                    // 1 if a fallback to serial sweeps was reported
  int cb_msg_dims[3];                 // last reported checkerboard grid
  class PairSwapLocal *pair_local;    // local energy interface of the pair style
  SwapLocalCaps caps;                 // its capabilities
  class NeighList *swap_list;         // occasional full neighbor list
  int comm_mode;                      // COMM_TYPEQ or COMM_FLAG
  int map_temporary;                  // 1 if the atom map was created by this fix
  int local_active;                   // 1 between swap_local_begin() and swap_local_end()
  int energy_valid;                   // 1 if energy_stored is the current full energy

  double nswap_null;       // # of attempts that ended as null (empty) moves
  double verify_maxdev;    // max |dE_local - dE_full| in energy verify mode

  // per swap type atom lists for global proposals (rmax = 0)

  std::vector<std::vector<int>> tlist;
  std::vector<bigint> tcount, tbefore;
  std::vector<int> elist;    // all eligible atoms (proposal geometric)
  bigint ecount, ebefore;

  // fixed geometric graph of eligible atom pairs within rmax (rmax > 0)
  // each edge is stored on the proc owning the atom with the smaller ID

  std::vector<int> eligible;               // per local+ghost atom: 1 if swappable
  std::vector<int> edge_i, edge_j;         // local indices of edge end points (owned if possible)
  std::vector<int> edge_jimg;              // local index of the image of j within rmax of i
  std::vector<int> adj_first, adj_edge;    // edges incident to each local index (CSR)
  std::vector<int> het_list, het_pos;      // hetero edge set and reverse index
  std::vector<bigint> proc_count;          // # of selectable edges on each proc
  bigint gcount;                           // # of selectable edges on all procs

  struct SwapTrial {
    int idx[2];        // local index of owned atom on this proc or -1
    tagint tag[2];     // atom IDs (valid on all procs)
    int oldtype[2];    // atom types before the swap
    int pos[2];        // positions in local atom lists or -1
    int list[2];       // swap type index of the lists or -1
  };

  // checkerboard data: edges inside one cell grouped by color and cell

  struct ColorCells {
    std::vector<bigint> cell;      // sorted IDs of cells with local edges
    std::vector<int> first;        // edges of cell n: edges[first[n]] .. edges[first[n+1]-1]
    std::vector<int> edges;        // edge indices
    std::vector<bigint> mlocal;    // local # of selectable edges per cell
    std::vector<double> wlocal;    // local sum of selection weights per cell
  };
  std::vector<bigint> edge_cell;    // cell ID of each edge, -1 if not inside one cell
  ColorCells color_cells[8];
  int cb_dims[3];    // checkerboard grid

  // local energy and proposal operations

  class PairSwapLocal *local_pair_interface();
  int swap_eval_batch(int, const SwapProposal *, double *);
  void swap_commit_batch(int, const SwapProposal *, const int *);
  void swap_apply_accepted(int, const SwapProposal *, const int *);
  void swap_forward_comm();
  void request_swap_list(double);

  // extension methods

  int local_energy_check(std::string &);
  int local_dynamic_check(std::string &);
  double ghost_cutoff();
  void local_eval_failed();
  int checkerboard_check(std::string &);
  void end_mc_invocation();
  bigint serial_sweep();
  bigint checkerboard_sweep(bigint &);
  int attempt_semi_grand_local();
  int attempt_swap_general();
  int select_legacy(SwapTrial &);
  int select_global(SwapTrial &);
  int select_graph(SwapTrial &);
  int pick_from_list(int, int *);
  void share_trial(SwapTrial &);
  int find_owner(bigint, bigint &);
  void update_type_lists();
  void update_type_counts();
  void update_lists_after_swap(const SwapTrial &);
  void build_graph();
  void update_graph_counts();
  bigint delta_hetero(const SwapTrial &);
  void update_hetero(const SwapTrial &);
  void make_proposal(const SwapTrial &, SwapProposal &);
  void apply_trial(const SwapTrial &, int);
  void set_proposal_types(const SwapProposal &, int);
  void rebuild_lists_if_needed();
  int is_eligible(int);
  double selection_weight(int, int) const;
  int local_delta(int, const SwapProposal *, double *);
  void full_delta_batch(int, const SwapProposal *, double *);
  void verify_batch(int, const SwapProposal *, const double *, const int *);
  void setup_checkerboard_cells(const double *);
  double counter_uniform(int, int, int, bigint, int) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
