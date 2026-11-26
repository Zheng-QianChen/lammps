/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// unit tests for fix atom/swap: legacy behavior, local energy changes from
// lj/cut, checkerboard sweeps across MPI sub-domains, the conditions for
// local energy changes, optionality of extension keywords, and shrink-wrap
// / dynamic triclinic NPT regressions.  can be run with any number of MPI ranks.

#include "MC/fix_atom_swap.h"
#include "atom.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "exceptions.h"
#include "fix.h"
#include "info.h"
#include "input.h"
#include "lammps.h"
#include "modify.h"
#include "platform.h"
#include "update.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "../testing/test_mpi_main.h"

#include <algorithm>
#include <cmath>
#include <utility>

using ::testing::HasSubstr;
using ::testing::Not;

namespace LAMMPS_NS {

namespace {
// max. deviation of local and full energy changes with energy verify
const double EPS_VERIFY = 1.0e-10;

using Lines = std::vector<std::string>;

// geometry of pairs_system(): 4x4x4 pairs on a lattice with constant 6.0
const int NPAIRS           = 64;
const double PAIRS_LATTICE = 6.0;
const double PAIRS_BOX     = 24.0;

// binary fcc lattice with lj/cut and unequal per-type cutoffs
Lines lj_fcc(int n, const std::string &newton = "on", const std::string &style = "atomic")
{
    const std::string box = fmt::format("region box block 0 {0} 0 {0} 0 {0}", n);
    return {"units lj",
            "atom_style " + style,
            "atom_modify map array",
            "newton " + newton,
            "lattice fcc 0.8442",
            box,
            "create_box 2 box",
            "create_atoms 1 box",
            "mass * 1.0",
            "set type 1 type/fraction 2 0.5 4321",
            "velocity all create 1.5 87287 loop geom",
            "pair_style lj/cut 2.5",
            "pair_coeff 1 1 1.0 1.0 2.5",
            "pair_coeff 2 2 0.8 1.1 2.0",
            "pair_coeff 1 2 0.9 1.05 2.8",
            "neighbor 0.3 bin"};
}

// isolated pairs of atoms of type 1 and 2 at distance 1.0 on a simple cubic
// lattice of the pairs with lattice constant 6.0.  pairs with the first atom
// at x = 11.5 and x = 23.5 cross the sub-domain boundaries for "processors
// 2 * *".  with spectator yes, an atom of type 1 is placed at distance 1.8 of
// the first atom of each pair, outside rmax but inside the pair cutoff, so
// that swaps change the energy.  every checkerboard cell contains at most one
// pair of atoms within rmax = 1.5 (M_c <= 1).
Lines pairs_system(bool split, bool spectator)
{
    Lines lines = {"units metal", "atom_style atomic", "atom_modify map array"};
    if (split) lines.emplace_back("processors 2 * *");
    std::string lattice = "lattice custom 6.0 a1 1 0 0 a2 0 1 0 a3 0 0 1 "
                          "basis 0.0 0.0 0.0 basis 0.1666666666666667 0.0 0.0";
    std::string create  = "create_atoms 1 box basis 1 1 basis 2 2";
    if (spectator) {
        lattice += " basis 0.7 0.0 0.0";
        create += " basis 3 1";
    }
    lattice += " origin 0.9166666666666667 0.5 0.5";
    lines.emplace_back(lattice);
    lines.emplace_back("region box block 0 4 0 4 0 4");
    lines.emplace_back("create_box 2 box");
    lines.emplace_back(create);
    lines.emplace_back("mass * 1.0");
    lines.emplace_back("pair_style lj/cut 2.5");
    if (spectator) {
        lines.emplace_back("pair_coeff 1 1 1.0 1.0");
        lines.emplace_back("pair_coeff 2 2 0.6 0.9");
        lines.emplace_back("pair_coeff 1 2 0.3 1.1");
    } else {
        lines.emplace_back("pair_coeff * * 1.0 1.0");
    }
    lines.emplace_back("neighbor 0.3 bin");
    return lines;
}
} // namespace

class FixAtomSwapTest : public ::testing::Test {
protected:
    LAMMPS *lmp = nullptr;
    int me      = 0;
    int nprocs  = 1;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &me);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        lmp = create(MPI_COMM_WORLD);
    }

    void TearDown() override { destroy(lmp); }

    static LAMMPS *create(MPI_Comm comm)
    {
        LAMMPS::argv args = {"FixAtomSwap", "-log", "none", "-echo", "screen", "-nocite"};
        if (!verbose) ::testing::internal::CaptureStdout();
        auto *instance = new LAMMPS(args, comm);
        if (!verbose) ::testing::internal::GetCapturedStdout();
        return instance;
    }

    static void destroy(LAMMPS *&instance)
    {
        if (!verbose) ::testing::internal::CaptureStdout();
        delete instance;
        instance = nullptr;
        if (!verbose) ::testing::internal::GetCapturedStdout();
    }

    // replace the LAMMPS instance by a new one
    void reset()
    {
        destroy(lmp);
        lmp = create(MPI_COMM_WORLD);
    }

    // execute commands and return the screen output (only on MPI rank 0)
    static std::string run(LAMMPS *instance, const Lines &lines)
    {
        ::testing::internal::CaptureStdout();
        try {
            for (const auto &line : lines)
                instance->input->one(line);
        } catch (LAMMPSException &) {
            auto out = ::testing::internal::GetCapturedStdout();
            utils::print(stdout, "{}", out);
            throw;
        }
        auto out = ::testing::internal::GetCapturedStdout();
        if (verbose) utils::print(stdout, "{}", out);
        return out;
    }

    std::string run(const Lines &lines) { return run(lmp, lines); }

    // execute commands and return the error message, empty if there was no error
    std::string error_of(const Lines &lines)
    {
        std::string mesg;
        ::testing::internal::CaptureStdout();
        try {
            for (const auto &line : lines)
                lmp->input->one(line);
        } catch (LAMMPSException &e) {
            mesg = e.what();
        }
        auto out = ::testing::internal::GetCapturedStdout();
        if (verbose) utils::print(stdout, "{}", out);
        return mesg;
    }

    static double fix_vector(LAMMPS *instance, const std::string &id, int n)
    {
        auto *fix = instance->modify->get_fix_by_id(id);
        if (!fix) return -1.0;
        return fix->compute_vector(n);
    }

    double fix_vector(const std::string &id, int n) { return fix_vector(lmp, id, n); }

    // atom types indexed by atom ID, identical on all MPI ranks
    static std::vector<int> types(LAMMPS *instance)
    {
        const auto natoms = (std::size_t)instance->atom->natoms;
        std::vector<int> local(natoms + 1, 0), global(natoms + 1, 0);
        for (int i = 0; i < instance->atom->nlocal; ++i)
            local[instance->atom->tag[i]] = instance->atom->type[i];
        MPI_Allreduce(local.data(), global.data(), (int)natoms + 1, MPI_INT, MPI_MAX,
                      instance->world);
        return global;
    }

    // atom types (or owning MPI rank + 1) of the atoms of pairs_system()
    // indexed by lattice site, independent of the atom IDs, which depend on
    // the number of MPI ranks.  site = nbasis * pair + basis
    static std::vector<int> site_values(LAMMPS *instance, int nbasis, bool owner)
    {
        std::vector<int> local(nbasis * NPAIRS, 0), global(nbasis * NPAIRS, 0);
        double **x = instance->atom->x;
        for (int i = 0; i < instance->atom->nlocal; ++i) {
            const double u = std::fmod(x[i][0] - 5.5 + 2.0 * PAIRS_BOX, PAIRS_BOX);
            const int ix   = static_cast<int>((u + 0.5) / PAIRS_LATTICE);
            const int iy   = static_cast<int>(x[i][1] / PAIRS_LATTICE);
            const int iz   = static_cast<int>(x[i][2] / PAIRS_LATTICE);
            const double r = u - ix * PAIRS_LATTICE;
            const int b    = (r < 0.5) ? 0 : ((r < 2.0) ? 1 : 2);
            const int site = nbasis * (ix + 4 * (iy + 4 * iz)) + b;
            if ((ix > 3) || (iy > 3) || (iz > 3) || (b >= nbasis)) continue;
            local[site] = owner ? instance->comm->me + 1 : instance->atom->type[i];
        }
        MPI_Allreduce(local.data(), global.data(), nbasis * NPAIRS, MPI_INT, MPI_MAX,
                      instance->world);
        return global;
    }

    // Cartesian coordinates indexed by global atom tag (3 * (natoms+1))
    static std::vector<double> coords_by_tag(LAMMPS *instance)
    {
        const auto natoms = (std::size_t)instance->atom->natoms;
        std::vector<double> local(3 * (natoms + 1), 0.0), global(3 * (natoms + 1), 0.0);
        for (int i = 0; i < instance->atom->nlocal; ++i) {
            const auto t     = (std::size_t)instance->atom->tag[i];
            local[3 * t]     = instance->atom->x[i][0];
            local[3 * t + 1] = instance->atom->x[i][1];
            local[3 * t + 2] = instance->atom->x[i][2];
        }
        MPI_Allreduce(local.data(), global.data(), (int)(3 * (natoms + 1)), MPI_DOUBLE, MPI_SUM,
                      instance->world);
        return global;
    }

    bool has_style(const std::string &category, const std::string &name)
    {
        Info info(lmp);
        return info.has_style(category, name);
    }
};

// 1: without the new keywords fix atom/swap keeps its original behavior: 2
// output values, full energies, no messages about local energy changes,
// and bitwise reproducible results of the unchanged code path.

TEST_F(FixAtomSwapTest, legacy)
{
    Lines input = lj_fcc(4);
    input.emplace_back("fix 1 all nve");
    input.emplace_back("fix mc all atom/swap 5 10 29494 1.0 ke yes types 1 2");
    input.emplace_back("fix sg all atom/swap 7 5 1234 1.5 semi-grand yes types 1 2 mu 0.0 0.3");
    input.emplace_back("run 30 post no");

    auto out  = run(input);
    auto *fix = lmp->modify->get_fix_by_id("mc");
    ASSERT_NE(fix, nullptr);
    EXPECT_EQ(fix->size_vector, 2);
    EXPECT_EQ(lmp->modify->get_fix_by_id("sg")->size_vector, 2);
    EXPECT_THAT(out, Not(HasSubstr("Fix atom/swap: using")));
    EXPECT_DOUBLE_EQ(fix_vector("mc", 0), 60.0);
    EXPECT_DOUBLE_EQ(fix_vector("sg", 0), 25.0);
    EXPECT_GT(fix_vector("mc", 1), 0.0);
    const auto first    = types(lmp);
    const double acc[2] = {fix_vector("mc", 1), fix_vector("sg", 1)};
    const double pe     = lmp->modify->get_compute_by_id("thermo_pe")->compute_scalar();

    // the same input in a new instance must give bitwise identical results

    reset();
    run(input);
    EXPECT_EQ(types(lmp), first);
    EXPECT_EQ(fix_vector("mc", 1), acc[0]);
    EXPECT_EQ(fix_vector("sg", 1), acc[1]);
    EXPECT_NEAR(lmp->modify->get_compute_by_id("thermo_pe")->compute_scalar(), pe, 1.0e-10);
}

// 2: lj/cut with unequal per-type cutoffs, serial and checkerboard sweeps,
// newton on and off

TEST_F(FixAtomSwapTest, lj_cut_verify)
{
    for (const std::string newton : {"on", "off"}) {
        for (const std::string mode : {"serial", "checkerboard"}) {
            SCOPED_TRACE("newton " + newton + ", parallel " + mode);
            reset();
            Lines input = lj_fcc((mode == "checkerboard") ? 7 : 4, newton);
            input.emplace_back("fix 1 all nve");
            input.emplace_back(fmt::format("fix mc all atom/swap 5 {} 12345 2.0 types 1 2 rmax 1.5 "
                                           "energy verify parallel {}",
                                           (mode == "checkerboard") ? 1 : 10, mode));
            input.emplace_back("run 10 post no");
            auto out = run(input);
            EXPECT_EQ(lmp->modify->get_fix_by_id("mc")->size_vector, 4);
            EXPECT_GT(fix_vector("mc", 0), 0.0);
            EXPECT_GT(fix_vector("mc", 1), 0.0);
            EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
            if (me == 0) {
                EXPECT_THAT(out, HasSubstr("using local energy changes from pair style lj/cut"));
                if (mode == "checkerboard") {
                    EXPECT_THAT(out, HasSubstr("using checkerboard sweeps with 4x4x4 cells"));
                    EXPECT_THAT(out, Not(HasSubstr("using serial sweeps")));
                }
            }
        }
    }

    // random pairs of atoms of the two types and semi-grand changes

    reset();
    Lines input = lj_fcc(4, "off");
    input.emplace_back("fix 1 all nve");
    input.emplace_back("fix mc all atom/swap 2 5 12345 2.0 types 1 2 energy verify");
    input.emplace_back("fix sg all atom/swap 3 5 4321 2.0 semi-grand yes types 1 2 mu 0.0 0.2 "
                       "energy verify");
    input.emplace_back("run 6 post no");
    run(input);
    EXPECT_GT(fix_vector("mc", 1), 0.0);
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
    EXPECT_GT(fix_vector("sg", 1), 0.0);
    EXPECT_LT(fix_vector("sg", 3), EPS_VERIFY);
}

// checkerboard swaps of pairs of atoms owned by different MPI ranks are
// accepted, all ranks apply them consistently, and the result is the same
// as for a single MPI rank

TEST_F(FixAtomSwapTest, cross_rank_checkerboard)
{
    const bool split = (nprocs % 2) == 0;
    Lines input      = pairs_system(split, true);
    input.emplace_back("fix mc all atom/swap 1 1 7531 600.0 types 1 2 rmax 1.5 "
                       "energy verify parallel checkerboard");
    input.emplace_back("run 6 post no");

    auto out = run(input);
    ASSERT_EQ(lmp->atom->natoms, 3 * NPAIRS);
    EXPECT_GT(fix_vector("mc", 1), 0.0);
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
    if (me == 0) {
        EXPECT_THAT(out, HasSubstr("using checkerboard sweeps with 8x8x8 cells"));
    }

    // initially the first atom of each pair has type 1, the second type 2.
    // the atoms of a pair must have different types and the spectator must
    // keep type 1.

    const auto now   = site_values(lmp, 3, false);
    const auto owner = site_values(lmp, 3, true);
    int nflip = 0, nflip_cross = 0;
    for (int p = 0; p < NPAIRS; ++p) {
        EXPECT_NE(now[3 * p], 0);
        EXPECT_NE(now[3 * p], now[3 * p + 1]);
        EXPECT_EQ(now[3 * p + 2], 1);
        if (now[3 * p] == 2) {
            ++nflip;
            if (owner[3 * p] != owner[3 * p + 1]) ++nflip_cross;
        }
    }
    EXPECT_GT(nflip, 0);
    if (split) {
        EXPECT_GT(nflip_cross, 0);
    }

    // same input with one MPI rank, run by each rank independently

    LAMMPS *serial = create(MPI_COMM_SELF);
    Lines sinput   = pairs_system(false, true);
    sinput.emplace_back("fix mc all atom/swap 1 1 7531 600.0 types 1 2 rmax 1.5 "
                        "energy verify parallel checkerboard");
    sinput.emplace_back("run 6 post no");
    run(serial, sinput);
    EXPECT_EQ(site_values(serial, 3, false), now);
    EXPECT_NEAR(fix_vector(serial, "mc", 0), fix_vector("mc", 0), 1.0e-12);
    EXPECT_NEAR(fix_vector(serial, "mc", 1), fix_vector("mc", 1), 1.0e-12);
    EXPECT_NEAR(fix_vector(serial, "mc", 2), fix_vector("mc", 2), 1.0e-12);
    destroy(serial);
}

// 6: cells with a single selectable pair (M_c = 1) make a null move with
// probability 1/2.  with X = 2 rounds and all swaps accepted, the pair
// states must mix instead of returning to the initial state at each
// invocation (period-2 lockstep).

TEST_F(FixAtomSwapTest, single_edge_cells_mix)
{
    const Lines setup = pairs_system((nprocs % 2) == 0, false);
    Lines input       = setup;
    input.emplace_back("fix mc all atom/swap 1 2 2468 300.0 types 1 2 rmax 1.5 "
                       "proposal hetero energy local parallel checkerboard");
    input.emplace_back("run 0 post no");
    run(input);
    ASSERT_EQ(lmp->atom->natoms, 2 * NPAIRS);
    auto prev       = site_values(lmp, 2, false);
    const auto init = prev;

    int nchanged = 0;
    for (int n = 0; n < 12; ++n) {
        run({"run 1 post no"});
        const auto now = site_values(lmp, 2, false);
        for (int p = 0; p < NPAIRS; ++p) {
            EXPECT_NE(now[2 * p], now[2 * p + 1]);
            if (now[2 * p] != prev[2 * p]) ++nchanged;
        }
        prev = now;
    }
    int nflipped = 0;
    for (int p = 0; p < NPAIRS; ++p)
        if (prev[2 * p] != init[2 * p]) ++nflipped;

    // all attempted swaps have zero energy change and are accepted

    EXPECT_GT(nchanged, 0);
    EXPECT_GT(nflipped, 16);
    EXPECT_LT(nflipped, 48);
    EXPECT_GT(fix_vector("mc", 2), 0.0);
    EXPECT_GT(fix_vector("mc", 1), 0.0);

    // decisions depend only on the seed, the timestep, and the cell

    LAMMPS *serial = create(MPI_COMM_SELF);
    Lines sinput   = pairs_system(false, false);
    sinput.emplace_back("fix mc all atom/swap 1 2 2468 300.0 types 1 2 rmax 1.5 "
                        "proposal hetero energy local parallel checkerboard");
    sinput.emplace_back("run 12 post no");
    run(serial, sinput);
    EXPECT_EQ(site_values(serial, 2, false), prev);
    EXPECT_NEAR(fix_vector(serial, "mc", 1), fix_vector("mc", 1), 1.0e-12);
    destroy(serial);
}

// AABB has one eligible nearest-neighbor pair; its swap produces ABAB,
// with three.  The single-pair null probability changes the Hastings
// acceptance at zero energy change from 1/3 to 2/3.

TEST_F(FixAtomSwapTest, CheckerboardSinglePairCellHastingsRatio)
{
    // Read cell assignments through a base-class member pointer, without
    // changing the fix.  Only grids containing the entire chain in one
    // cell contribute to the acceptance statistic.
    struct CellAssignments : FixAtomSwap {
        static auto member() { return &CellAssignments::edge_cell; }
    };

    auto out =
        run({"units lj", "atom_style atomic", "atom_modify map array sort 0 0.0",
             fmt::format("processors {} 1 1", nprocs), "region box block 0 16 0 16 0 16",
             "create_box 2 box", "create_atoms 1 single 7.85 8 8 units box",
             "create_atoms 1 single 7.95 8 8 units box", "create_atoms 2 single 8.05 8 8 units box",
             "create_atoms 2 single 8.15 8 8 units box", "mass * 1.0", "pair_style lj/cut 4.0",
             "pair_coeff * * 0.0 0.1", "neighbor 0.3 bin",
             "fix mc all atom/swap 1 1 2468 1.0 ke no types 1 2 rmax 0.15 "
             "proposal hetero energy local parallel checkerboard",
             "run 1 post no"});
    if (me == 0) EXPECT_THAT(out, HasSubstr("using checkerboard sweeps with 2x2x2 cells"));
    auto *fix = static_cast<FixAtomSwap *>(lmp->modify->get_fix_by_id("mc"));
    ASSERT_NE(fix, nullptr);
    ASSERT_EQ(fix->size_vector, 4);

    double proposed = 0.0, accepted = 0.0;
    for (int trial = 0; trial < 2000; ++trial) {
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            lmp->atom->type[i] = (lmp->atom->tag[i] <= 2) ? 1 : 2;
        ++lmp->update->ntimestep;
        const double attempts0 = fix->compute_vector(0);
        const double accepts0  = fix->compute_vector(1);
        const double nulls0    = fix->compute_vector(2);
        fix->pre_exchange();

        int local[9] = {}, global[9] = {};
        for (const auto cell : fix->*CellAssignments::member())
            ++local[(cell >= 0 && cell < 8) ? cell : 8];
        MPI_Allreduce(local, global, 9, MPI_INT, MPI_SUM, lmp->world);
        EXPECT_DOUBLE_EQ(fix->compute_vector(0) - attempts0, 8.0);
        if (std::find(global, global + 8, 3) == global + 8) continue;

        // Exclude empty cells and the single-pair 1/2 null moves.
        proposed += fix->compute_vector(0) - attempts0 - (fix->compute_vector(2) - nulls0);
        accepted += fix->compute_vector(1) - accepts0;
    }
    ASSERT_GT(proposed, 800.0);
    EXPECT_NEAR(accepted / proposed, 2.0 / 3.0, 0.05);
}

// 7: MPI ranks without atoms or without candidates must not deadlock

TEST_F(FixAtomSwapTest, empty_ranks)
{
    Lines input;
    if ((nprocs % 2) == 0) input.emplace_back("processors 2 * *");
    for (const auto &line : lj_fcc(4)) {
        if (utils::strmatch(line, "^region box"))
            input.emplace_back("region box block 0 8 0 4 0 4");
        else if (utils::strmatch(line, "^create_atoms")) {
            input.emplace_back("region slab block 0 3.5 INF INF INF INF");
            input.emplace_back("create_atoms 1 region slab");
        } else
            input.push_back(line);
    }
    input.emplace_back("fix cb all atom/swap 1 2 12345 2.0 types 1 2 rmax 1.5 "
                       "energy verify parallel checkerboard");
    input.emplace_back("fix se all atom/swap 1 3 54321 2.0 types 1 2 rmax 1.5 energy verify");
    input.emplace_back("fix rp all atom/swap 1 3 13579 2.0 types 1 2 energy local");
    input.emplace_back("run 3 post no");
    run(input);
    for (const std::string id : {"cb", "se", "rp"}) {
        SCOPED_TRACE("fix " + id);
        EXPECT_GT(fix_vector(id, 0), 0.0);
        EXPECT_GT(fix_vector(id, 1), 0.0);
        EXPECT_LT(fix_vector(id, 3), EPS_VERIFY);
    }
    if ((nprocs % 2) == 0) {
        int nempty = (lmp->atom->nlocal == 0) ? 1 : 0;
        int nall   = 0;
        MPI_Allreduce(&nempty, &nall, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(nall, nprocs / 2);
    }

    // no atoms of type 2: no candidates on any rank

    const double acc[3] = {fix_vector("cb", 1), fix_vector("se", 1), fix_vector("rp", 1)};
    const double att[3] = {fix_vector("cb", 0), fix_vector("se", 0), fix_vector("rp", 0)};
    run({"set group all type 1", "run 2 post no"});
    EXPECT_EQ(fix_vector("cb", 1), acc[0]);
    EXPECT_EQ(fix_vector("se", 1), acc[1]);
    EXPECT_EQ(fix_vector("rp", 1), acc[2]);
    EXPECT_GT(fix_vector("cb", 0), att[0]);
    EXPECT_GT(fix_vector("se", 0), att[1]);
    EXPECT_GT(fix_vector("rp", 0), att[2]);
}

// 8: triclinic box that changes with fix npt

TEST_F(FixAtomSwapTest, triclinic_npt)
{
    Lines input;
    for (const auto &line : lj_fcc(7)) {
        if (utils::strmatch(line, "^region box"))
            input.emplace_back("region box prism 0 7 0 7 0 7 0.6 0.4 0.3");
        else
            input.push_back(line);
    }
    input.emplace_back("fix 1 all npt temp 1.5 1.5 0.5 tri 2.0 2.0 2.0");
    input.emplace_back("fix mc all atom/swap 10 1 12345 2.0 types 1 2 rmax 1.5 "
                       "energy verify parallel checkerboard");
    input.emplace_back("run 0 post no");
    run(input);
    const double xy  = lmp->domain->xy;
    const double vol = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
    auto out         = run({"run 20 post no"});
    EXPECT_NE(lmp->domain->xy, xy);
    EXPECT_NE(lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd, vol);
    EXPECT_GT(fix_vector("mc", 1), 0.0);
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
    if (me == 0) {
        EXPECT_THAT(out, Not(HasSubstr("using serial sweeps")));
    }
}

// 9: settings that do not allow local energy changes

TEST_F(FixAtomSwapTest, local_energy_gates)
{
    const std::string fix = "fix mc all atom/swap 1 2 12345 2.0 types 1 2 energy local";
    struct Gate {
        std::string name, style;
        Lines before, after;
        std::string reason;
    };
    const std::vector<Gate> gates = {
        {"exclude",
         "atomic",
         {},
         {"neigh_modify exclude type 1 2"},
         "neigh_modify exclude is used"},
        {"tiled", "atomic", {}, {"comm_style tiled"}, "local energy requires comm_style brick"},
        {"respa", "atomic", {}, {"run_style respa 2 2"}, "run_style respa is used"},
        {"efield",
         "charge",
         {},
         {"fix ef all efield 0.1 0.0 0.0", "fix_modify ef energy yes"},
         "fix ef contributes to the potential energy"},
        {"rmass", "sphere", {}, {}, "per-atom masses are not supported"}};

    for (const auto &g : gates) {
        SCOPED_TRACE("gate " + g.name);
        reset();
        Lines input = lj_fcc(4, "on", g.style);
        if (g.style == "sphere") {
            // per-atom masses replace the per-type masses
            input.erase(std::remove(input.begin(), input.end(), "mass * 1.0"), input.end());
        }
        input.insert(input.end(), g.after.begin(), g.after.end());
        input.push_back(fix);
        input.emplace_back("run 0 post no");
        EXPECT_THAT(error_of(input),
                    HasSubstr("Fix atom/swap energy local is not possible: " + g.reason));
    }

    // fix efield without contribution to the energy is fine

    reset();
    Lines input = lj_fcc(4, "on", "charge");
    input.emplace_back("fix ef all efield 0.1 0.0 0.0");
    input.push_back(fix);
    input.emplace_back("run 1 post no");
    EXPECT_EQ(error_of(input), "");

    // energy auto uses full energies instead

    reset();
    input = lj_fcc(4);
    input.emplace_back("neigh_modify exclude type 1 2");
    input.emplace_back("fix mc all atom/swap 1 2 12345 2.0 types 1 2 energy auto");
    input.emplace_back("run 1 post no");
    auto out = run(input);
    EXPECT_GT(fix_vector("mc", 0), 0.0);
    if (me == 0) {
        EXPECT_THAT(out, HasSubstr("using full energy evaluation"));
    }

    // pairs within rmax cannot be found with neighbor list exclusions

    reset();
    input = lj_fcc(4);
    input.emplace_back("neigh_modify exclude type 1 2");
    input.emplace_back("fix mc all atom/swap 1 2 12345 2.0 types 1 2 rmax 1.5 energy auto");
    input.emplace_back("run 0 post no");
    EXPECT_THAT(error_of(input),
                HasSubstr("Fix atom/swap rmax is not compatible with neigh_modify exclude"));

    // ghost atoms must cover rmax

    reset();
    input = lj_fcc(4);
    input.emplace_back("fix mc all atom/swap 1 2 12345 2.0 types 1 2 rmax 4.0 energy local");
    input.emplace_back("run 0 post no");
    EXPECT_THAT(error_of(input), HasSubstr("exceeds the communication cutoff"));

    reset();
    input = lj_fcc(4);
    input.emplace_back("comm_modify cutoff 4.5");
    input.emplace_back("fix mc all atom/swap 1 2 12345 2.0 types 1 2 rmax 4.0 energy verify");
    input.emplace_back("run 1 post no");
    EXPECT_EQ(error_of(input), "");
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
}

// binary restart with the same number of MPI ranks and checkerboard sweeps.
// all ranks must keep the seed of the fix command, else they make different
// decisions and wait for each other forever.  atoms do not move and the
// random numbers of checkerboard sweeps depend only on the seed and the
// timestep, so the result must be the same as without restart.

TEST_F(FixAtomSwapTest, checkerboard_restart)
{
    const std::string file = "test_fix_atom_swap_checkerboard.restart";
    const std::string fix  = "fix mc all atom/swap 1 1 7531 600.0 types 1 2 rmax 1.5 "
                             "energy verify parallel checkerboard";
    const bool split       = (nprocs % 2) == 0;

    Lines input = pairs_system(split, true);
    input.push_back(fix);
    input.emplace_back("run 4 post no");
    input.emplace_back("write_restart " + file);
    run(input);
    EXPECT_GT(fix_vector("mc", 1), 0.0);

    reset();
    input.clear();
    if (split) input.emplace_back("processors 2 * *");
    input.emplace_back("read_restart " + file);
    input.push_back(fix);
    input.emplace_back("run 4 post no");
    run(input);
    EXPECT_DOUBLE_EQ(fix_vector("mc", 0), 8.0 * 512.0);
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);
    const auto restarted = site_values(lmp, 3, false);
    const double acc     = fix_vector("mc", 1);

    MPI_Barrier(MPI_COMM_WORLD);
    if (me == 0) platform::unlink(file);

    // the same run without restart

    reset();
    input = pairs_system(split, true);
    input.push_back(fix);
    input.emplace_back("run 8 post no");
    run(input);
    EXPECT_EQ(site_values(lmp, 3, false), restarted);
    EXPECT_EQ(fix_vector("mc", 1), acc);
}

// 12: with neighbor list exclusions by type the neighbor lists must be
// rebuilt after a change of types also for equal cutoffs.  in this system,
// atoms 1 and 2 of type 1 are close, so every swap creates a strongly
// repulsive 1-2 pair (energy change > +40) and must be rejected.  with
// stale exclusions the energy change would be negative (-4.0 or -2.0).

TEST_F(FixAtomSwapTest, exclude_full_energy)
{
    const Lines setup = {"units lj",
                         "atom_style atomic",
                         "atom_modify map array",
                         "region box block 0 10 0 10 0 10",
                         "create_box 2 box",
                         "create_atoms 1 single 5.0 5.0 5.0 units box",
                         "create_atoms 1 single 5.8 5.0 5.0 units box",
                         "create_atoms 2 single 5.0 6.1225 5.0 units box",
                         "mass * 1.0",
                         "pair_style lj/cut 2.5",
                         "pair_coeff 1 1 5.0 1.0",
                         "pair_coeff 2 2 1.0 1.0",
                         "pair_coeff 1 2 1.0 1.0",
                         "neighbor 0.3 bin",
                         "neigh_modify exclude type 1 1"};

    for (const std::string proposal : {"", "proposal hetero "}) {
        SCOPED_TRACE("proposal " + proposal);
        reset();
        Lines input = setup;
        input.emplace_back("fix mc all atom/swap 1 10 12345 1.0 types 1 2 " + proposal +
                           "energy full");
        input.emplace_back("run 0 post no");
        run(input);
        ASSERT_EQ(lmp->atom->natoms, 3);
        auto *pe        = lmp->modify->get_compute_by_id("thermo_pe");
        const double e0 = pe->compute_scalar();
        const auto t0   = types(lmp);

        run({"run 5 post no"});
        EXPECT_DOUBLE_EQ(fix_vector("mc", 0), 50.0);
        EXPECT_EQ(fix_vector("mc", 1), 0.0);
        EXPECT_EQ(types(lmp), t0);
        run({"run 0 post no"});
        EXPECT_NEAR(pe->compute_scalar(), e0, 1.0e-12);
    }
}

// 13: type labels in the weights of proposal geometric

TEST_F(FixAtomSwapTest, weights_type_labels)
{
    Lines input = lj_fcc(4);
    input.emplace_back("labelmap atom 1 A 2 B");
    input.emplace_back("fix mc all atom/swap 1 20 12345 2.0 types 1 2 "
                       "proposal geometric weights A B 0.5 energy verify");
    input.emplace_back("run 2 post no");
    EXPECT_EQ(error_of(input), "");
    EXPECT_DOUBLE_EQ(fix_vector("mc", 0), 40.0);
    EXPECT_GT(fix_vector("mc", 1), 0.0);
    EXPECT_LT(fix_vector("mc", 3), EPS_VERIFY);

    reset();
    input = lj_fcc(4);
    input.emplace_back("labelmap atom 1 A 2 B");
    input.emplace_back("fix mc all atom/swap 1 5 12345 2.0 types 1 2 "
                       "proposal geometric weights A A 0.5");
    input.emplace_back("run 0 post no");
    EXPECT_THAT(error_of(input), HasSubstr("must refer to two different types"));
}

// omitted extension keywords keep the original 2-vector algorithm; explicit
// energy full or parallel serial select the 4-vector extension path

TEST_F(FixAtomSwapTest, optionality)
{
    Lines input = lj_fcc(4);
    input.emplace_back("fix 1 all nve");
    input.emplace_back("fix mc all atom/swap 5 10 29494 1.0 ke yes types 1 2");
    input.emplace_back("run 0 post no");
    run(input);
    EXPECT_EQ(lmp->modify->get_fix_by_id("mc")->size_vector, 2);

    reset();
    input = lj_fcc(4);
    input.emplace_back("fix 1 all nve");
    input.emplace_back("fix mc all atom/swap 5 10 29494 1.0 ke yes types 1 2 energy full");
    input.emplace_back("run 0 post no");
    run(input);
    EXPECT_EQ(lmp->modify->get_fix_by_id("mc")->size_vector, 4);

    reset();
    input = lj_fcc(4);
    input.emplace_back("fix 1 all nve");
    input.emplace_back("fix mc all atom/swap 5 10 29494 1.0 ke yes types 1 2 parallel serial");
    input.emplace_back("run 0 post no");
    run(input);
    EXPECT_EQ(lmp->modify->get_fix_by_id("mc")->size_vector, 4);
}

// dynamic triclinic NPT: neighbor bin and nsq must agree on accepts and PE

TEST_F(FixAtomSwapTest, DynamicTriclinicNPTBinMatchesNSQ)
{
    if (!has_style("fix", "npt")) GTEST_SKIP() << "Fix npt is not available";

    auto run_style = [this](const std::string &neigh) {
        reset();
        Lines input;
        for (const auto &line : lj_fcc(4)) {
            if (utils::strmatch(line, "^region box"))
                input.emplace_back("region box prism 0 4 0 4 0 4 0.6 0.4 0.3");
            else if (utils::strmatch(line, "^neighbor"))
                input.emplace_back("neighbor 0.1 " + neigh);
            else
                input.push_back(line);
        }
        input.emplace_back("fix 1 all npt temp 1.5 1.5 0.5 tri 2.0 2.0 2.0");
        input.emplace_back("fix mc all atom/swap 5 10 29494 1.0 ke yes types 1 2");
        input.emplace_back("run 20 post no");
        run(input);
        const double accepts = fix_vector("mc", 1);
        const double pe      = lmp->modify->get_compute_by_id("thermo_pe")->compute_scalar();
        return std::make_pair(accepts, pe);
    };

    const auto bin = run_style("bin");
    const auto nsq = run_style("nsq");
    EXPECT_EQ(bin.first, nsq.first);
    EXPECT_NEAR(bin.second, nsq.second, 1.0e-10);
}

// expanding orthogonal shrink-wrap: four atoms stay present, none lost

TEST_F(FixAtomSwapTest, ExpandingShrinkWrapNoLostAtoms)
{
    const Lines input = {"units lj",
                         "atom_style atomic",
                         "atom_modify map array",
                         "boundary s s s",
                         "region box block -3 3 -3 3 -3 3",
                         "create_box 2 box",
                         "create_atoms 1 single -1 -1 -1 units box",
                         "create_atoms 2 single 1 1 -1 units box",
                         "create_atoms 1 single -1 1 1 units box",
                         "create_atoms 2 single 1 -1 1 units box",
                         "mass * 1.0",
                         "variable vx atom 2.0*x",
                         "variable vy atom 2.0*y",
                         "variable vz atom 2.0*z",
                         "velocity all set v_vx v_vy v_vz units box",
                         "pair_style lj/cut 2.5",
                         "pair_coeff * * 0.0 1.0",
                         "neighbor 0.3 bin",
                         "neigh_modify every 1 delay 0 check no",
                         "timestep 0.01",
                         "fix integrate all nve",
                         "fix mc all atom/swap 1 5 12345 1.0 types 1 2",
                         "thermo 1",
                         "run 0 post no"};
    run(input);
    const double volume_start = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
    const auto out            = run({"run 10 post no"});
    EXPECT_EQ(lmp->atom->natoms, 4);
    EXPECT_EQ(lmp->update->ntimestep, 10);
    int nlocal = lmp->atom->nlocal, count = 0;
    MPI_Allreduce(&nlocal, &count, 1, MPI_INT, MPI_SUM, lmp->world);
    EXPECT_EQ(count, 4);
    EXPECT_THAT(out, Not(HasSubstr("Lost atoms")));
    EXPECT_GT(lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd, volume_start);
}

// triclinic shrink-wrap with ncycles 0 must match plain Verlet (expanding)

TEST_F(FixAtomSwapTest, TriclinicShrinkWrapIdempotentZeroCycles)
{
    const Lines setup = {"units lj",
                         "atom_style atomic",
                         "boundary s s p",
                         "atom_modify map array sort 0 0.0",
                         "region box prism -5 5 -5 5 -5 5 2.0 0.0 0.0 units box",
                         "create_box 2 box",
                         "create_atoms 1 single -3 -3 0 units box",
                         "create_atoms 2 single 3 3 0 units box",
                         "create_atoms 1 single -1 -1 1 units box",
                         "create_atoms 2 single 1 1 -1 units box",
                         "mass * 1",
                         "pair_style lj/cut 2.5",
                         "pair_coeff * * 0.0 1.0",
                         "neighbor 0.3 bin",
                         "neigh_modify every 1 delay 0 check no",
                         "group left id 1 3",
                         "group right id 2 4",
                         "group top id 2 3",
                         "group bottom id 1 4",
                         "velocity left set -2.0 0.0 0.0",
                         "velocity right set 2.0 0.0 0.0",
                         "velocity top set NULL 1.0 NULL",
                         "velocity bottom set NULL -1.0 NULL",
                         "timestep 0.0001",
                         "fix integrate all nve",
                         "thermo 1"};
    run(setup);
    run({"run 2 post no"});
    const double volume_start = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
    run({"run 8 post no"});
    ASSERT_EQ(lmp->atom->natoms, 4);
    ASSERT_EQ(lmp->update->ntimestep, 10);
    const double boxlo[3] = {lmp->domain->boxlo[0], lmp->domain->boxlo[1], lmp->domain->boxlo[2]};
    const double boxhi[3] = {lmp->domain->boxhi[0], lmp->domain->boxhi[1], lmp->domain->boxhi[2]};
    const double xy       = lmp->domain->xy;
    const double volume   = lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd;
    const auto coords     = coords_by_tag(lmp);
    EXPECT_GT(volume, volume_start);

    reset();
    run(setup);
    run({"run 2 post no", "fix mc all atom/swap 1 0 12345 1.5 types 1 2", "run 8 post no"});
    ASSERT_EQ(lmp->atom->natoms, 4);
    ASSERT_EQ(lmp->update->ntimestep, 10);
    for (int d = 0; d < 3; ++d) {
        EXPECT_NEAR(lmp->domain->boxlo[d], boxlo[d], 1.0e-10 * (1.0 + std::fabs(boxlo[d])));
        EXPECT_NEAR(lmp->domain->boxhi[d], boxhi[d], 1.0e-10 * (1.0 + std::fabs(boxhi[d])));
    }
    EXPECT_NEAR(lmp->domain->xy, xy, 1.0e-10 * (1.0 + std::fabs(xy)));
    EXPECT_NEAR(lmp->domain->xprd * lmp->domain->yprd * lmp->domain->zprd, volume,
                1.0e-10 * volume);
    const auto actual = coords_by_tag(lmp);
    ASSERT_EQ(actual.size(), coords.size());
    for (std::size_t i = 0; i < coords.size(); ++i)
        EXPECT_NEAR(actual[i], coords[i], 1.0e-10 * (1.0 + std::fabs(coords[i])));
}

// checkerboard restart across a 4 -> 2 -> 1 rank reduction via subcommunicators

TEST_F(FixAtomSwapTest, checkerboard_restart_rank_reduction)
{
    if (nprocs < 4) GTEST_SKIP() << "needs at least 4 MPI ranks";

    const std::string file = "test_fix_atom_swap_rank_reduction.restart";
    const std::string fix  = "fix mc all atom/swap 1 1 7531 600.0 types 1 2 rmax 1.5 "
                             "energy verify parallel checkerboard";

    MPI_Comm comm4 = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, (me < 4) ? 0 : MPI_UNDEFINED, me, &comm4);
    if (me < 4) {
        LAMMPS *l4  = create(comm4);
        Lines input = pairs_system(true, true);
        input.push_back(fix);
        input.emplace_back("run 4 post no");
        input.emplace_back("write_restart " + file);
        run(l4, input);
        EXPECT_GT(fix_vector(l4, "mc", 1), 0.0);
        destroy(l4);
        MPI_Comm_free(&comm4);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Comm comm2 = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, (me < 2) ? 0 : MPI_UNDEFINED, me, &comm2);
    if (me < 2) {
        LAMMPS *l2  = create(comm2);
        Lines input = {"processors 2 * *", "read_restart " + file, fix, "run 2 post no",
                       "write_restart " + file};
        run(l2, input);
        EXPECT_GT(fix_vector(l2, "mc", 0), 0.0);
        EXPECT_LT(fix_vector(l2, "mc", 3), EPS_VERIFY);
        destroy(l2);
        MPI_Comm_free(&comm2);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (me == 0) {
        LAMMPS *l1  = create(MPI_COMM_SELF);
        Lines input = {"read_restart " + file, fix, "run 1 post no"};
        run(l1, input);
        EXPECT_GT(fix_vector(l1, "mc", 0), 0.0);
        EXPECT_LT(fix_vector(l1, "mc", 3), EPS_VERIFY);
        destroy(l1);
        platform::unlink(file);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace LAMMPS_NS
