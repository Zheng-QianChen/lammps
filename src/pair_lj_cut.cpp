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
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_cut.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair_swap_local.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <typeinfo>

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLJCut::PairLJCut(LAMMPS *lmp) :
    Pair(lmp), cut(nullptr), epsilon(nullptr), sigma(nullptr), lj1(nullptr), lj2(nullptr),
    lj3(nullptr), lj4(nullptr), offset(nullptr), cut_respa(nullptr), swaplocal_list(nullptr),
    swaplocal_centers(nullptr)
{
  respa_enable = 1;
  born_matrix_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairLJCut::~PairLJCut()
{
  if (copymode) return;

  delete swaplocal_centers;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCut::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }

        if (eflag) {
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairLJCut::compute_inner()
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_inner;
  ilist = list->ilist_inner;
  numneigh = list->numneigh_inner;
  firstneigh = list->firstneigh_inner;

  double cut_out_on = cut_respa[0];
  double cut_out_off = cut_respa[1];

  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 - rsw * rsw * (3.0 - 2.0 * rsw);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCut::compute_middle()
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum_middle;
  ilist = list->ilist_middle;
  numneigh = list->numneigh_middle;
  firstneigh = list->firstneigh_middle;

  double cut_in_off = cut_respa[0];
  double cut_in_on = cut_respa[1];
  double cut_out_on = cut_respa[2];
  double cut_out_off = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_out_diff = cut_out_off - cut_out_on;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;
  double cut_out_on_sq = cut_out_on * cut_out_on;
  double cut_out_off_sq = cut_out_off * cut_out_off;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cut_out_off_sq && rsq > cut_in_off_sq) {
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        jtype = type[j];
        forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;
        if (rsq < cut_in_on_sq) {
          rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
          fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
        }
        if (rsq > cut_out_on_sq) {
          rsw = (sqrt(rsq) - cut_out_on) / cut_out_diff;
          fpair *= 1.0 + rsw * rsw * (2.0 * rsw - 3.0);
        }

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairLJCut::compute_outer(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair;
  double rsq, r2inv, r6inv, forcelj, factor_lj, rsw;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double cut_in_off = cut_respa[2];
  double cut_in_on = cut_respa[3];

  double cut_in_diff = cut_in_on - cut_in_off;
  double cut_in_off_sq = cut_in_off * cut_in_off;
  double cut_in_on_sq = cut_in_on * cut_in_on;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        if (rsq > cut_in_off_sq) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
          fpair = factor_lj * forcelj * r2inv;
          if (rsq < cut_in_on_sq) {
            rsw = (sqrt(rsq) - cut_in_off) / cut_in_diff;
            fpair *= rsw * rsw * (3.0 - 2.0 * rsw);
          }

          f[i][0] += delx * fpair;
          f[i][1] += dely * fpair;
          f[i][2] += delz * fpair;
          if (newton_pair || j < nlocal) {
            f[j][0] -= delx * fpair;
            f[j][1] -= dely * fpair;
            f[j][2] -= delz * fpair;
          }
        }

        if (eflag) {
          r2inv = 1.0 / rsq;
          r6inv = r2inv * r2inv * r2inv;
          evdwl = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (vflag) {
          if (rsq <= cut_in_off_sq) {
            r2inv = 1.0 / rsq;
            r6inv = r2inv * r2inv * r2inv;
            forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
            fpair = factor_lj * forcelj * r2inv;
          } else if (rsq < cut_in_on_sq)
            fpair = factor_lj * forcelj * r2inv;
        }

        if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
      }
    }
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairLJCut::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;

  memory->create(cutsq, n, n, "pair:cutsq");

  memory->create(cut, n, n, "pair:cut");
  memory->create(epsilon, n, n, "pair:epsilon");
  memory->create(sigma, n, n, "pair:sigma");
  memory->create(lj1, n, n, "pair:lj1");
  memory->create(lj2, n, n, "pair:lj2");
  memory->create(lj3, n, n, "pair:lj3");
  memory->create(lj4, n, n, "pair:lj4");
  memory->create(offset, n, n, "pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairLJCut::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR, "Illegal pair_style command");

  cut_global = utils::numeric(FLERR, arg[0], false, lmp);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i, j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairLJCut::coeff(int narg, char **arg)
{
  if (narg < 4 || narg > 5)
    error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  double epsilon_one = utils::numeric(FLERR, arg[2], false, lmp);
  double sigma_one = utils::numeric(FLERR, arg[3], false, lmp);

  double cut_one = cut_global;
  if (narg == 5) cut_one = utils::numeric(FLERR, arg[4], false, lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients" + utils::errorurl(21));
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairLJCut::init_style()
{
  // request regular or rRESPA neighbor list

  int list_style = NeighConst::REQ_DEFAULT;

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style, "^respa")) {
    auto *respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->level_inner >= 0) list_style = NeighConst::REQ_RESPA_INOUT;
    if (respa->level_middle >= 0) list_style = NeighConst::REQ_RESPA_ALL;
  }
  neighbor->add_request(this, list_style);

  // set rRESPA cutoffs

  if (utils::strmatch(update->integrate_style, "^respa") &&
      (dynamic_cast<Respa *>(update->integrate))->level_inner >= 0)
    cut_respa = (dynamic_cast<Respa *>(update->integrate))->cutoff;
  else
    cut_respa = nullptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairLJCut::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i], epsilon[j][j], sigma[i][i], sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i], sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j], 6.0);

  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio, 12.0) - pow(ratio, 6.0));
  } else
    offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR, "Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2], all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count, all, 2, MPI_DOUBLE, MPI_SUM, world);

    double sig2 = sigma[i][j] * sigma[i][j];
    double sig6 = sig2 * sig2 * sig2;
    double rc3 = cut[i][j] * cut[i][j] * cut[i][j];
    double rc6 = rc3 * rc3;
    double rc9 = rc3 * rc6;
    double prefactor = 8.0 * MY_PI * all[0] * all[1] * epsilon[i][j] * sig6 / (9.0 * rc9);
    etail_ij = prefactor * (sig6 - 3.0 * rc6);
    ptail_ij = 2.0 * prefactor * (2.0 * sig6 - 3.0 * rc6);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCut::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i, j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j], sizeof(double), 1, fp);
        fwrite(&sigma[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCut::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i, j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &epsilon[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &sigma[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&epsilon[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&sigma[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairLJCut::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&tail_flag, sizeof(int), 1, fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairLJCut::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &tail_flag, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&tail_flag, 1, MPI_INT, 0, world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairLJCut::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++) fprintf(fp, "%d %g %g\n", i, epsilon[i][i], sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairLJCut::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g\n", i, j, epsilon[i][j], sigma[i][j], cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairLJCut::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                         double /*factor_coul*/, double factor_lj, double &fforce)
{
  double r2inv, r6inv, forcelj, philj;

  r2inv = 1.0 / rsq;
  r6inv = r2inv * r2inv * r2inv;
  forcelj = r6inv * (lj1[itype][jtype] * r6inv - lj2[itype][jtype]);
  fforce = factor_lj * forcelj * r2inv;

  philj = r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
  return factor_lj * philj;
}

/* ---------------------------------------------------------------------- */

void PairLJCut::born_matrix(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                            double /*factor_coul*/, double factor_lj, double &dupair,
                            double &du2pair)
{
  double rinv, r2inv, r6inv, du, du2;

  r2inv = 1.0 / rsq;
  rinv = sqrt(r2inv);
  r6inv = r2inv * r2inv * r2inv;

  // Reminder: lj1 = 48*e*s^12, lj2 = 24*e*s^6
  // so dupair = -forcelj/r = -fforce*r (forcelj from single method)

  du = r6inv * rinv * (lj2[itype][jtype] - lj1[itype][jtype] * r6inv);
  du2 = r6inv * r2inv * (13 * lj1[itype][jtype] * r6inv - 7 * lj2[itype][jtype]);

  dupair = factor_lj * du;
  du2pair = factor_lj * du2;
}

/* ---------------------------------------------------------------------- */

void *PairLJCut::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "epsilon") == 0) return (void *) epsilon;
  if (strcmp(str, "sigma") == 0) return (void *) sigma;
  return nullptr;
}

/* ----------------------------------------------------------------------
   local energy changes for atom type swaps, see pair_swap_local.h
   only enabled for this exact class.  derived classes (accelerator
   variants or styles with additional terms) must opt in explicitly.
------------------------------------------------------------------------- */

PairSwapLocal *PairLJCut::swap_local()
{
  if (typeid(*this) != typeid(PairLJCut)) return nullptr;
  return this;
}

/* ----------------------------------------------------------------------
   the site energy of atom c is 1/2 sum_j phi(r_cj), so it depends only on
   atoms within the largest cutoff.  pair terms are additive, so two edits
   are independent if no pair of their atoms is within the cutoff.
------------------------------------------------------------------------- */

SwapLocalCaps PairLJCut::swap_local_caps(std::string &reason) const
{
  SwapLocalCaps caps;
  caps.supported = 1;
  caps.supports_charge = 1;    // energy does not depend on charges
  caps.dependency_radius = cutforce;
  caps.conflict_radius = cutforce;

  if (typeid(*this) != typeid(PairLJCut)) {
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

/* ---------------------------------------------------------------------- */

void PairLJCut::swap_local_begin(NeighList *full_list)
{
  delete swaplocal_centers;
  swaplocal_centers = new SwapLocalCenters;
  swaplocal_list = full_list;
  if (swaplocal_centers->setup(atom, full_list)) swaplocal_list = nullptr;
}

/* ---------------------------------------------------------------------- */

void PairLJCut::swap_local_end()
{
  swaplocal_list = nullptr;
  delete swaplocal_centers;
  swaplocal_centers = nullptr;
}

/* ----------------------------------------------------------------------
   pair energy of atoms of type itype and jtype at distance^2 rsq,
   including the energy offset of pair_modify shift yes
------------------------------------------------------------------------- */

double PairLJCut::swaplocal_pair(int itype, int jtype, double rsq) const
{
  if (rsq >= cutsq[itype][jtype]) return 0.0;
  const double r2inv = 1.0 / rsq;
  const double r6inv = r2inv * r2inv * r2inv;
  return r6inv * (lj3[itype][jtype] * r6inv - lj4[itype][jtype]) - offset[itype][jtype];
}

/* ----------------------------------------------------------------------
   owner computes energy changes with site energies E_c = 1/2 sum_j phi(r_cj)
   for each local or ghost image a of an edited atom:
   - if a is owned: change of E_a with all neighbors in their new types
     (this includes other edited atoms and periodic images of a itself)
   - for every unedited owned atom c with a in its list: 1/2 phi change of
     the pair c-a, once per image a, so atoms near several images are exact
   summed over all ranks every pair with at least one edited atom gets its
   full energy change.  special bond factors are not applied, the caller
   rejects setups with special bond exclusions.
------------------------------------------------------------------------- */

int PairLJCut::swap_local_delta_energy_batch(int n, const SwapProposal *proposals, double *dE) const
{
  if (!swaplocal_list) return 1;

  const double *const *const x = atom->x;
  const int *const type = atom->type;
  const tagint *const tag = atom->tag;
  const int *const sametag = atom->sametag;
  const int nlocal = atom->nlocal;
  const int *const numneigh = swaplocal_list->numneigh;
  int **const firstneigh = swaplocal_list->firstneigh;

  for (int k = 0; k < n; k++) {
    const SwapProposal &p = proposals[k];
    double de = 0.0;

    for (int m = 0; m < p.count; m++) {
      const int told = p.oldtype[m];
      const int tnew = p.newtype[m];

      for (int a = atom->map(p.tag[m]); a >= 0; a = sametag[a]) {
        const double xa = x[a][0];
        const double ya = x[a][1];
        const double za = x[a][2];

        if (a < nlocal) {
          const int *jlist = firstneigh[a];
          const int jnum = numneigh[a];
          for (int jj = 0; jj < jnum; jj++) {
            const int j = jlist[jj] & NEIGHMASK;
            const double delx = xa - x[j][0];
            const double dely = ya - x[j][1];
            const double delz = za - x[j][2];
            const double rsq = delx * delx + dely * dely + delz * delz;
            const int w = swap_local_which(p, tag[j]);
            const int jold = (w < 0) ? type[j] : p.oldtype[w];
            const int jnew = (w < 0) ? type[j] : p.newtype[w];
            de += 0.5 * (swaplocal_pair(tnew, jnew, rsq) - swaplocal_pair(told, jold, rsq));
          }
        }

        swaplocal_centers->for_each(a, [&](int c) {
          if (swap_local_which(p, tag[c]) >= 0) return;
          const double delx = x[c][0] - xa;
          const double dely = x[c][1] - ya;
          const double delz = x[c][2] - za;
          const double rsq = delx * delx + dely * dely + delz * delz;
          const int tc = type[c];
          de += 0.5 * (swaplocal_pair(tc, tnew, rsq) - swaplocal_pair(tc, told, rsq));
        });
      }
    }
    dE[k] = de;
  }
  return 0;
}
