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
   read-only helpers for exact local energy changes of atom type swaps.
   the arithmetic mirrors getscreen(), calc_rho1(), meam_dens_final() and
   the energy part of meam_force() for MEAM (not MS-MEAM), so that local
   energy changes agree with full energy evaluations.
------------------------------------------------------------------------- */

#include "meam.h"

#include "lmptype.h"
#include "math_special.h"

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;

/* ----------------------------------------------------------------------
   largest ebound of all element pairs.  atoms screening a pair i-j are
   within sqrt(ebound)*r_ij of both i and j.
------------------------------------------------------------------------- */

double MEAM::swap_ebound_max() const
{
  double ebmax = 0.0;
  for (int i = 0; i < neltypes; i++)
    for (int j = 0; j < neltypes; j++) ebmax = std::max(ebmax, ebound_meam[i][j]);
  return ebmax;
}

/* ----------------------------------------------------------------------
   screening function sij and radial cutoff function fcij of pair i-j with
   the screening atoms taken from the full neighbor list of atom i.
   eltfull[kn] = element of neighbor kn of atom i (-1 to skip).
   same as the first part of getscreen().
------------------------------------------------------------------------- */

void MEAM::swap_screen(int i, int j, int elti, int eltj, const double *const *x, int numneigh_full,
                       const int *firstneigh_full, const int *eltfull, double &sij,
                       double &fcij) const
{
  sij = 0.0;
  fcij = 0.0;
  if ((elti < 0) || (eltj < 0)) return;

  const double drinv = 1.0 / delr_meam;
  const double xitmp = x[i][0];
  const double yitmp = x[i][1];
  const double zitmp = x[i][2];
  const double xjtmp = x[j][0];
  const double yjtmp = x[j][1];
  const double zjtmp = x[j][2];
  const double delxij = xjtmp - xitmp;
  const double delyij = yjtmp - yitmp;
  const double delzij = zjtmp - zitmp;
  const double rij2 = delxij * delxij + delyij * delyij + delzij * delzij;

  if (rij2 > cutforcesq) return;

  const double rbound = ebound_meam[elti][eltj] * rij2;
  const double rij = sqrt(rij2);
  const double rnorm = (cutforce - rij) * drinv;
  double s = 1.0;

  for (int kn = 0; kn < numneigh_full; kn++) {
    const int k = firstneigh_full[kn] & NEIGHMASK;
    if (k == j) continue;
    const int eltk = eltfull[kn];
    if (eltk < 0) continue;

    const double xktmp = x[k][0];
    const double yktmp = x[k][1];
    const double zktmp = x[k][2];

    const double delxjk = xktmp - xjtmp;
    const double delyjk = yktmp - yjtmp;
    const double delzjk = zktmp - zjtmp;
    const double rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
    if (rjk2 > rbound) continue;

    const double delxik = xktmp - xitmp;
    const double delyik = yktmp - yitmp;
    const double delzik = zktmp - zitmp;
    const double rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
    if (rik2 > rbound) continue;

    const double xik = rik2 / rij2;
    const double xjk = rjk2 / rij2;
    const double a = 1 - (xik - xjk) * (xik - xjk);
    if (a <= 0.0) continue;

    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    const double Cmax = Cmax_meam[elti][eltj][eltk];
    const double Cmin = Cmin_meam[elti][eltj][eltk];
    if (cikj >= Cmax) continue;
    if (cikj <= Cmin) {
      s = 0.0;
      break;
    }
    const double delc = Cmax - Cmin;
    cikj = (cikj - Cmin) / delc;
    s *= fcut(cikj);
  }

  double dfc;
  fcij = dfcut(rnorm, dfc);
  sij = s;
}

/* ----------------------------------------------------------------------
   pair energy (meam_force()) of pair i-j with screening scr and cutoff
   function fc, and partial density contributions of atom j to atom i
   (calc_rho1()).  delij = x[j] - x[i].  the density contributions are
   multiplied by sign and added to densi.
------------------------------------------------------------------------- */

void MEAM::swap_pair(double scr, double fc, int elti, int eltj, const double *delij, double rij2,
                     double scaleij, double sign, double &epair, double *densi) const
{
  epair = 0.0;
  if ((elti < 0) || (eltj < 0)) return;
  if (iszero(scr)) return;
  if (!(rij2 < cutforcesq)) return;

  const double sij = scr * fc;
  const double rij = sqrt(rij2);

  // pair energy

  const int ind = eltind[elti][eltj];
  double pp = rij * rdrar;
  int kk = (int) pp;
  kk = std::min(kk, nrar - 2);
  pp = pp - kk;
  pp = std::min(pp, 1.0);
  const double phi =
      ((phirar3[ind][kk] * pp + phirar2[ind][kk]) * pp + phirar1[ind][kk]) * pp + phirar[ind][kk];
  epair = phi * scaleij * sij;

  // partial densities at atom i from atom j

  const double aj = rij / re_meam[eltj][eltj] - 1.0;
  const double ro0j = rho0_meam[eltj];
  const double rhoa0j = ro0j * MathSpecial::fm_exp(-beta0_meam[eltj] * aj) * sij;
  double rhoa1j = ro0j * MathSpecial::fm_exp(-beta1_meam[eltj] * aj) * sij;
  double rhoa2j = ro0j * MathSpecial::fm_exp(-beta2_meam[eltj] * aj) * sij;
  double rhoa3j = ro0j * MathSpecial::fm_exp(-beta3_meam[eltj] * aj) * sij;
  if (ialloy == 1) {
    rhoa1j = rhoa1j * t1_meam[eltj];
    rhoa2j = rhoa2j * t2_meam[eltj];
    rhoa3j = rhoa3j * t3_meam[eltj];
  }

  double *rho0i = densi;
  double *arho2bi = densi + 1;
  double *arho1i = densi + 2;
  double *arho2i = densi + 5;
  double *arho3i = densi + 11;
  double *arho3bi = densi + 21;
  double *t_avei = densi + 24;
  double *tsq_avei = densi + 27;

  *rho0i += sign * rhoa0j;
  if (ialloy != 2) {
    t_avei[0] += sign * (t1_meam[eltj] * rhoa0j);
    t_avei[1] += sign * (t2_meam[eltj] * rhoa0j);
    t_avei[2] += sign * (t3_meam[eltj] * rhoa0j);
  }
  if (ialloy == 1) {
    tsq_avei[0] += sign * (t1_meam[eltj] * t1_meam[eltj] * rhoa0j);
    tsq_avei[1] += sign * (t2_meam[eltj] * t2_meam[eltj] * rhoa0j);
    tsq_avei[2] += sign * (t3_meam[eltj] * t3_meam[eltj] * rhoa0j);
  }
  *arho2bi += sign * rhoa2j;

  const double A1j = rhoa1j / rij;
  const double A2j = rhoa2j / rij2;
  const double A3j = rhoa3j / (rij2 * rij);
  int nv2 = 0;
  int nv3 = 0;
  for (int m = 0; m < 3; m++) {
    arho1i[m] += sign * (A1j * delij[m]);
    arho3bi[m] += sign * (rhoa3j * delij[m] / rij);
    for (int n = m; n < 3; n++) {
      arho2i[nv2] += sign * (A2j * delij[m] * delij[n]);
      nv2 = nv2 + 1;
      for (int p = n; p < 3; p++) {
        arho3i[nv3] += sign * (A3j * delij[m] * delij[n] * delij[p]);
        nv3 = nv3 + 1;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   embedding energy of an atom of element elti with partial densities dens,
   same as meam_dens_final() for MEAM (not MS-MEAM).  dens is not modified.
------------------------------------------------------------------------- */

double MEAM::swap_embedding(const double *dens, int elti, double scaleii, int &errorflag) const
{
  errorflag = 0;
  if (elti < 0) return 0.0;

  const double rho0 = dens[0];
  const double arho2b = dens[1];
  const double *arho1 = dens + 2;
  const double *arho2 = dens + 5;
  const double *arho3 = dens + 11;
  const double *arho3b = dens + 21;
  double t_ave[3] = {dens[24], dens[25], dens[26]};
  const double *tsq_ave = dens + 27;

  double rho1 = 0.0;
  double rho2 = -1.0 / 3.0 * arho2b * arho2b;
  double rho3 = 0.0;
  for (int m = 0; m < 3; m++) {
    rho1 = rho1 + arho1[m] * arho1[m];
    rho3 = rho3 - 3.0 / 5.0 * arho3b[m] * arho3b[m];
  }
  for (int m = 0; m < 6; m++) rho2 = rho2 + v2D[m] * arho2[m] * arho2[m];
  for (int m = 0; m < 10; m++) rho3 = rho3 + v3D[m] * arho3[m] * arho3[m];

  if (rho0 > 0.0) {
    if (ialloy == 1) {
      t_ave[0] = fdiv_zero(t_ave[0], tsq_ave[0]);
      t_ave[1] = fdiv_zero(t_ave[1], tsq_ave[1]);
      t_ave[2] = fdiv_zero(t_ave[2], tsq_ave[2]);
    } else if (ialloy == 2) {
      t_ave[0] = t1_meam[elti];
      t_ave[1] = t2_meam[elti];
      t_ave[2] = t3_meam[elti];
    } else {
      t_ave[0] = t_ave[0] / rho0;
      t_ave[1] = t_ave[1] / rho0;
      t_ave[2] = t_ave[2] / rho0;
    }
  }

  double gamma = t_ave[0] * rho1 + t_ave[1] * rho2 + t_ave[2] * rho3;
  if (rho0 > 0.0) gamma = gamma / (rho0 * rho0);

  const double Z = get_Zij(lattce_meam[elti][elti]);
  const double G = G_gam(gamma, ibar_meam[elti], errorflag);
  if (errorflag != 0) return 0.0;

  double shp[3];
  get_shpfcn(lattce_meam[elti][elti], stheta_meam[elti][elti], ctheta_meam[elti][elti], shp);

  // same error check of the reference state as in meam_dens_final()

  if (ibar_meam[elti] > 0) {
    double gam;
    if (mix_ref_t == 1)
      gam = (t_ave[0] * shp[0] + t_ave[1] * shp[1] + t_ave[2] * shp[2]) / (Z * Z);
    else
      gam = (t1_meam[elti] * shp[0] + t2_meam[elti] * shp[1] + t3_meam[elti] * shp[2]) / (Z * Z);
    G_gam(gam, ibar_meam[elti], errorflag);
    if (errorflag != 0) return 0.0;
  }

  double rho_bkgd;
  const double rho = rho0 * G;

  if (mix_ref_t == 1) {
    double Gbar = 1.0;
    if (ibar_meam[elti] > 0) {
      double dGbar;
      const double gam = (t_ave[0] * shp[0] + t_ave[1] * shp[1] + t_ave[2] * shp[2]) / (Z * Z);
      Gbar = dG_gam(gam, ibar_meam[elti], dGbar);
    }
    rho_bkgd = rho0_meam[elti] * Z * Gbar;
  } else {
    if (bkgd_dyn == 1) {
      rho_bkgd = rho0_meam[elti] * Z;
    } else {
      rho_bkgd = rho_ref_meam[elti];
    }
  }
  const double rhob = rho / rho_bkgd;

  double dF;
  return embedding(A_meam[elti], Ec_meam[elti][elti], rhob, dF) * scaleii;
}
