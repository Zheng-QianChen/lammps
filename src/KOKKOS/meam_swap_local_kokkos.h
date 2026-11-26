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

/* ----------------------------------------------------------------------
   Device read-only MEAM helpers for exact local energy changes of atom
   type swaps.  Arithmetic mirrors MEAM::swap_screen / swap_pair /
   swap_embedding (not MS-MEAM).  Small parameter arrays travel as a POD
   copy; pair-potential tables are small device views.
------------------------------------------------------------------------- */

#ifndef LMP_MEAM_SWAP_LOCAL_KOKKOS_H
#define LMP_MEAM_SWAP_LOCAL_KOKKOS_H

#include "math_special_kokkos.h"
#include "meam_kokkos.h"
#include "pair_swap_local_kokkos.h"

#include <cmath>

namespace LAMMPS_NS {

struct MeamSwapParams {
  static constexpr int MAX = MAXELT;
  static constexpr int NDENS = MEAM::NSWAPDENS;

  double cutforce, cutforcesq, delr_meam, rdrar, gsmooth_factor;
  int nrar, ialloy, mix_ref_t, bkgd_dyn, emb_lin_neg;
  int v2D[6], v3D[10];
  int eltind[MAX][MAX];
  int ibar_meam[MAX];
  MEAM::lattice_t lattce_meam[MAX][MAX];
  double ebound_meam[MAX][MAX];
  double Cmin_meam[MAX][MAX][MAX];
  double Cmax_meam[MAX][MAX][MAX];
  double Ec_meam[MAX][MAX], re_meam[MAX][MAX];
  double A_meam[MAX], rho0_meam[MAX], rho_ref_meam[MAX];
  double beta0_meam[MAX], beta1_meam[MAX], beta2_meam[MAX], beta3_meam[MAX];
  double t1_meam[MAX], t2_meam[MAX], t3_meam[MAX];
  double stheta_meam[MAX][MAX], ctheta_meam[MAX][MAX];
};

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static bool meam_swap_iszero(const double f)
{
  return Kokkos::fabs(f) < 1e-20;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_fdiv_zero(const double n, const double d)
{
  if (meam_swap_iszero(d)) return 0.0;
  return n / d;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_fcut(const double xi)
{
  if (xi >= 1.0) return 1.0;
  if (xi <= 0.0) return 0.0;
  double a = 1.0 - xi;
  a *= a;
  a *= a;
  a = 1.0 - a;
  return a * a;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_dfcut(const double xi, double &dfc)
{
  if (xi >= 1.0) {
    dfc = 0.0;
    return 1.0;
  }
  if (xi <= 0.0) {
    dfc = 0.0;
    return 0.0;
  }
  const double a = 1.0 - xi;
  const double a3 = a * a * a;
  const double a4 = a * a3;
  const double a1m4 = 1.0 - a4;
  dfc = 8.0 * a1m4 * a3;
  return a1m4 * a1m4;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_G_gam(const MeamSwapParams &p, const double gamma, const int ibar,
                              int &errorflag)
{
  switch (ibar) {
    case 0:
    case 4: {
      const double gsmooth_switchpoint = -p.gsmooth_factor / (p.gsmooth_factor + 1.0);
      if (gamma < gsmooth_switchpoint) {
        if (gamma == 0.0) return 0.0;
        const double G = 1.0 / (p.gsmooth_factor + 1.0) *
            Kokkos::pow(gsmooth_switchpoint / gamma, p.gsmooth_factor);
        return Kokkos::sqrt(G);
      }
      return Kokkos::sqrt(1.0 + gamma);
    }
    case 1:
      return MathSpecialKokkos::fm_exp(gamma / 2.0);
    case 3:
      return 2.0 / (1.0 + MathSpecialKokkos::fm_exp(-gamma));
    case -5:
      if ((1.0 + gamma) >= 0.0) return Kokkos::sqrt(1.0 + gamma);
      return -Kokkos::sqrt(-1.0 - gamma);
  }
  errorflag = 1;
  return 0.0;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_dG_gam(const MeamSwapParams &p, const double gamma, const int ibar,
                               double &dG)
{
  switch (ibar) {
    case 0:
    case 4: {
      const double gsmooth_switchpoint = -p.gsmooth_factor / (p.gsmooth_factor + 1.0);
      if (gamma < gsmooth_switchpoint) {
        double G = 1.0 / (p.gsmooth_factor + 1.0) *
            Kokkos::pow(gsmooth_switchpoint / gamma, p.gsmooth_factor);
        G = Kokkos::sqrt(G);
        dG = -p.gsmooth_factor * G / (2.0 * gamma);
        return G;
      }
      const double G = Kokkos::sqrt(1.0 + gamma);
      dG = 1.0 / (2.0 * G);
      return G;
    }
    case 1: {
      const double G = MathSpecialKokkos::fm_exp(gamma / 2.0);
      dG = G / 2.0;
      return G;
    }
    case 3: {
      const double G = 2.0 / (1.0 + MathSpecialKokkos::fm_exp(-gamma));
      dG = G * (2.0 - G) / 2.0;
      return G;
    }
    case -5:
      if ((1.0 + gamma) >= 0.0) {
        const double G = Kokkos::sqrt(1.0 + gamma);
        dG = 1.0 / (2.0 * G);
        return G;
      } else {
        const double G = -Kokkos::sqrt(-1.0 - gamma);
        dG = -1.0 / (2.0 * G);
        return G;
      }
  }
  dG = 1.0;
  return 0.0;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_embedding_F(const MeamSwapParams &p, const double A, const double Ec,
                                    const double rhobar, double &dF)
{
  const double AEc = A * Ec;
  if (rhobar > 0.0) {
    const double lrb = Kokkos::log(rhobar);
    dF = AEc * (1.0 + lrb);
    return AEc * rhobar * lrb;
  }
  if (p.emb_lin_neg == 0) {
    dF = 0.0;
    return 0.0;
  }
  dF = -AEc;
  return -AEc * rhobar;
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static void meam_swap_get_shpfcn(const MEAM::lattice_t latt, const double sthe, const double cthe,
                                 double (&s)[3])
{
  switch (latt) {
    case MEAM::FCC:
    case MEAM::BCC:
    case MEAM::B1:
    case MEAM::B2:
    case MEAM::SC:
      s[0] = s[1] = s[2] = 0.0;
      break;
    case MEAM::HCP:
      s[0] = s[1] = 0.0;
      s[2] = 1.0 / 3.0;
      break;
    case MEAM::CH4:
    case MEAM::DIA:
    case MEAM::DIA3:
      s[0] = s[1] = 0.0;
      s[2] = 32.0 / 9.0;
      break;
    case MEAM::DIM:
      s[0] = 1.0;
      s[1] = 2.0 / 3.0;
      s[2] = 0.40;
      break;
    case MEAM::LIN:
      s[0] = 0.0;
      s[1] = 8.0 / 3.0;
      s[2] = 0.0;
      break;
    case MEAM::ZIG:
    case MEAM::TRI:
      s[0] = 4.0 * cthe * cthe;
      s[1] = 4.0 * (cthe * cthe * cthe * cthe + sthe * sthe * sthe * sthe - 1.0 / 3.0);
      s[2] = 4.0 * (cthe * cthe * (3.0 * sthe * sthe * sthe * sthe + cthe * cthe * cthe * cthe));
      s[2] = s[2] - 0.6 * s[0];
      break;
    default:
      s[0] = s[1] = s[2] = 0.0;
      break;
  }
}

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static int meam_swap_get_Zij(const MEAM::lattice_t latt)
{
  switch (latt) {
    case MEAM::FCC:
      return 12;
    case MEAM::BCC:
      return 8;
    case MEAM::HCP:
      return 12;
    case MEAM::DIA:
    case MEAM::DIA3:
      return 4;
    case MEAM::DIM:
      return 1;
    case MEAM::B1:
    case MEAM::SC:
      return 6;
    case MEAM::C11:
      return 10;
    case MEAM::L12:
      return 12;
    case MEAM::B2:
      return 8;
    case MEAM::CH4:
      return 4;
    case MEAM::LIN:
    case MEAM::ZIG:
    case MEAM::TRI:
      return 2;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

// use_new != 0 selects newtype overrides from prop for screening atoms

template <class XView, class NeighView, class MapView, class TypeView, class TagView>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION void
meam_swap_screen(const MeamSwapParams &p, int i, int j, int elti, int eltj, const XView &x,
                 int numneigh_full, const NeighView &neighbors, const MapView &map,
                 const TypeView &type, const TagView &tag, const SwapProposal &prop, int use_new,
                 double &sij, double &fcij)
{
  sij = 0.0;
  fcij = 0.0;
  if ((elti < 0) || (eltj < 0)) return;

  const double drinv = 1.0 / p.delr_meam;
  const double delxij = x(j, 0) - x(i, 0);
  const double delyij = x(j, 1) - x(i, 1);
  const double delzij = x(j, 2) - x(i, 2);
  const double rij2 = delxij * delxij + delyij * delyij + delzij * delzij;
  if (rij2 > p.cutforcesq) return;

  const double rbound = p.ebound_meam[elti][eltj] * rij2;
  const double rij = Kokkos::sqrt(rij2);
  const double rnorm = (p.cutforce - rij) * drinv;
  double s = 1.0;

  for (int kn = 0; kn < numneigh_full; kn++) {
    const int k = neighbors(i, kn) & NEIGHMASK;
    if (k == j) continue;
    const int tk = use_new ? swap_kk_newtype(prop, tag(k), type(k))
                           : swap_kk_oldtype(prop, tag(k), type(k));
    const int eltk = map(tk);
    if (eltk < 0) continue;

    const double delxjk = x(k, 0) - x(j, 0);
    const double delyjk = x(k, 1) - x(j, 1);
    const double delzjk = x(k, 2) - x(j, 2);
    const double rjk2 = delxjk * delxjk + delyjk * delyjk + delzjk * delzjk;
    if (rjk2 > rbound) continue;

    const double delxik = x(k, 0) - x(i, 0);
    const double delyik = x(k, 1) - x(i, 1);
    const double delzik = x(k, 2) - x(i, 2);
    const double rik2 = delxik * delxik + delyik * delyik + delzik * delzik;
    if (rik2 > rbound) continue;

    const double xik = rik2 / rij2;
    const double xjk = rjk2 / rij2;
    const double a = 1.0 - (xik - xjk) * (xik - xjk);
    if (a <= 0.0) continue;

    double cikj = (2.0 * (xik + xjk) + a - 2.0) / a;
    const double Cmax = p.Cmax_meam[elti][eltj][eltk];
    const double Cmin = p.Cmin_meam[elti][eltj][eltk];
    if (cikj >= Cmax) continue;
    if (cikj <= Cmin) {
      s = 0.0;
      break;
    }
    cikj = (cikj - Cmin) / (Cmax - Cmin);
    s *= meam_swap_fcut(cikj);
  }

  double dfc;
  fcij = meam_swap_dfcut(rnorm, dfc);
  sij = s;
}

/* ---------------------------------------------------------------------- */

// Evaluate both screening products with one walk over the geometry.  Most
// screening atoms keep their element, so their cutoff factor is shared.

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_screen_factor(const MeamSwapParams &p, int ei, int ej, int ek, double c)
{
  const double cmax = p.Cmax_meam[ei][ej][ek];
  const double cmin = p.Cmin_meam[ei][ej][ek];
  if (c >= cmax) return 1.0;
  if (c <= cmin) return 0.0;
  return meam_swap_fcut((c - cmin) / (cmax - cmin));
}

template <class XView, class NeighView, class MapView, class TypeView, class TagView>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION void
meam_swap_screen_both(const MeamSwapParams &p, int i, int j, int eio, int ejo, int ein, int ejn,
                      const XView &x, int nfull, const NeighView &neighbors, const MapView &map,
                      const TypeView &type, const TagView &tag, const SwapProposal &prop,
                      double &sold, double &snew, double &fc)
{
  const double dx = x(j, 0) - x(i, 0);
  const double dy = x(j, 1) - x(i, 1);
  const double dz = x(j, 2) - x(i, 2);
  const double rij2 = dx * dx + dy * dy + dz * dz;
  const double bo = p.ebound_meam[eio][ejo] * rij2;
  const double bn = p.ebound_meam[ein][ejn] * rij2;
  sold = snew = 1.0;
  for (int kn = 0; kn < nfull; ++kn) {
    const int k = neighbors(i, kn) & NEIGHMASK;
    if (k == j) continue;
    const double xjk = x(k, 0) - x(j, 0);
    const double yjk = x(k, 1) - x(j, 1);
    const double zjk = x(k, 2) - x(j, 2);
    const double rjk2 = xjk * xjk + yjk * yjk + zjk * zjk;
    if ((rjk2 > bo || sold == 0.0) && (rjk2 > bn || snew == 0.0)) continue;
    const double xik = x(k, 0) - x(i, 0);
    const double yik = x(k, 1) - x(i, 1);
    const double zik = x(k, 2) - x(i, 2);
    const double rik2 = xik * xik + yik * yik + zik * zik;
    const bool old_inside = sold != 0.0 && rjk2 <= bo && rik2 <= bo;
    const bool new_inside = snew != 0.0 && rjk2 <= bn && rik2 <= bn;
    if (!old_inside && !new_inside) continue;
    const double u = rik2 / rij2;
    const double v = rjk2 / rij2;
    const double a = 1.0 - (u - v) * (u - v);
    if (a <= 0.0) continue;
    const double c = (2.0 * (u + v) + a - 2.0) / a;
    const int eko = map(swap_kk_oldtype(prop, tag(k), type(k)));
    const int ekn = map(swap_kk_newtype(prop, tag(k), type(k)));
    double fo = 1.0;
    if (old_inside) {
      fo = meam_swap_screen_factor(p, eio, ejo, eko, c);
      sold *= fo;
    }
    if (new_inside) {
      if (old_inside && eio == ein && ejo == ejn && eko == ekn)
        snew *= fo;
      else
        snew *= meam_swap_screen_factor(p, ein, ejn, ekn, c);
    }
    if (sold == 0.0 && snew == 0.0) break;
  }
  double dfc;
  fc = meam_swap_dfcut((p.cutforce - Kokkos::sqrt(rij2)) * (1.0 / p.delr_meam), dfc);
}

template <class PhiView>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION void
meam_swap_pair(const MeamSwapParams &p, double scr, double fc, int elti, int eltj,
               const double *delij, double rij2, double scaleij, double sign, double &epair,
               double *densi, const PhiView &phirar, const PhiView &phirar1, const PhiView &phirar2,
               const PhiView &phirar3)
{
  epair = 0.0;
  if ((elti < 0) || (eltj < 0)) return;
  if (meam_swap_iszero(scr)) return;
  if (!(rij2 < p.cutforcesq)) return;

  const double sij = scr * fc;
  const double rij = Kokkos::sqrt(rij2);

  const int ind = p.eltind[elti][eltj];
  double pp = rij * p.rdrar;
  int kk = (int) pp;
  kk = Kokkos::min(kk, p.nrar - 2);
  pp = pp - kk;
  pp = Kokkos::min(pp, 1.0);
  const double phi =
      ((static_cast<double>(phirar3(ind, kk)) * pp + static_cast<double>(phirar2(ind, kk))) * pp +
       static_cast<double>(phirar1(ind, kk))) *
          pp +
      static_cast<double>(phirar(ind, kk));
  epair = phi * scaleij * sij;

  const double aj = rij / p.re_meam[eltj][eltj] - 1.0;
  const double ro0j = p.rho0_meam[eltj];
  const double rhoa0j = ro0j * MathSpecialKokkos::fm_exp(-p.beta0_meam[eltj] * aj) * sij;
  double rhoa1j = ro0j * MathSpecialKokkos::fm_exp(-p.beta1_meam[eltj] * aj) * sij;
  double rhoa2j = ro0j * MathSpecialKokkos::fm_exp(-p.beta2_meam[eltj] * aj) * sij;
  double rhoa3j = ro0j * MathSpecialKokkos::fm_exp(-p.beta3_meam[eltj] * aj) * sij;
  if (p.ialloy == 1) {
    rhoa1j *= p.t1_meam[eltj];
    rhoa2j *= p.t2_meam[eltj];
    rhoa3j *= p.t3_meam[eltj];
  }

  densi[0] += sign * rhoa0j;
  if (p.ialloy != 2) {
    densi[24] += sign * (p.t1_meam[eltj] * rhoa0j);
    densi[25] += sign * (p.t2_meam[eltj] * rhoa0j);
    densi[26] += sign * (p.t3_meam[eltj] * rhoa0j);
  }
  if (p.ialloy == 1) {
    densi[27] += sign * (p.t1_meam[eltj] * p.t1_meam[eltj] * rhoa0j);
    densi[28] += sign * (p.t2_meam[eltj] * p.t2_meam[eltj] * rhoa0j);
    densi[29] += sign * (p.t3_meam[eltj] * p.t3_meam[eltj] * rhoa0j);
  }
  densi[1] += sign * rhoa2j;

  const double A1j = rhoa1j / rij;
  const double A2j = rhoa2j / rij2;
  const double A3j = rhoa3j / (rij2 * rij);
  int nv2 = 0;
  int nv3 = 0;
  for (int m = 0; m < 3; m++) {
    densi[2 + m] += sign * (A1j * delij[m]);
    densi[21 + m] += sign * (rhoa3j * delij[m] / rij);
    for (int n = m; n < 3; n++) {
      densi[5 + nv2] += sign * (A2j * delij[m] * delij[n]);
      nv2++;
      for (int q = n; q < 3; q++) {
        densi[11 + nv3] += sign * (A3j * delij[m] * delij[n] * delij[q]);
        nv3++;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
static double meam_swap_embedding(const MeamSwapParams &p, const double *dens, int elti,
                                  double scaleii, int &errorflag)
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
    rho1 += arho1[m] * arho1[m];
    rho3 -= 3.0 / 5.0 * arho3b[m] * arho3b[m];
  }
  for (int m = 0; m < 6; m++) rho2 += p.v2D[m] * arho2[m] * arho2[m];
  for (int m = 0; m < 10; m++) rho3 += p.v3D[m] * arho3[m] * arho3[m];

  if (rho0 > 0.0) {
    if (p.ialloy == 1) {
      t_ave[0] = meam_swap_fdiv_zero(t_ave[0], tsq_ave[0]);
      t_ave[1] = meam_swap_fdiv_zero(t_ave[1], tsq_ave[1]);
      t_ave[2] = meam_swap_fdiv_zero(t_ave[2], tsq_ave[2]);
    } else if (p.ialloy == 2) {
      t_ave[0] = p.t1_meam[elti];
      t_ave[1] = p.t2_meam[elti];
      t_ave[2] = p.t3_meam[elti];
    } else {
      t_ave[0] /= rho0;
      t_ave[1] /= rho0;
      t_ave[2] /= rho0;
    }
  }

  double gamma = t_ave[0] * rho1 + t_ave[1] * rho2 + t_ave[2] * rho3;
  if (rho0 > 0.0) gamma /= (rho0 * rho0);

  const double Z = meam_swap_get_Zij(p.lattce_meam[elti][elti]);
  const double G = meam_swap_G_gam(p, gamma, p.ibar_meam[elti], errorflag);
  if (errorflag != 0) return 0.0;

  double shp[3];
  meam_swap_get_shpfcn(p.lattce_meam[elti][elti], p.stheta_meam[elti][elti],
                       p.ctheta_meam[elti][elti], shp);

  if (p.ibar_meam[elti] > 0) {
    double gam;
    if (p.mix_ref_t == 1)
      gam = (t_ave[0] * shp[0] + t_ave[1] * shp[1] + t_ave[2] * shp[2]) / (Z * Z);
    else
      gam = (p.t1_meam[elti] * shp[0] + p.t2_meam[elti] * shp[1] + p.t3_meam[elti] * shp[2]) /
          (Z * Z);
    meam_swap_G_gam(p, gam, p.ibar_meam[elti], errorflag);
    if (errorflag != 0) return 0.0;
  }

  double rho_bkgd;
  const double rho = rho0 * G;
  if (p.mix_ref_t == 1) {
    double Gbar = 1.0;
    if (p.ibar_meam[elti] > 0) {
      double dGbar;
      const double gam = (t_ave[0] * shp[0] + t_ave[1] * shp[1] + t_ave[2] * shp[2]) / (Z * Z);
      Gbar = meam_swap_dG_gam(p, gam, p.ibar_meam[elti], dGbar);
    }
    rho_bkgd = p.rho0_meam[elti] * Z * Gbar;
  } else if (p.bkgd_dyn == 1) {
    rho_bkgd = p.rho0_meam[elti] * Z;
  } else {
    rho_bkgd = p.rho_ref_meam[elti];
  }

  double dF;
  return meam_swap_embedding_F(p, p.A_meam[elti], p.Ec_meam[elti][elti], rho / rho_bkgd, dF) *
      scaleii;
}

/* ---------------------------------------------------------------------- */

template <class DeviceType> void MEAMKokkos<DeviceType>::fill_swap_params(MeamSwapParams &p) const
{
  p = {};
  p.cutforce = cutforce;
  p.cutforcesq = cutforcesq;
  p.delr_meam = delr_meam;
  p.rdrar = rdrar;
  p.gsmooth_factor = gsmooth_factor;
  p.nrar = nrar;
  p.ialloy = ialloy;
  p.mix_ref_t = mix_ref_t;
  p.bkgd_dyn = bkgd_dyn;
  p.emb_lin_neg = emb_lin_neg;
  for (int i = 0; i < 6; i++) p.v2D[i] = v2D[i];
  for (int i = 0; i < 10; i++) p.v3D[i] = v3D[i];
  for (int i = 0; i < neltypes; i++) {
    p.ibar_meam[i] = ibar_meam[i];
    p.A_meam[i] = A_meam[i];
    p.rho0_meam[i] = rho0_meam[i];
    p.rho_ref_meam[i] = rho_ref_meam[i];
    p.beta0_meam[i] = beta0_meam[i];
    p.beta1_meam[i] = beta1_meam[i];
    p.beta2_meam[i] = beta2_meam[i];
    p.beta3_meam[i] = beta3_meam[i];
    p.t1_meam[i] = t1_meam[i];
    p.t2_meam[i] = t2_meam[i];
    p.t3_meam[i] = t3_meam[i];
    for (int j = 0; j < neltypes; j++) {
      p.eltind[i][j] = eltind[i][j];
      p.lattce_meam[i][j] = lattce_meam[i][j];
      p.ebound_meam[i][j] = ebound_meam[i][j];
      p.Ec_meam[i][j] = Ec_meam[i][j];
      p.re_meam[i][j] = re_meam[i][j];
      p.stheta_meam[i][j] = stheta_meam[i][j];
      p.ctheta_meam[i][j] = ctheta_meam[i][j];
      for (int k = 0; k < neltypes; k++) {
        p.Cmin_meam[i][j][k] = Cmin_meam[i][j][k];
        p.Cmax_meam[i][j][k] = Cmax_meam[i][j][k];
      }
    }
  }
}

}    // namespace LAMMPS_NS

#endif
