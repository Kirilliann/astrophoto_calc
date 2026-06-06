import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

h = 6.62607015e-34
c = 2.99792458e8

radasec = 202625

ARCSEC2_TO_SR = (1/radasec)**2

def photons_per_pixel(mu_r, f_ratio, pixel_size_um, exposure_s=1.0):
    lam = 622e-9
    dlam = 138e-9

    nu = c / lam
    dnu = c * dlam / lam**2

    I_nu_arcsec2 = 3631e-26 * 10**(-0.4 * mu_r)
    I_nu = I_nu_arcsec2 / ARCSEC2_TO_SR

    s = pixel_size_um * 1e-6

    nph = (
        (np.pi / 4)
        * (s / f_ratio)**2
        * I_nu
        * dnu
        / (h * nu)
        * exposure_s
    )
    return nph


def est_snr(qnph, sig_rdn, gain, qnph_bkg=0):
    adu = qnph
    sig2_adu = (qnph + qnph_bkg + sig_rdn**2)
    snr = adu / np.sqrt(sig2_adu)
    return snr, adu, np.sqrt(sig2_adu)

def gain_eadu(gain, wde, bit=12):
    g = (wde / (2**bit)) * 10**(-gain/20)
    return g

def est_dn(gain, wde, bit=12):
    ne_max = wde / 10**(gain/20)
    geadu = gain_eadu(gain, wde, bit)
    dne = ne_max / 2**bit
    dnadu = dne / geadu
    return dnadu, dne, ne_max


def single_pixel_nph(mag_r, D, exposure_s=1.0):
    lam = 622e-9
    dlam = 138e-9

    nu = c / lam
    dnu = c * dlam / lam**2

    F_nu = 3631e-26 * 10**(-0.4 * mag_r)

    nph = (
        (np.pi / 4)
        * (D*1e-3)**2
        * F_nu
        * dnu
        / (h * nu)
        * exposure_s
    )

    return nph


def est_snr_ext(nph, fwhm_pix, gain, rdn, nph_bkg=0, size=9):
    s = fwhm_pix / (2*np.sqrt(2*np.log(2)))
    r = size // 2

    edges = np.arange(-r-0.5, r+1.5, 1.0)

    ix = 0.5 * (erf(edges[1:] / (np.sqrt(2)*s)) - erf(edges[:-1] / (np.sqrt(2)*s)))
    psf = np.outer(ix, ix)
    psf /= psf.sum()

    S = nph[:, None, None] * psf
    B = nph_bkg[:, None, None] * psf if np.ndim(nph_bkg) else nph_bkg * psf

    var = S + B + rdn**2

    snr_ap = S.sum(axis=(1,2)) / np.sqrt(var.sum(axis=(1,2)))
    snr_opt = np.sqrt(np.sum(S**2 / var, axis=(1,2)))

    max_adu = (nph[:, None, None] * psf.max()) / gain

    return snr_ap, snr_opt, max_adu



