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



gain0d1dB = 500 #np.arange(150, 300)
gaindB = 0.1 * gain0d1dB


D = 40#*np.sqrt(1 - 0.75)
F = 186
FD = F/D
s = 3.76

lam = 6200

fwhm_diff = radasec * 1.22 * lam * 1e-7 / D
s_asex = radasec * s * 1e-3 / F
print(fwhm_diff, s_asex)
psf_fwhm = fwhm_diff # asec

sig_rdn = 1
wde = 73e3
bit = 12
q = 0.91 * 0.65

mu_r = np.arange(15, 22, 0.1) # mu  mag/asec2
t = 0.5 # sec


nph = photons_per_pixel(mu_r, FD, s, t)

qnph = q * nph
g = gain_eadu(gaindB, wde, bit=bit)
snr_val, adu_val, sig_adu_val = est_snr(qnph, sig_rdn, g)
dnadu_val, dne_val, ne_max_val = est_dn(gaindB, wde, bit=bit)

discret = sig_adu_val / dnadu_val

plt.plot(mu_r, discret, label=r'$\sigma$/$\Delta N$')
plt.plot(mu_r, snr_val, label='SNR')
plt.axhline(3, linestyle='--', color='black')
plt.legend()
plt.yscale('log')
plt.show()

plt.plot(mu_r, adu_val, label='BKG level (ADU)')
plt.legend()
plt.show()


#plt.plot(mu_r, adu_val)
#plt.show()

magr_vals = np.arange(3, 20, 0.1)

qnph_bkg = q*photons_per_pixel(20, FD, s, t)
print('qnph_bkg: ', qnph_bkg)
nph_single_pixel = single_pixel_nph(magr_vals, D, t)
#plt.plot(magr_vals, nph_single_pixel)
#plt.yscale('log')
#plt.show()

qnph = q * nph_single_pixel
g = gain_eadu(gaindB, wde, bit=bit)
snr_val, adu_val, sig_adu_val = est_snr(qnph, sig_rdn, g, qnph_bkg=qnph_bkg)

plt.plot(magr_vals, snr_val, label='Point-source (single pixel)')

limit = 2**bit
mask = adu_val > limit
if np.any(mask):
    idx = np.where(mask)[0][np.argmin(np.abs(adu_val[mask] - limit))]
    plt.plot(magr_vals[idx], snr_val[idx], 'ko')

snr_aper0, snr_opt0, adu_max0 = est_snr_ext(qnph, 5, g, sig_rdn, 0, size=9)
snr_aper, snr_opt, adu_max = est_snr_ext(qnph, 5, g, sig_rdn, qnph_bkg, size=9)

plt.plot(magr_vals, snr_aper, label='PSF (aper snr)')
#plt.plot(magr_vals, snr_aper0, label='PSF (aper snr, bkg=0)')
plt.plot(magr_vals, snr_opt, label='PSF (opt. snr)')

mask = adu_max > limit
if np.any(mask):
    idx = np.where(mask)[0][np.argmin(np.abs(adu_max[mask] - limit))]
    plt.plot(magr_vals[idx], snr_aper[idx], 'ro')

plt.legend()

plt.yscale('log')
plt.axhline(3, linestyle='--', color='black')
plt.xlabel('mag')
plt.ylabel('SNR')
plt.show()



