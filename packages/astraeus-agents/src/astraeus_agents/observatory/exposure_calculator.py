"""
ASTRAEUS Observatory Intelligence - Physics-Based Exposure Time & SNR Calculator (WS7.5).
Calculates signal-to-noise ratios, limiting magnitudes, photometric uncertainty propagation,
and required integration times across Optical, NIR, Space UV/X-ray, and Radio regimes.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional
from dataclasses import dataclass
from scipy.stats import norm

from astraeus_agents.observatory.facility_registry import (
    InstrumentProfile,
    InstrumentFilter,
    WavelengthRegime,
)


@dataclass
class ExposureEstimate:
    """Photometric and statistical detection assessment for an exposure."""
    facility_id: str
    instrument_id: str
    filter_id: str
    target_apparent_mag: float
    effective_mag: float
    airmass: float
    exposure_time_sec: float
    snr: float
    photometric_uncertainty_mag: float
    limiting_mag_5sigma: float
    detection_probability: float
    is_detected_5sigma: bool


class ExposureCalculator:
    """Physics-based signal-to-noise and exposure time calculator."""

    @classmethod
    def compute_snr(
        cls,
        instrument: InstrumentProfile,
        filter_obj: InstrumentFilter,
        target_apparent_mag: float,
        airmass: float,
        seeing_fwhm_arcsec: float,
        exposure_time_sec: float,
        cloud_extinction_mag: float = 0.0,
    ) -> ExposureEstimate:
        """
        Computes accurate SNR for an astronomical point source given instrument,
        filter throughput, atmospheric extinction, seeing, and sky background.
        """
        t = max(instrument.min_exposure_sec, min(exposure_time_sec, instrument.max_exposure_sec))

        # Handle regimes
        if instrument.regime in [WavelengthRegime.OPTICAL, WavelengthRegime.NIR, WavelengthRegime.UV]:
            zp = filter_obj.zero_point_ab_mag or 26.0
            sky_mu = filter_obj.sky_brightness_mag_arcsec2 or 21.0
            kappa = filter_obj.extinction_coeff

            # Atmospheric and cloud extinction
            delta_ext = kappa * max(0.0, airmass - 1.0) + cloud_extinction_mag
            m_eff = target_apparent_mag + delta_ext

            # Source count rate in electrons / sec
            source_e_s = 10.0 ** (0.4 * (zp - m_eff))

            # Aperture photometry size (1.2 * FWHM diameter)
            aper_diam_arcsec = max(0.5, 1.2 * seeing_fwhm_arcsec)
            aper_area_arcsec2 = math.pi * ((aper_diam_arcsec / 2.0) ** 2)
            n_pix = aper_area_arcsec2 / max(1e-4, instrument.pixel_scale_arcsec ** 2)

            # Sky count rate in electrons / sec within aperture
            sky_e_s = (10.0 ** (0.4 * (zp - sky_mu))) * aper_area_arcsec2

            # Total counts and noise variance
            signal = source_e_s * t
            noise_var = (
                signal +
                sky_e_s * t +
                n_pix * (instrument.dark_current_e_per_s * t) +
                n_pix * (instrument.read_noise_electrons ** 2)
            )
            noise = math.sqrt(max(1.0, noise_var))
            snr = signal / noise

            # 5-sigma limiting magnitude for this exposure duration
            # 5 * noise_background / t = 10^(0.4 * (zp - m_lim))
            bkg_noise = math.sqrt(sky_e_s * t + n_pix * instrument.read_noise_electrons ** 2)
            if bkg_noise > 0 and t > 0:
                lim_flux = 5.0 * bkg_noise / t
                lim_mag = zp - 2.5 * math.log10(max(1e-10, lim_flux)) - delta_ext
            else:
                lim_mag = zp

        elif instrument.regime == WavelengthRegime.XRAY:
            # X-ray Poisson count rate: ~ 10^-11 erg/cm2/s ~ 0.2 cts/s for Swift-XRT
            # Map apparent magnitude proxy to energy flux: m ~ 20 -> 1e-13 erg/cm2/s
            flux_erg = 10.0 ** (-0.4 * (target_apparent_mag - 12.5) - 10.0)
            cts_per_s = flux_erg * 1e11 * (instrument.aperture_m / 0.5) ** 2
            bkg_cts_s = 0.001
            signal = cts_per_s * t
            noise = math.sqrt(signal + bkg_cts_s * t + 0.1)
            snr = signal / max(0.1, noise)
            m_eff = target_apparent_mag
            lim_mag = 23.0 + 1.25 * math.log10(t / 1000.0)

        else:  # Radio regime
            # Radio continuum thermal noise
            # S_lim ~ 50 uJy for MeerKAT / VLA in 1 hour
            signal_proxy = 10.0 ** (-0.4 * (target_apparent_mag - 15.0))
            thermal_noise = 0.005 / math.sqrt(t / 3600.0)
            snr = signal_proxy / max(1e-6, thermal_noise)
            m_eff = target_apparent_mag
            lim_mag = 20.0 + 1.25 * math.log10(t / 3600.0)

        # Photometric magnitude uncertainty: sigma_m = 1.0857 / SNR
        if snr > 0.1:
            phot_err = float(min(1.5, 1.0857 / snr))
        else:
            phot_err = 2.5

        # Detection probability (Gaussian tail probability above 5-sigma)
        det_prob = float(norm.cdf(snr - 5.0))

        return ExposureEstimate(
            facility_id=instrument.instrument_id,
            instrument_id=instrument.instrument_id,
            filter_id=filter_obj.filter_id,
            target_apparent_mag=round(target_apparent_mag, 3),
            effective_mag=round(m_eff, 3),
            airmass=round(airmass, 2),
            exposure_time_sec=round(t, 1),
            snr=round(snr, 2),
            photometric_uncertainty_mag=round(phot_err, 4),
            limiting_mag_5sigma=round(lim_mag, 2),
            detection_probability=round(det_prob, 4),
            is_detected_5sigma=(snr >= 5.0),
        )

    @classmethod
    def calculate_required_exposure_sec(
        cls,
        instrument: InstrumentProfile,
        filter_obj: InstrumentFilter,
        target_apparent_mag: float,
        airmass: float,
        seeing_fwhm_arcsec: float,
        target_snr: float = 10.0,
    ) -> float:
        """
        Calculates minimum exposure time (sec) required to reach target_snr.
        Bounded by [instrument.min_exposure_sec, instrument.max_exposure_sec].
        """
        # Binary search / Newton iteration between min and max exposure
        low = instrument.min_exposure_sec
        high = instrument.max_exposure_sec

        for _ in range(12):
            mid = 0.5 * (low + high)
            est = cls.compute_snr(
                instrument=instrument,
                filter_obj=filter_obj,
                target_apparent_mag=target_apparent_mag,
                airmass=airmass,
                seeing_fwhm_arcsec=seeing_fwhm_arcsec,
                exposure_time_sec=mid,
            )
            if est.snr < target_snr:
                low = mid
            else:
                high = mid

        return round(high, 1)
