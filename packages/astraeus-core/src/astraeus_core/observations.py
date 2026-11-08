"""
ASTRAEUS Observations Module.
Data structures for astronomical observations, photometric measurements, and light curves.
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import numpy as np

from astraeus_core.epistemic import EpistemicTier


class CalibratedMeasurement(BaseModel):
    """
    An individual physical measurement with value, standard error, unit, and passband.
    """
    quantity_name: str = Field(..., description="Name of measured physical property (e.g. 'flux_density', 'magnitude').")
    value: float = Field(..., description="Measured central scalar value.")
    uncertainty: float = Field(..., ge=0.0, description="1-sigma standard measurement error.")
    unit: str = Field(..., description="Physical unit string (e.g. 'mag_AB', 'mJy', 'km/s').")
    filter_band: Optional[str] = Field(default=None, description="Telescope filter passband (e.g. 'ZTF_g', 'Gaia_G').")
    quality_flags: int = Field(default=0, description="Bitmask of instrumental and quality flags.")
    epistemic_tier: EpistemicTier = Field(
        default=EpistemicTier.OBSERVED,
        description="Epistemic status of this measurement."
    )


class ObservationRecord(BaseModel):
    """
    Metadata and calibrated readout of a single astronomical exposure.
    Follows IVOA ObsCore conventions.
    """
    observation_id: str = Field(..., description="Unique observation ID (e.g. 'astraeus:obs:ztf:202603124819').")
    survey: str = Field(..., description="Survey name (e.g. 'ZTF', 'TESS', 'Gaia', 'Rubin').")
    instrument: str = Field(..., description="Instrument identifier (e.g. 'ZTF-Camera', 'TESS-CCD1').")
    filter_band: str = Field(..., description="Filter bandpass (e.g. 'ZTF_g', 'ZTF_r').")
    mjd: float = Field(..., description="Modified Julian Date of exposure midpoint.")
    exposure_time_sec: Optional[float] = Field(default=None, ge=0.0)
    calibrated_flux: CalibratedMeasurement = Field(..., description="Primary calibrated flux measurement.")
    quality_flags: int = Field(default=0, description="0 indicates clean observation; >0 indicates artifact/warning.")
    ivoa_provenance_id: str = Field(..., description="Link to IVOA Provenance activity generating this observation.")


class PhotometricTimeSeries(BaseModel):
    """
    Multi-epoch photometric time series (light curve) across multiple passbands.
    """
    times_mjd: List[float] = Field(..., description="Array of observation epochs in MJD.")
    flux_values: List[float] = Field(..., description="Observed flux or magnitude values.")
    flux_uncertainties: List[float] = Field(..., description="Photometric measurement errors.")
    filter_bands: List[str] = Field(..., description="Passband for each observation epoch.")
    quality_flags: List[int] = Field(default_factory=list, description="Quality bitmasks.")

    @field_validator("flux_values")
    @classmethod
    def validate_lengths(cls, v: List[float], info) -> List[float]:
        times = info.data.get("times_mjd")
        if times is not None and len(v) != len(times):
            raise ValueError(f"Length of flux_values ({len(v)}) must match times_mjd ({len(times)}).")
        return v

    def filter_clean(self) -> "PhotometricTimeSeries":
        """Returns a new PhotometricTimeSeries retaining only clean epochs (quality_flag == 0)."""
        if not self.quality_flags:
            return self
        clean_indices = [i for i, flag in enumerate(self.quality_flags) if flag == 0]
        return PhotometricTimeSeries(
            times_mjd=[self.times_mjd[i] for i in clean_indices],
            flux_values=[self.flux_values[i] for i in clean_indices],
            flux_uncertainties=[self.flux_uncertainties[i] for i in clean_indices],
            filter_bands=[self.filter_bands[i] for i in clean_indices],
            quality_flags=[0] * len(clean_indices),
        )

    def sort_chronological(self) -> "PhotometricTimeSeries":
        """Sorts epochs chronologically by MJD."""
        sorted_indices = np.argsort(self.times_mjd)
        return PhotometricTimeSeries(
            times_mjd=[self.times_mjd[i] for i in sorted_indices],
            flux_values=[self.flux_values[i] for i in sorted_indices],
            flux_uncertainties=[self.flux_uncertainties[i] for i in sorted_indices],
            filter_bands=[self.filter_bands[i] for i in sorted_indices],
            quality_flags=[self.quality_flags[i] for i in sorted_indices] if self.quality_flags else [],
        )
