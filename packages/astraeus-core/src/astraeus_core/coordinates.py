"""
ASTRAEUS Coordinates Module.
Encodes celestial coordinates, reference frames, and HEALPix spatial partitioning.
"""

from typing import Optional, Dict, Tuple, Literal
from pydantic import BaseModel, Field, field_validator
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix

from astraeus_core.exceptions import CoordinateValidationError


class CelestialCoordinates(BaseModel):
    """
    Standard celestial coordinate representation with astrometric uncertainties and HEALPix partitioning.
    """
    ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Right Ascension in decimal degrees [0, 360).")
    dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Declination in decimal degrees [-90, +90].")
    reference_frame: str = Field(default="ICRS", description="Coordinate reference frame (e.g. 'ICRS', 'FK5').")
    epoch_jyr: float = Field(default=2016.0, description="Equinox / epoch in Julian years (e.g. 2016.0 for Gaia DR3).")
    pm_ra_cosdec_mas_yr: Optional[float] = Field(default=None, description="Proper motion in RA * cos(Dec) in mas/yr.")
    pm_dec_mas_yr: Optional[float] = Field(default=None, description="Proper motion in Dec in mas/yr.")
    parallax_mas: Optional[float] = Field(default=None, description="Parallax in milliarcseconds.")
    uncertainty_mas: Dict[str, float] = Field(
        default_factory=dict,
        description="Astrometric coordinate uncertainties in milliarcseconds: {'ra': float, 'dec': float}."
    )
    healpix_index_nside32: Optional[int] = Field(
        default=None,
        description="Spatial HEALPix index computed at N_side=32."
    )

    @field_validator("ra_deg")
    @classmethod
    def validate_ra(cls, v: float) -> float:
        if not (0.0 <= v < 360.0):
            raise CoordinateValidationError(f"Right Ascension {v} out of bounds [0.0, 360.0).")
        return v

    @field_validator("dec_deg")
    @classmethod
    def validate_dec(cls, v: float) -> float:
        if not (-90.0 <= v <= 90.0):
            raise CoordinateValidationError(f"Declination {v} out of bounds [-90.0, +90.0].")
        return v

    def to_skycoord(self) -> SkyCoord:
        """Converts to Astropy SkyCoord representation."""
        kwargs = {
            "ra": self.ra_deg * u.deg,
            "dec": self.dec_deg * u.deg,
            "frame": self.reference_frame.lower(),
        }
        if self.pm_ra_cosdec_mas_yr is not None and self.pm_dec_mas_yr is not None:
            kwargs["pm_ra_cosdec"] = self.pm_ra_cosdec_mas_yr * (u.mas / u.yr)
            kwargs["pm_dec"] = self.pm_dec_mas_yr * (u.mas / u.yr)
        if self.parallax_mas is not None and self.parallax_mas > 0:
            kwargs["distance"] = (1000.0 / self.parallax_mas) * u.pc
        return SkyCoord(**kwargs)

    def to_galactic(self) -> Tuple[float, float]:
        """Returns Galactic longitude (l) and latitude (b) in degrees."""
        coord = self.to_skycoord()
        gal = coord.galactic
        return gal.l.deg, gal.b.deg

    def compute_healpix_index(self, nside: int = 32, order: Literal["nested", "ring"] = "nested") -> int:
        """
        Computes HEALPix spatial pixel index for given N_side.
        N_side = 32 corresponds to 12,288 equal-area pixels (~1.83 deg diameter).
        """
        hp = HEALPix(nside=nside, order=order, frame=self.reference_frame.lower())
        coord = self.to_skycoord()
        pix = int(hp.skycoord_to_healpix(coord))
        if nside == 32:
            self.healpix_index_nside32 = pix
        return pix

    def get_sky_partition(
        self,
        nside: int = 32,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Literal["train", "val", "test"]:
        """
        Deterministically maps celestial coordinate to a dataset partition based on spatial HEALPix pixel.
        Guarantees zero cross-split leakage.
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Partition ratios must sum to 1.0"
        pix = self.compute_healpix_index(nside=nside, order="nested")
        
        # Deterministic hash of pixel + seed
        import hashlib
        h = int(hashlib.sha256(f"{pix}_{seed}".encode()).hexdigest(), 16)
        normalized = (h % 1000000) / 1000000.0
        
        if normalized < train_ratio:
            return "train"
        elif normalized < (train_ratio + val_ratio):
            return "val"
        else:
            return "test"
