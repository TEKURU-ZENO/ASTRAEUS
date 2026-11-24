"""
ASTRAEUS Phase 7 Data Access Policy & Quarantine Firewall.
Strictly preserves the Phase 5 Gate 3 Scientific Quarantine Covenant:
- 1,808 frozen HEALPix pixels (NSIDE=32, NESTED, ICRS)
- 586 quarantined transient candidate objects in candidates_10k.parquet
- Prevents autonomous agents from querying or ingesting quarantined ground truth.

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

from __future__ import annotations

import os
import json
from typing import Set, Optional, Dict, Any
from astropy_healpix import HEALPix
from astropy.coordinates import SkyCoord
import astropy.units as u


class QuarantineViolationError(PermissionError):
    """Raised when an agent, pipeline, or query attempts to access Gate 3 quarantined data."""
    pass


class QuarantineFirewall:
    """Runtime firewall guarding the Gate 3 zero-touch quarantine covenant."""

    _QUARANTINED_PIXELS: Optional[Set[int]] = None
    _HP: Optional[HEALPix] = None
    _CONFIG_PATH = os.path.join("configs", "data_splits_healpix.json")

    @classmethod
    def _initialize(cls) -> None:
        if cls._QUARANTINED_PIXELS is None:
            if not os.path.exists(cls._CONFIG_PATH):
                raise FileNotFoundError(f"Missing required quarantine configuration: {cls._CONFIG_PATH}")

            with open(cls._CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            test_pixels = set(data["splits"]["test"])
            if len(test_pixels) != 1808:
                raise ValueError(f"Corrupted quarantine config: expected 1,808 pixels, found {len(test_pixels)}")

            cls._QUARANTINED_PIXELS = test_pixels
            cls._HP = HEALPix(nside=32, order="nested", frame="icrs")

    @classmethod
    def get_healpix(cls) -> HEALPix:
        cls._initialize()
        return cls._HP

    @classmethod
    def get_quarantined_pixels(cls) -> Set[int]:
        cls._initialize()
        return set(cls._QUARANTINED_PIXELS)

    @classmethod
    def get_quarantined_pixel_count(cls) -> int:
        cls._initialize()
        return len(cls._QUARANTINED_PIXELS)

    @classmethod
    def compute_healpix_nside32(cls, ra_deg: float, dec_deg: float) -> int:
        cls._initialize()
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        return int(cls._HP.skycoord_to_healpix(coord))

    @classmethod
    def is_quarantined_pixel(cls, pixel: int) -> bool:
        cls._initialize()
        return pixel in cls._QUARANTINED_PIXELS

    @classmethod
    def is_quarantined_coordinate(cls, ra_deg: float, dec_deg: float) -> bool:
        pixel = cls.compute_healpix_nside32(ra_deg, dec_deg)
        return cls.is_quarantined_pixel(pixel)

    @classmethod
    def assert_not_quarantined(
        cls,
        ra_deg: float,
        dec_deg: float,
        identifier: Optional[str] = None,
        context: str = "agent_query",
    ) -> int:
        """
        Asserts that celestial coordinates do not lie inside the Gate 3 quarantine footprint.
        Returns the computed HEALPix index if valid, or raises QuarantineViolationError.
        """
        pixel = cls.compute_healpix_nside32(ra_deg, dec_deg)
        if cls.is_quarantined_pixel(pixel):
            target_desc = f"'{identifier}' at ({ra_deg:.4f}, {dec_deg:.4f})" if identifier else f"({ra_deg:.4f}, {dec_deg:.4f})"
            raise QuarantineViolationError(
                f"GATE 3 QUARANTINE VIOLATION [{context}]: Target {target_desc} maps to "
                f"HEALPix pixel {pixel}, which belongs to the 1,808 quarantined Gate 3 test partition. "
                f"Autonomous agent access is strictly prohibited by scientific covenant."
            )
        return pixel

    @classmethod
    def assert_safe_file_access(cls, file_path: str) -> None:
        """Blocks any runtime file open on quarantined candidate datasets."""
        normalized = os.path.normpath(file_path).lower()
        if "candidates_10k.parquet" in normalized:
            raise QuarantineViolationError(
                f"GATE 3 QUARANTINE VIOLATION: Direct file read of '{file_path}' blocked. "
                f"Observational catalog contains sequestered Gate 3 test objects."
            )


# Auto-initialize firewall if config exists
if os.path.exists(QuarantineFirewall._CONFIG_PATH):
    try:
        QuarantineFirewall._initialize()
    except Exception:
        pass

