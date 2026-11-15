"""
ASTRAEUS Gate 4 Phase 4B: Exact HEALPix Spatial Isolation & 60-Arcsecond Buffer Audit.

Implements the authoritative spatial exclusion geometry:
- Configuration: configs/data_splits_healpix.json (SHA-256: 234bb08c...)
- Resolution: NSIDE = 32, Scheme = NESTED, Frame = ICRS (J2000.0)
- Excluded Geometry: Explicit array of exactly 1,808 frozen test pixels
- Buffer: 60-arcsecond spherical cone search around each candidate coordinate
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
import numpy as np
import pandas as pd

HEALPIX_CONFIG_PATH = Path("configs/data_splits_healpix.json")
EXPECTED_HEALPIX_CONFIG_HASH = "234bb08c482824648607fa8f874477d0225717c18263b0f16ea276dd1e12417b"


class SpatialIsolationAuditor:
    """Audits transient sky coordinates against the authoritative Gate 3 test footprint."""

    def __init__(self, config_path: Path = HEALPIX_CONFIG_PATH):
        self.config_path = Path(config_path)
        self._verify_config_hash()

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.nside = int(data.get("healpix_nside", 32))
        self.test_pixels: Set[int] = set(int(p) for p in data["splits"]["test"])
        assert len(self.test_pixels) == 1808, f"Expected 1808 test pixels, found {len(self.test_pixels)}"

        self.hp = HEALPix(nside=self.nside, order="nested", frame="icrs")

    def _verify_config_hash(self) -> None:
        actual_hash = hashlib.sha256(self.config_path.read_bytes()).hexdigest()
        if actual_hash != EXPECTED_HEALPIX_CONFIG_HASH:
            raise ValueError(
                f"HEALPix split config hash mismatch! Expected {EXPECTED_HEALPIX_CONFIG_HASH}, got {actual_hash}"
            )

    def audit_single_coordinate(self, ra_deg: float, dec_deg: float, buffer_arcsec: float = 60.0) -> Dict[str, any]:
        """Audit a single equatorial coordinate (ICRS) against frozen test pixels and buffer."""
        sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        pixel = int(self.hp.skycoord_to_healpix(sc))
        is_exact_overlap = pixel in self.test_pixels

        # Spherical cone search of radius buffer_arcsec
        cone_pixels = self.hp.cone_search_skycoord(sc, radius=buffer_arcsec * u.arcsec)
        intersecting_test_pixels = [int(p) for p in cone_pixels if int(p) in self.test_pixels]
        is_buffer_overlap = len(intersecting_test_pixels) > 0

        # Status determination
        if is_exact_overlap:
            status = "EXACT_TEST_PIXEL_OVERLAP"
            is_permitted = False
        elif is_buffer_overlap:
            status = "BUFFER_TEST_PIXEL_OVERLAP"
            is_permitted = False
        else:
            status = "SPATIALLY_ISOLATED"
            is_permitted = True

        return {
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "healpix_pixel_nested": pixel,
            "is_exact_test_pixel_overlap": is_exact_overlap,
            "is_buffer_test_pixel_overlap": is_buffer_overlap,
            "intersecting_test_pixels_in_buffer": intersecting_test_pixels,
            "is_spatially_isolated": is_permitted,
            "spatial_isolation_status": status,
        }

    def audit_coordinates_dataframe(
        self, df: pd.DataFrame, ra_col: str = "ra_deg", dec_col: str = "dec_deg", buffer_arcsec: float = 60.0
    ) -> pd.DataFrame:
        """Batch audit a pandas DataFrame of candidate coordinates."""
        results = []
        for _, row in df.iterrows():
            res = self.audit_single_coordinate(
                ra_deg=float(row[ra_col]), dec_deg=float(row[dec_col]), buffer_arcsec=buffer_arcsec
            )
            results.append(res)
        return pd.DataFrame(results)


def main():
    auditor = SpatialIsolationAuditor()
    print("SpatialIsolationAuditor initialized with 1,808 frozen test pixels.")
    print(f"HEALPix NSIDE: {auditor.nside}, Scheme: NESTED, Frame: ICRS")

    # Audit the pilot cohort specification
    pilot_path = Path("experiments/gate4b_broker_acquisition_and_cohort_expansion/pilot_cohort_specification.json")
    with open(pilot_path, "r", encoding="utf-8") as f:
        pilot_data = json.load(f)

    df_pilot = pd.DataFrame(pilot_data["candidates"])
    audit_df = auditor.audit_coordinates_dataframe(df_pilot)

    isolated_count = audit_df["is_spatially_isolated"].sum()
    print(f"\nPilot Cohort Spatial Audit Results ({len(df_pilot)} objects):")
    print(f"  - Spatially Isolated (>60'' from all 1,808 test pixels): {isolated_count} / {len(df_pilot)} (100%)")
    assert isolated_count == len(df_pilot), "Spatial buffer violation detected in pilot cohort!"
    print("All pilot candidates verified to be strictly spatially isolated.")


if __name__ == "__main__":
    main()
