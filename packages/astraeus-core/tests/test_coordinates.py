import pytest
from astraeus_core.coordinates import CelestialCoordinates
from astraeus_core.exceptions import CoordinateValidationError


def test_valid_coordinates():
    coord = CelestialCoordinates(
        ra_deg=184.283910,
        dec_deg=-5.819203,
        reference_frame="ICRS",
        epoch_jyr=2016.0,
        uncertainty_mas={"ra": 0.12, "dec": 0.09},
    )
    assert coord.ra_deg == 184.283910
    assert coord.dec_deg == -5.819203
    assert coord.reference_frame == "ICRS"


def test_invalid_coordinates():
    with pytest.raises((CoordinateValidationError, ValueError)):
        CelestialCoordinates(ra_deg=365.0, dec_deg=0.0)

    with pytest.raises((CoordinateValidationError, ValueError)):
        CelestialCoordinates(ra_deg=100.0, dec_deg=-95.0)


def test_galactic_transformation():
    coord = CelestialCoordinates(
        ra_deg=0.0,
        dec_deg=0.0,
        reference_frame="ICRS",
    )
    l, b = coord.to_galactic()
    assert 0.0 <= l <= 360.0
    assert -90.0 <= b <= 90.0


def test_healpix_partitioning_nside32():
    coord = CelestialCoordinates(
        ra_deg=150.0,
        dec_deg=2.5,
    )
    pix = coord.compute_healpix_index(nside=32)
    assert 0 <= pix < 12 * (32 ** 2)  # 12 * 1024 = 12288
    assert coord.healpix_index_nside32 == pix

    # Deterministic partitioning
    part1 = coord.get_sky_partition(nside=32, seed=42)
    part2 = coord.get_sky_partition(nside=32, seed=42)
    assert part1 == part2
    assert part1 in {"train", "val", "test"}
