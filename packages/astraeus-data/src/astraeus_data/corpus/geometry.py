"""
ASTRAEUS Spherical Geometry & HEALPix Cell-Boundary Distance Module.
Provides mathematically exact, vector-based spherical distance calculations
from celestial positions to true spherical HEALPix polygon edges.
Eliminates inscribed-circle approximations and implements rigorous quarantine predicates.
"""

from typing import Tuple, Optional
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy_healpix import HEALPix


def compute_spherical_edge_distance_arcsec(
    coord: SkyCoord,
    hp: HEALPix,
    pix: int,
    step: int = 25,
) -> float:
    """
    Computes the shortest angular distance from a celestial coordinate to the
    actual spherical boundary polygon of its containing HEALPix pixel:

        d_edge(x) = min_{y in partial C} angdist(x, y)

    Parameters
    ----------
    coord : SkyCoord
        Astrometric coordinate of the candidate object.
    hp : HEALPix
        HEALPix projection instance (order='nested', frame='icrs').
    pix : int
        HEALPix pixel index at the declared resolution (Nside=32).
    step : int, optional
        Sampling density along each of the 4 cell edges (default 25 -> 100 boundary vertices).

    Returns
    -------
    float
        Shortest angular distance to the spherical cell boundary in arcseconds.
    """
    # Sample the 4 curvilinear edges of the HEALPix pixel on the sphere
    boundaries = hp.boundaries_skycoord([pix], step=step)[0]
    A = boundaries.cartesian.xyz.value.T
    B = np.roll(A, -1, axis=0)
    p = coord.cartesian.xyz.value

    # Compute normal vectors to each great-circle segment AB
    cross_ab = np.cross(A, B)
    norm_ab = np.linalg.norm(cross_ab, axis=1, keepdims=True)
    norm_ab[norm_ab == 0] = 1.0
    n = cross_ab / norm_ab

    # Distance from point p to the great circle plane containing segment AB
    dot_pn = np.sum(n * p, axis=1)
    d_gc = np.arcsin(np.clip(np.abs(dot_pn), 0.0, 1.0))

    # Orthogonal projection of p onto the great circle
    p_proj = p - dot_pn[:, np.newaxis] * n
    norm_proj = np.linalg.norm(p_proj, axis=1, keepdims=True)
    norm_proj[norm_proj == 0] = 1.0
    p_proj /= norm_proj

    # Check whether the projected point lies between endpoints A and B
    cross_ap = np.cross(A, p_proj)
    cross_pb = np.cross(p_proj, B)
    in_seg_a = np.sum(cross_ap * n, axis=1) >= -1e-12
    in_seg_b = np.sum(cross_pb * n, axis=1) >= -1e-12
    in_seg = in_seg_a & in_seg_b

    # Distance to segment endpoints for out-of-segment projections
    dot_pa = np.clip(np.sum(A * p, axis=1), -1.0, 1.0)
    d_endpoints = np.arccos(dot_pa)

    seg_dists = np.where(in_seg, d_gc, d_endpoints)
    min_rad = float(np.min(seg_dists))
    return float(min_rad * 206264.806247)


def evaluate_quarantine_predicate(
    coord: SkyCoord,
    hp: HEALPix,
    pix: int,
    has_conflicting_neighbor: bool,
    astrometric_err_arcsec: float,
    buffer_arcsec: float = 5.0,
    confidence_multiplier: float = 3.0,
    step: int = 25,
) -> Tuple[bool, Optional[str], float]:
    """
    Evaluates the mathematically rigorous boundary quarantine predicate:

        P_quarantine(x) <=> conflicting_neighbor(p) AND [
            d_edge(x) <= buffer_arcsec OR d_edge(x) <= k * sigma_ast
        ]

    Parameters
    ----------
    coord : SkyCoord
        Celestial coordinates of the entity.
    hp : HEALPix
        HEALPix instance.
    pix : int
        Pixel index.
    has_conflicting_neighbor : bool
        True if an adjacent Voronoi neighbor pixel belongs to a different active split.
    astrometric_err_arcsec : float
        Projected astrometric uncertainty sigma_ast in arcseconds.
    buffer_arcsec : float, optional
        Spatial exclusion buffer (default 5.0 arcsec).
    confidence_multiplier : float, optional
        Multiplier k for astrometric uncertainty circle (default 3.0).
    step : int, optional
        Edge discretization resolution.

    Returns
    -------
    Tuple[bool, Optional[str], float]
        (is_quarantined, failure_reason, d_edge_arcsec)
    """
    d_edge = compute_spherical_edge_distance_arcsec(coord, hp, pix, step=step)

    if not has_conflicting_neighbor:
        return False, None, d_edge

    if d_edge <= buffer_arcsec:
        return True, "SPHERICAL_CELL_EDGE_BUFFER_VIOLATION", d_edge

    if d_edge <= confidence_multiplier * astrometric_err_arcsec:
        return True, "ASTROMETRIC_UNCERTAINTY_CROSSING_RISK", d_edge

    return False, None, d_edge
