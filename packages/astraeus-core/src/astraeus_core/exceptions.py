"""
ASTRAEUS Core Exceptions Module.
Defines foundational epistemic and provenance domain errors.
"""

class AstraeusError(Exception):
    """Base exception for all ASTRAEUS errors."""
    pass


class EpistemicViolationError(AstraeusError):
    """
    Raised when an epistemic stratification rule is violated,
    such as attempting to present a model inference or simulation as directly observed data.
    """
    pass


class ProvenanceMissingError(AstraeusError):
    """
    Raised when a data entity or model result lacks required IVOA Provenance metadata.
    """
    pass


class DataLeakageError(AstraeusError):
    """
    Raised when spatial (HEALPix) or temporal cross-split boundaries are violated.
    """
    pass


class CoordinateValidationError(AstraeusError):
    """
    Raised when celestial coordinates fall outside physical bounds or lack valid reference frames.
    """
    pass
