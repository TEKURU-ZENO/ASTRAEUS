"""
ASTRAEUS Stable Entity Registry Module (Durable v2.0).

Author: Dev Mehta
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 16, 2026
Document Reference: ASTRAEUS-P5-XMATCH-002
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from pydantic import BaseModel, Field
import pandas as pd


class StableEntityRecord(BaseModel):
    """
    Persistent astronomical transient entity record.
    Decouples permanent entity identity from observational measurements,
    preserving native survey IDs, host context, and spectroscopic evidence.
    """
    astraeus_id: str = Field(..., description="Permanent internal registry ID (e.g. ASTRAEUS-EVT-00000001).")
    iau_name: Optional[str] = Field(default=None, description="Official IAU transient designation (e.g. SN 2020adv).")
    ztf_id: Optional[str] = Field(default=None, description="Native ZTF transient identifier.")
    atlas_id: Optional[str] = Field(default=None, description="Native ATLAS transient identifier (None until real association).")
    decam_id: Optional[str] = Field(default=None, description="Native DECam candidate identifier (None until real association).")
    ps1_id: Optional[str] = Field(default=None, description="Native Pan-STARRS1 source identifier (None until real association).")
    
    discovery_mjd: Optional[float] = Field(default=None, description="Discovery epoch in MJD.")
    reference_ra_deg: float = Field(..., ge=0.0, lt=360.0, description="Best-estimate equatorial RA in degrees.")
    reference_dec_deg: float = Field(..., ge=-90.0, le=90.0, description="Best-estimate equatorial Dec in degrees.")
    
    identity_status: str = Field(default="REGISTERED", description="'REGISTERED', 'SECURELY_ASSOCIATED', 'AMBIGUOUS_CONFLICT', 'ORPHAN_UNASSOCIATED'.")
    
    # Host Galaxy Context
    host_galaxy_name: Optional[str] = Field(default=None, description="Associated host galaxy designation.")
    host_galaxy_redshift: Optional[float] = Field(default=None, description="Spectroscopic or photometric redshift of host.")
    
    # Spectroscopic Evidence
    spectral_class: Optional[str] = Field(default=None, description="Normalized ASTRAEUS physical class.")
    raw_spectral_type: Optional[str] = Field(default=None, description="Raw reported classification string.")
    spectral_source: Optional[str] = Field(default=None, description="Reporting instrument/observatory (e.g. SEDM, P200).")
    evidence_tier: str = Field(default="PROVISIONAL_S2", description="'PROVISIONAL_S2', 'CERTIFIED_S1', etc.")
    
    # Operational tracking
    observations_count: int = Field(default=0, description="Total associated multi-survey observations.")
    associated_surveys: List[str] = Field(default_factory=list, description="List of surveys with confirmed observations.")


class StableEntityRegistry:
    """
    Durable, Parquet-backed entity registry enforcing permanent IDs and survey indexing.
    """

    def __init__(self):
        self._entities: Dict[str, StableEntityRecord] = {}
        self._ztf_index: Dict[str, str] = {}
        self._iau_index: Dict[str, str] = {}
        self._atlas_index: Dict[str, str] = {}
        self._decam_index: Dict[str, str] = {}
        self._ps1_index: Dict[str, str] = {}

    def register_entity(
        self,
        reference_ra_deg: float,
        reference_dec_deg: float,
        iau_name: Optional[str] = None,
        ztf_id: Optional[str] = None,
        atlas_id: Optional[str] = None,
        decam_id: Optional[str] = None,
        ps1_id: Optional[str] = None,
        discovery_mjd: Optional[float] = None,
        spectral_class: Optional[str] = None,
        raw_spectral_type: Optional[str] = None,
        spectral_source: Optional[str] = None,
        host_redshift: Optional[float] = None,
        forced_id: Optional[str] = None,
    ) -> StableEntityRecord:
        """
        Registers a new astrophysical entity with a permanent immutable ID.
        """
        # Deduplication lookup across primary aliases
        if ztf_id and ztf_id in self._ztf_index:
            return self._entities[self._ztf_index[ztf_id]]
        if iau_name and iau_name in self._iau_index:
            return self._entities[self._iau_index[iau_name]]
        if atlas_id and atlas_id in self._atlas_index:
            return self._entities[self._atlas_index[atlas_id]]

        seq = len(self._entities) + 1
        entity_id = forced_id or f"ASTRAEUS-EVT-{seq:08d}"

        surveys = []
        if ztf_id: surveys.append("ZTF")
        if atlas_id: surveys.append("ATLAS")
        if decam_id: surveys.append("DECam")
        if ps1_id: surveys.append("PS1")

        rec = StableEntityRecord(
            astraeus_id=entity_id,
            iau_name=iau_name,
            ztf_id=ztf_id,
            atlas_id=atlas_id,
            decam_id=decam_id,
            ps1_id=ps1_id,
            discovery_mjd=discovery_mjd,
            reference_ra_deg=reference_ra_deg,
            reference_dec_deg=reference_dec_deg,
            identity_status="REGISTERED",
            host_galaxy_redshift=host_redshift,
            spectral_class=spectral_class,
            raw_spectral_type=raw_spectral_type,
            spectral_source=spectral_source,
            evidence_tier="PROVISIONAL_S2",
            observations_count=0,
            associated_surveys=surveys,
        )

        self._entities[entity_id] = rec
        if ztf_id: self._ztf_index[ztf_id] = entity_id
        if iau_name: self._iau_index[iau_name] = entity_id
        if atlas_id: self._atlas_index[atlas_id] = entity_id
        if decam_id: self._decam_index[decam_id] = entity_id
        if ps1_id: self._ps1_index[ps1_id] = entity_id

        return rec

    def attach_survey_association(
        self,
        entity_id: str,
        survey: str,
        target_id: str,
        status: str = "SECURELY_ASSOCIATED",
    ) -> StableEntityRecord:
        """
        Attaches a confirmed external survey association to a registered entity.
        Only called when an empirical association decision has been validated.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Entity '{entity_id}' not found in registry.")

        if survey.upper() == "ATLAS":
            entity.atlas_id = target_id
            self._atlas_index[target_id] = entity_id
        elif survey.upper() == "DECAM":
            entity.decam_id = target_id
            self._decam_index[target_id] = entity_id
        elif survey.upper() == "PS1":
            entity.ps1_id = target_id
            self._ps1_index[target_id] = entity_id

        if survey.upper() not in entity.associated_surveys:
            entity.associated_surveys.append(survey.upper())

        entity.identity_status = status
        return entity

    def get_by_id(self, astraeus_id: str) -> Optional[StableEntityRecord]:
        return self._entities.get(astraeus_id)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [e.model_dump() for e in self._entities.values()]
        return pd.DataFrame(rows)

    def save_to_parquet(self, path: Path | str) -> None:
        """Persists registry durably to Parquet."""
        df = self.to_dataframe()
        df.to_parquet(path, index=False)

    @classmethod
    def load_from_parquet(cls, path: Path | str) -> "StableEntityRegistry":
        """Loads registry from a durable Parquet file."""
        df = pd.read_parquet(path)
        reg = cls()
        for _, row in df.iterrows():
            rec = StableEntityRecord(**row.to_dict())
            reg._entities[rec.astraeus_id] = rec
            if rec.ztf_id: reg._ztf_index[rec.ztf_id] = rec.astraeus_id
            if rec.iau_name: reg._iau_index[rec.iau_name] = rec.astraeus_id
            if rec.atlas_id: reg._atlas_index[rec.atlas_id] = rec.astraeus_id
            if rec.decam_id: reg._decam_index[rec.decam_id] = rec.astraeus_id
            if rec.ps1_id: reg._ps1_index[rec.ps1_id] = rec.astraeus_id
        return reg
