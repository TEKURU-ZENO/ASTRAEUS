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
import re
from pydantic import BaseModel, Field
import pandas as pd

SUPPORTED_SURVEYS = {"ATLAS", "DECAM", "PS1", "ZTF", "RUBIN"}


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
    rubin_id: Optional[str] = Field(default=None, description="Native Rubin/LSST alert identifier (None until real association).")
    
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
    evidence_tier: str = Field(default="PROVISIONAL_S2_UNVERIFIED_SPECTRAL_CHAIN", description="'PROVISIONAL_S2_UNVERIFIED_SPECTRAL_CHAIN', 'CERTIFIED_S1', etc.")
    
    # Operational tracking
    observations_count: int = Field(default=0, description="Total associated multi-survey observations.")
    associated_surveys: List[str] = Field(default_factory=list, description="List of surveys with confirmed observations.")


class StableEntityRegistry:
    """
    Durable, Parquet-backed entity registry enforcing permanent IDs, collision-safe
    association governance, and multi-survey cross-indexing.
    """

    def __init__(self):
        self._entities: Dict[str, StableEntityRecord] = {}
        self._ztf_index: Dict[str, str] = {}
        self._iau_index: Dict[str, str] = {}
        self._atlas_index: Dict[str, str] = {}
        self._decam_index: Dict[str, str] = {}
        self._ps1_index: Dict[str, str] = {}
        self._rubin_index: Dict[str, str] = {}
        self._max_seq: int = 0

    def register_entity(
        self,
        reference_ra_deg: float,
        reference_dec_deg: float,
        iau_name: Optional[str] = None,
        ztf_id: Optional[str] = None,
        atlas_id: Optional[str] = None,
        decam_id: Optional[str] = None,
        ps1_id: Optional[str] = None,
        rubin_id: Optional[str] = None,
        discovery_mjd: Optional[float] = None,
        spectral_class: Optional[str] = None,
        raw_spectral_type: Optional[str] = None,
        spectral_source: Optional[str] = None,
        host_redshift: Optional[float] = None,
        forced_id: Optional[str] = None,
    ) -> StableEntityRecord:
        """
        Registers a new astrophysical entity with a permanent immutable ID.
        Enforces alias deduplication, uniqueness guards on forced_id, and monotonic IDs.
        """
        # Deduplication lookup across primary aliases
        if ztf_id and ztf_id in self._ztf_index:
            return self._entities[self._ztf_index[ztf_id]]
        if iau_name and iau_name in self._iau_index:
            return self._entities[self._iau_index[iau_name]]
        if atlas_id and atlas_id in self._atlas_index:
            return self._entities[self._atlas_index[atlas_id]]
        if decam_id and decam_id in self._decam_index:
            return self._entities[self._decam_index[decam_id]]
        if ps1_id and ps1_id in self._ps1_index:
            return self._entities[self._ps1_index[ps1_id]]
        if rubin_id and rubin_id in self._rubin_index:
            return self._entities[self._rubin_index[rubin_id]]

        if forced_id:
            if forced_id in self._entities:
                raise ValueError(f"Entity ID collision: forced_id '{forced_id}' already exists in registry.")
            entity_id = forced_id
        else:
            self._max_seq += 1
            entity_id = f"ASTRAEUS-EVT-{self._max_seq:08d}"
            while entity_id in self._entities:
                self._max_seq += 1
                entity_id = f"ASTRAEUS-EVT-{self._max_seq:08d}"

        surveys = []
        if ztf_id: surveys.append("ZTF")
        if atlas_id: surveys.append("ATLAS")
        if decam_id: surveys.append("DECam")
        if ps1_id: surveys.append("PS1")
        if rubin_id: surveys.append("RUBIN")

        rec = StableEntityRecord(
            astraeus_id=entity_id,
            iau_name=iau_name,
            ztf_id=ztf_id,
            atlas_id=atlas_id,
            decam_id=decam_id,
            ps1_id=ps1_id,
            rubin_id=rubin_id,
            discovery_mjd=discovery_mjd,
            reference_ra_deg=reference_ra_deg,
            reference_dec_deg=reference_dec_deg,
            identity_status="REGISTERED",
            host_galaxy_redshift=host_redshift,
            spectral_class=spectral_class,
            raw_spectral_type=raw_spectral_type,
            spectral_source=spectral_source,
            evidence_tier="PROVISIONAL_S2_UNVERIFIED_SPECTRAL_CHAIN",
            observations_count=0,
            associated_surveys=surveys,
        )

        self._entities[entity_id] = rec
        if ztf_id: self._ztf_index[ztf_id] = entity_id
        if iau_name: self._iau_index[iau_name] = entity_id
        if atlas_id: self._atlas_index[atlas_id] = entity_id
        if decam_id: self._decam_index[decam_id] = entity_id
        if ps1_id: self._ps1_index[ps1_id] = entity_id
        if rubin_id: self._rubin_index[rubin_id] = entity_id

        return rec

    def attach_survey_association(
        self,
        entity_id: str,
        survey: str,
        target_id: str,
        status: str = "SECURELY_ASSOCIATED",
        allow_overwrite: bool = False,
    ) -> StableEntityRecord:
        """
        Attaches a confirmed external survey association to a registered entity.
        Enforces survey validation, collision prevention, and auditability.
        """
        entity = self._entities.get(entity_id)
        if entity is None:
            raise KeyError(f"Entity '{entity_id}' not found in registry.")

        survey_norm = survey.strip().upper()
        if survey_norm not in SUPPORTED_SURVEYS:
            raise ValueError(f"Unsupported survey '{survey}'. Must be one of: {sorted(SUPPORTED_SURVEYS)}")

        index_map = {
            "ATLAS": (self._atlas_index, "atlas_id"),
            "DECAM": (self._decam_index, "decam_id"),
            "PS1": (self._ps1_index, "ps1_id"),
            "ZTF": (self._ztf_index, "ztf_id"),
            "RUBIN": (self._rubin_index, "rubin_id"),
        }
        target_index, attr_name = index_map[survey_norm]

        # Check collision: External ID cannot belong to another entity
        if target_id in target_index:
            existing_owner = target_index[target_id]
            if existing_owner != entity_id and not allow_overwrite:
                raise ValueError(
                    f"Identity collision: External survey ID '{target_id}' ({survey_norm}) is already associated "
                    f"with entity '{existing_owner}'. Cannot associate with '{entity_id}'."
                )

        # Check existing entity attribute
        current_val = getattr(entity, attr_name)
        if current_val and current_val != target_id and not allow_overwrite:
            raise ValueError(
                f"Entity '{entity_id}' already has {attr_name}='{current_val}'. "
                f"Cannot overwrite with '{target_id}' unless allow_overwrite=True."
            )

        setattr(entity, attr_name, target_id)
        target_index[target_id] = entity_id

        if survey_norm not in entity.associated_surveys:
            entity.associated_surveys.append(survey_norm)

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
        """Loads registry from a durable Parquet file, reconstructing monotonic seq and indices."""
        df = pd.read_parquet(path)
        reg = cls()
        max_seq = 0
        seq_pattern = re.compile(r"^ASTRAEUS-EVT-(\d+)$")

        def _clean_val(v):
            if isinstance(v, (list, tuple)):
                return list(v)
            if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
                return list(v)
            try:
                if pd.isna(v):
                    return None
            except Exception:
                pass
            return v

        for _, row in df.iterrows():
            rec_dict = row.to_dict()
            # Clean up NaN values to None for Pydantic validation
            rec_dict = {k: _clean_val(v) for k, v in rec_dict.items()}
            rec = StableEntityRecord(**rec_dict)
            reg._entities[rec.astraeus_id] = rec

            m = seq_pattern.match(rec.astraeus_id)
            if m:
                seq_num = int(m.group(1))
                if seq_num > max_seq:
                    max_seq = seq_num

            if rec.ztf_id: reg._ztf_index[rec.ztf_id] = rec.astraeus_id
            if rec.iau_name: reg._iau_index[rec.iau_name] = rec.astraeus_id
            if rec.atlas_id: reg._atlas_index[rec.atlas_id] = rec.astraeus_id
            if rec.decam_id: reg._decam_index[rec.decam_id] = rec.astraeus_id
            if rec.ps1_id: reg._ps1_index[rec.ps1_id] = rec.astraeus_id
            if getattr(rec, "rubin_id", None): reg._rubin_index[rec.rubin_id] = rec.astraeus_id

        reg._max_seq = max_seq
        return reg
