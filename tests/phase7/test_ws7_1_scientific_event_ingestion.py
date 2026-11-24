"""
ASTRAEUS Phase 7 WS7.1 Scientific Event Ingestion & Epistemic Guardrails Test Suite.
Validates:
1. Scientific Capability Registry declarations & EpistemicPromotionViolation guardrails
2. Quarantine Firewall runtime enforcement over Gate 3 cohort (1,808 pixels)
3. Investigation Reproducibility Contract & deterministic SHA-256 digest sealing
4. Rubin LSST diaSource / diaObject alert packet parsing & normalization
5. LVK Gravitational-Wave GCN notice parsing, counterpart scoring & normalization
6. Idempotent deduplication and dead-letter quarantine routing

Author: Dev Mehta <devmehta060505@gmail.com>
Lead Investigator & Mission Director, Project ASTRAEUS
Date: November 24, 2026
"""

import os
import json
import pytest
from pathlib import Path

from engines.capability_registry import (
    ScientificCapabilityRegistry,
    EngineCapabilityTier,
    EpistemicPromotionViolation,
)
from engines.quarantine_firewall import (
    QuarantineFirewall,
    QuarantineViolationError,
)
from engines.reproducibility import (
    InvestigationProvenanceRecord,
)
from astraeus_data.schemas.rubin_alert import (
    RubinAlertPacket,
    RubinDiaSource,
    RubinDiaObject,
)
from astraeus_data.schemas.gw_notice import (
    GwNoticePacket,
)
from astraeus_data.schemas.ztf_alert import (
    ZtfAlertPacket,
    ZtfCandidate,
)
from astraeus_data.ingestion.scientific_event_engine import (
    ScientificEventIngestionEngine,
    UnifiedScientificEvent,
    IngestionStatus,
)


# ============================================================================
# 1. SCIENTIFIC CAPABILITY REGISTRY & EPISTEMIC GUARDRAILS
# ============================================================================

class TestScientificCapabilityRegistry:
    """Verifies that every computational engine carries explicit capability tiers."""

    def test_registry_initialization_and_engine_tiers(self):
        ScientificCapabilityRegistry.initialize_phase7_defaults()
        engines = ScientificCapabilityRegistry.list_all()

        assert "engines.hydrodynamics.pino3d" in engines
        assert "engines.cosmology.differentiable" in engines
        assert "engines.experimental_design.ppo_scheduler" in engines
        assert "engines.experimental_design.greedy_eig" in engines
        assert "engines.cosmology.pm_solver" in engines

        # PINO3D is an experimental prototype due to IllustrisTNG 0.9999 error
        pino = engines["engines.hydrodynamics.pino3d"]
        assert pino.tier == EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE
        assert any("0.9999" in lim for lim in pino.empirical_limitations)

        # Differentiable cosmology is experimental prototype (N=5 mocks, Cramer-Rao bound)
        cosmo = engines["engines.cosmology.differentiable"]
        assert cosmo.tier == EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE

        # PPO scheduler is experimental prototype (Greedy EIG outperformed PPO)
        ppo = engines["engines.experimental_design.ppo_scheduler"]
        assert ppo.tier == EngineCapabilityTier.EXPERIMENTAL_PROTOTYPE
        assert any("122.3" in lim for lim in ppo.empirical_limitations)

        # Greedy EIG is validated instrument
        greedy = engines["engines.experimental_design.greedy_eig"]
        assert greedy.tier == EngineCapabilityTier.VALIDATED_INSTRUMENT

    def test_epistemic_promotion_blocked_for_experimental_prototypes(self):
        pino = ScientificCapabilityRegistry.get_engine("engines.hydrodynamics.pino3d")

        # Standard exploratory use is permitted
        pino.assert_usage_authorized("candidate_visualization", claiming_validated_discovery=False)

        # Attempting to claim validated scientific discovery raises EpistemicPromotionViolation
        with pytest.raises(EpistemicPromotionViolation) as excinfo:
            pino.assert_usage_authorized("autonomous_discovery", claiming_validated_discovery=True)

        assert "EPISTEMIC VIOLATION" in str(excinfo.value)
        assert "EXPERIMENTAL_PROTOTYPE" in str(excinfo.value)


# ============================================================================
# 2. RUNTIME QUARANTINE FIREWALL & GATE 3 PRESERVATION
# ============================================================================

class TestQuarantineFirewall:
    """Verifies runtime protection over Phase 5 Gate 3 1,808 quarantined test pixels."""

    def test_quarantine_pixel_count_and_covenant_integrity(self):
        count = QuarantineFirewall.get_quarantined_pixel_count()
        assert count == 1808

    def test_safe_coordinate_passes_quarantine_check(self):
        # Coordinates known to lie outside test split (e.g. pixel 0 is train split)
        # Compute coordinate for known safe pixel
        ra, dec = 180.0, 0.0
        pix = QuarantineFirewall.compute_healpix_nside32(ra, dec)
        if not QuarantineFirewall.is_quarantined_pixel(pix):
            returned_pix = QuarantineFirewall.assert_not_quarantined(ra, dec, identifier="test_safe")
            assert returned_pix == pix

    def test_quarantined_coordinate_triggers_violation(self):
        # Load known test pixel from configs/data_splits_healpix.json
        cfg_path = os.path.join("configs", "data_splits_healpix.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        quarantine_pixel = data["splits"]["test"][0]

        # Get sky coordinate for this quarantined pixel center
        center_coord = QuarantineFirewall._HP.healpix_to_skycoord(quarantine_pixel)
        ra_q = float(center_coord.ra.deg)
        dec_q = float(center_coord.dec.deg)

        with pytest.raises(QuarantineViolationError) as excinfo:
            QuarantineFirewall.assert_not_quarantined(ra_q, dec_q, identifier="quarantined_source")

        assert "GATE 3 QUARANTINE VIOLATION" in str(excinfo.value)
        assert str(quarantine_pixel) in str(excinfo.value)

    def test_quarantined_file_access_blocked(self):
        with pytest.raises(QuarantineViolationError):
            QuarantineFirewall.assert_safe_file_access("data/processed/real_survey_10k_v1/candidates_10k.parquet")


# ============================================================================
# 3. INVESTIGATION REPRODUCIBILITY CONTRACT
# ============================================================================

class TestInvestigationReproducibilityContract:
    """Verifies deterministic audit sealing for Phase 7 autonomous investigations."""

    def test_investigation_record_creation_and_sealing(self, tmp_path):
        record = InvestigationProvenanceRecord(
            investigation_id="inv_20261124_test_001",
            target_event_id="evt:rubin:99887766",
            target_coordinates={"ra_deg": 145.234, "dec_deg": 12.567},
            healpix_pixel_nside32=4521,
            random_seed=42,
            engine_capabilities_used={
                "engines.experimental_design.greedy_eig": "VALIDATED_INSTRUMENT",
                "engines.hydrodynamics.pino3d": "EXPERIMENTAL_PROTOTYPE",
            },
            checkpoint_hashes={"pino_hydro_surrogate.pt": "sha256:6e5e808..."},
            dataset_hashes={"camels_train_001.npz": "sha256:abcd1234..."},
        )

        record.record_tool_call(
            tool_name="greedy_eig_scheduler",
            arguments={"target_id": "TGT_01", "airmass": 1.25},
            output_summary="Optimal filter r-band, EIG=3.45",
        )

        record.record_hypothesis(
            hypothesis_id="H1",
            name="Kilonova",
            prior_prob=0.10,
            posterior_prob=0.75,
            evidence_citations=["Metzger (2019)", "Smartt et al. (2017)"],
        )

        digest1 = record.seal()
        assert len(digest1) == 64
        assert record.sealed_digest == digest1

        # Re-sealing is deterministic
        digest2 = record.seal()
        assert digest1 == digest2

        # Export JSON
        export_dir = str(tmp_path / "provenance")
        fpath = record.export_json(export_dir)
        assert os.path.exists(fpath)

        with open(fpath, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["sealed_digest"] == digest1
        assert len(loaded["hypotheses"]) == 1


# ============================================================================
# 4. RUBIN LSST & GW EVENT SCHEMAS AND INGESTION ENGINE
# ============================================================================

class TestScientificEventIngestionEngine:
    """Verifies unified multi-survey event ingestion with live quarantine checks."""

    @pytest.fixture
    def engine(self, tmp_path):
        return ScientificEventIngestionEngine(storage_root=tmp_path / "events")

    def test_rubin_alert_ingestion_and_normalization(self, engine):
        dia_src = RubinDiaSource(
            diaSourceId=88371928471,
            diaObjectId=10928374,
            midPointTai=60650.25,
            filterName="r",
            ra=165.5,
            dec=25.0,
            raErr=0.0001,
            decErr=0.0001,
            psfFlux=5000.0,       # 5,000 nJy ~ 22.15 AB mag
            psfFluxErr=250.0,     # SNR ~ 20.0
            snr=20.0,
            isDiffPos=True,
        )

        packet = RubinAlertPacket(
            alertId=1001,
            diaSource=dia_src,
            diaObject=RubinDiaObject(
                diaObjectId=10928374,
                ra=165.5,
                dec=25.0,
                nDiaSources=1,
            ),
            prvDiaSources=[],
        )

        # Ensure coordinate is safe from quarantine
        pix = QuarantineFirewall.compute_healpix_nside32(165.5, 25.0)
        if QuarantineFirewall.is_quarantined_pixel(pix):
            pytest.skip("Test coordinate happens to hit quarantine pixel; tested in quarantine test.")

        result = engine.ingest_rubin_packet(packet)

        assert result.status == IngestionStatus.INGESTED
        assert result.unified_event is not None
        assert result.unified_event.survey_origin == "RUBIN"
        assert result.unified_event.ra_deg == 165.5
        assert result.unified_event.dec_deg == 25.0
        assert len(result.unified_event.photometry_points) == 1
        assert result.unified_event.photometry_points[0]["filter"] == "r"
        assert abs(result.unified_event.photometry_points[0]["mag_ab"] - 22.15) < 0.1

        # Test idempotency deduplication
        dup_result = engine.ingest_rubin_packet(packet)
        assert dup_result.status == IngestionStatus.DUPLICATE_SKIPPED

    def test_gw_notice_ingestion_and_counterpart_promising(self, engine):
        # Use center coordinate of a verified train split pixel (pixel 0)
        center_0 = QuarantineFirewall._HP.healpix_to_skycoord(0)
        ra_0 = float(center_0.ra.deg)
        dec_0 = float(center_0.dec.deg)

        notice = GwNoticePacket(
            superevent_id="S261124a",
            notice_type="INITIAL",
            time_utc="2026-11-24T00:15:30Z",
            event_time_mjd=60650.0107,
            false_alarm_rate_hz=1e-12,
            far_yr=3.17e-5,
            area_90_deg2=48.5,             # Tight localization < 50 deg2
            distance_mean_mpc=120.0,
            distance_std_mpc=25.0,
            prob_bns=0.88,                 # High BNS probability
            prob_nsbh=0.08,
            prob_bbh=0.01,
            prob_terrestrial=0.03,
            prob_has_ns=0.96,
            prob_has_remnant=0.92,
            central_ra_deg=ra_0,
            central_dec_deg=dec_0,
        )

        assert notice.is_electromagnetic_counterpart_promising is True

        result = engine.ingest_gw_notice(notice)

        assert result.status == IngestionStatus.INGESTED
        assert result.unified_event is not None
        assert result.unified_event.survey_origin == "LVK_GW"
        assert result.unified_event.is_too_promising is True
        assert result.unified_event.event_metadata["prob_bns"] == 0.88

    def test_quarantined_alert_routed_to_dead_letter(self, engine):
        # Find a coordinate in the quarantined test split
        cfg_path = os.path.join("configs", "data_splits_healpix.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        q_pixel = data["splits"]["test"][10]

        center_coord = QuarantineFirewall._HP.healpix_to_skycoord(q_pixel)
        ra_q = float(center_coord.ra.deg)
        dec_q = float(center_coord.dec.deg)

        dia_src = RubinDiaSource(
            diaSourceId=99999999999,
            midPointTai=60650.30,
            filterName="g",
            ra=ra_q,
            dec=dec_q,
            raErr=0.0001,
            decErr=0.0001,
            psfFlux=3000.0,
            psfFluxErr=150.0,
            snr=20.0,
            isDiffPos=True,
        )
        packet = RubinAlertPacket(alertId=9999, diaSource=dia_src)

        result = engine.ingest_rubin_packet(packet)

        # Must be quarantined by covenant firewall
        assert result.status == IngestionStatus.QUARANTINED_COVENANT
        assert result.unified_event is None
        assert "GATE 3 QUARANTINE VIOLATION" in result.error_message
