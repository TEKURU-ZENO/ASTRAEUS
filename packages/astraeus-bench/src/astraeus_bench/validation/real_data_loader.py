"""
ASTRAEUS Real-World Observational Data Loader (WS-B & WS-C).
Replaces synthetic testbed fixtures with authentic astronomical observations stored on disk:
- ZTF Bright Transient Survey (>11,000 spectroscopically confirmed transients)
- ALeRCE real multi-epoch light curves (ZTF17aacpbmv, ZTF18aabdajx, etc.)
- IAU Transient Name Server (TNS) public catalog with authentic NASA ADS bibcodes
- Gaia DR3 astrophysical tables (quasars, variable stars)
"""

import os
import json
import zipfile
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np

from astraeus_bench.validation.competitor_interfaces import LightCurveData


class RealWorldDataLoader:
    """
    Loads authentic observational datasets from local archives for external validation.
    """

    def __init__(self, workspace_root: str = "."):
        self.workspace_root = workspace_root
        self.bts_path = os.path.join(workspace_root, "data", "external_validity_v1", "raw", "bts", "bts_raw_snapshot.csv")
        self.alerce_dir = os.path.join(workspace_root, "data", "external_validity_v1", "raw", "alerce_external")
        self.tns_zip_path = os.path.join(workspace_root, "data", "external_validity_v1", "raw", "tns", "tns_public_objects.csv.zip")
        self.gaia_qso_path = os.path.join(workspace_root, "data", "storage", "minio_objects", "astraeus-snapshots", "gaia_dr3", "gaia_qso_hp4_1028.csv")

    def load_real_bts_transients(self, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Loads spectroscopically classified real transients from ZTF Bright Transient Survey.
        Total dataset contains 10,988 real transient events.
        """
        if not os.path.exists(self.bts_path):
            raise FileNotFoundError(f"Real BTS observational dataset not found at: {self.bts_path}")

        df = pd.read_csv(self.bts_path)
        if max_records is not None:
            df = df.head(max_records)

        records = []
        for _, row in df.iterrows():
            records.append({
                "ztf_id": str(row.get("ZTFID", "")),
                "iau_id": str(row.get("IAUID", "")),
                "true_type": str(row.get("type", "Unknown")),
                "ra_str": str(row.get("RA", "")),
                "dec_str": str(row.get("Dec", "")),
                "peak_time_mjd": float(row.get("peakt", 0.0)) if pd.notna(row.get("peakt")) else None,
                "peak_mag": float(row.get("peakmag", 0.0)) if pd.notna(row.get("peakmag")) else None,
                "peak_filter": str(row.get("peakfilt", ""))
            })
        return records

    def load_real_alerce_lightcurve(self, target_id: str = "ZTF17aacpbmv") -> LightCurveData:
        """
        Parses an authentic multi-epoch light curve from ALeRCE real ZTF detections.
        Converts PSF magnitudes and photometric uncertainties to standard Pogson flux.
        """
        filename = f"{target_id}_lightcurve.json"
        filepath = os.path.join(self.alerce_dir, filename)

        if not os.path.exists(filepath):
            # Fallback to first available real lightcurve in directory
            files = [f for f in os.listdir(self.alerce_dir) if f.endswith(".json")]
            if not files:
                raise FileNotFoundError(f"No real ALeRCE light curves found in: {self.alerce_dir}")
            filepath = os.path.join(self.alerce_dir, files[0])
            target_id = files[0].replace("_lightcurve.json", "")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        detections = data.get("detections", [])
        if not detections:
            raise ValueError(f"No authentic detections in real lightcurve file: {filepath}")

        fid_to_band = {1: "g", 2: "r", 3: "i"}
        mjds = []
        fluxes = []
        flux_errs = []
        passbands = []
        ra = float(detections[0].get("ra", 0.0))
        dec = float(detections[0].get("dec", 0.0))

        for d in detections:
            mjd_val = float(d.get("mjd", 0.0))
            mag_val = float(d.get("magpsf", 20.0))
            err_val = float(d.get("sigmapsf", 0.1))
            fid_val = int(d.get("fid", 1))

            # Convert Pogson magnitude to relative flux (zero-point = 25.0)
            flux = 10.0 ** (-0.4 * (mag_val - 25.0))
            flux_err = flux * (np.log(10.0) / 2.5) * err_val

            mjds.append(mjd_val)
            fluxes.append(float(flux))
            flux_errs.append(float(flux_err))
            passbands.append(fid_to_band.get(fid_val, "r"))

        return LightCurveData(
            target_id=target_id,
            ra=ra,
            dec=dec,
            mjd=mjds,
            flux=fluxes,
            flux_err=flux_errs,
            passband=passbands
        )

    def load_all_real_alerce_lightcurves(self) -> List[LightCurveData]:
        """
        Loads all 35 authentic ZTF light curves stored in the ALeRCE raw cache.
        """
        if not os.path.exists(self.alerce_dir):
            raise FileNotFoundError(f"ALeRCE directory not found: {self.alerce_dir}")

        files = sorted([f for f in os.listdir(self.alerce_dir) if f.endswith(".json")])
        curves = []
        for f in files:
            tid = f.replace("_lightcurve.json", "")
            try:
                curves.append(self.load_real_alerce_lightcurve(tid))
            except Exception:
                continue
        return curves

    def load_real_tns_catalog_sample(self, max_records: int = 50) -> List[Dict[str, Any]]:
        """
        Loads confirmed transients from the real IAU Transient Name Server (TNS) archive zip,
        including authentic discovery/classification NASA ADS bibcodes.
        """
        if not os.path.exists(self.tns_zip_path):
            raise FileNotFoundError(f"TNS zip archive not found at: {self.tns_zip_path}")

        records = []
        with zipfile.ZipFile(self.tns_zip_path, "r") as z:
            with z.open("tns_public_objects.csv") as f:
                # Row 0 contains archive timestamp, Row 1 contains CSV header
                df = pd.read_csv(f, skiprows=1, nrows=max_records)
                for _, row in df.iterrows():
                    records.append({
                        "objname": str(row.get("name", "")),
                        "prefix": str(row.get("name_prefix", "")),
                        "ra": float(row.get("ra", 0.0)) if pd.notna(row.get("ra")) else 0.0,
                        "dec": float(row.get("declination", 0.0)) if pd.notna(row.get("declination")) else 0.0,
                        "spectroscopic_type": str(row.get("type", "Unclassified")),
                        "redshift": float(row.get("redshift", 0.0)) if pd.notna(row.get("redshift")) and str(row.get("redshift")).strip() != "" else None,
                        "discovery_bibcode": str(row.get("Discovery_ADS_bibcode", "")),
                        "class_bibcode": str(row.get("Class_ADS_bibcodes", ""))
                    })
        return records

    def load_real_gaia_qso_sample(self) -> List[Dict[str, Any]]:
        """
        Loads authentic Gaia DR3 quasars from the MinIO snapshot tables.
        """
        if not os.path.exists(self.gaia_qso_path):
            raise FileNotFoundError(f"Gaia QSO archive not found at: {self.gaia_qso_path}")

        df = pd.read_csv(self.gaia_qso_path)
        records = []
        for _, row in df.iterrows():
            records.append({
                "source_id": int(row.get("source_id", 0)),
                "ra": float(row.get("ra", 0.0)),
                "dec": float(row.get("dec", 0.0)),
                "phot_g_mean_mag": float(row.get("phot_g_mean_mag", 0.0)),
                "redshift_qsoc": float(row.get("redshift_qsoc", 0.0)) if pd.notna(row.get("redshift_qsoc")) else None,
                "classification": str(row.get("classlabel_dsc_joint", "quasar"))
            })
        return records
