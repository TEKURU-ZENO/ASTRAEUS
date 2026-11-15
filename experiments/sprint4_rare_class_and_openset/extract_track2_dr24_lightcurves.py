"""
Extracts ZTF DR24 light curves for Track 2 Unclassified L5 candidates.
Matches candidate sky coordinates within 1.5 arcsec against DR24 parquet partitions
and saves observation records to data/raw/real_survey_v1/ztf_dr24/lightcurves/<candidate_id>.json.
"""

import json
import time
from pathlib import Path
import astropy.units as u
from astropy.coordinates import SkyCoord, search_around_sky
import numpy as np
import pandas as pd


def main():
    t0 = time.time()
    candidates_path = Path("data/processed/real_survey_10k_v1/candidates_10k.parquet")
    dr24_dir = Path("data/raw/real_survey_v1/ztf_dr24")
    out_dir = dr24_dir / "lightcurves"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(candidates_path)
    t2 = df[df["coarse_class"] == "UNCLASSIFIED"].copy().reset_index(drop=True)
    cids = t2["candidate_id"].values
    ras = t2["ra_deg"].values
    decs = t2["dec_deg"].values

    print(f"Loaded {len(t2)} Track 2 candidates. Indexing coordinates...")
    c_sc = SkyCoord(ra=ras * u.deg, dec=decs * u.deg)

    # Accumulate observations per candidate
    cand_obs = {cid: [] for cid in cids}

    parquet_files = sorted(dr24_dir.glob("*.parquet"))
    for pfile in parquet_files:
        t_p = time.time()
        print(f"Processing {pfile.name}...")
        p_df = pd.read_parquet(pfile)
        p_sc = SkyCoord(ra=p_df["objra"].values * u.deg, dec=p_df["objdec"].values * u.deg)

        idx_c, idx_p, _, _ = search_around_sky(c_sc, p_sc, 1.5 * u.arcsec)
        print(f"  Found {len(idx_c)} raw match pairs in {time.time() - t_p:.2f}s")

        for c_i, p_i in zip(idx_c, idx_p):
            cid = cids[c_i]
            row = p_df.iloc[p_i]
            fid = int(row["filterid"])
            lc = row["lightcurve"]
            if not isinstance(lc, dict):
                continue
            hmjds = lc.get("hmjd", [])
            mags = lc.get("mag", [])
            magerrs = lc.get("magerr", [])
            catflags = lc.get("catflags", [0] * len(hmjds))

            for h, m, e, cf in zip(hmjds, mags, magerrs, catflags):
                cand_obs[cid].append({
                    "hmjd": float(h),
                    "mag": float(m),
                    "magerr": float(e),
                    "filterid": fid,
                    "catflags": int(cf),
                })

    # Save to JSON
    saved_count = 0
    for cid, obs in cand_obs.items():
        if obs:
            obs.sort(key=lambda x: x["hmjd"])
            out_file = out_dir / f"{cid}.json"
            with open(out_file, "w") as f:
                json.dump({"candidate_id": cid, "observations": obs}, f)
            saved_count += 1

    print(f"Extracted and saved {saved_count} Track 2 DR24 light curves in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
