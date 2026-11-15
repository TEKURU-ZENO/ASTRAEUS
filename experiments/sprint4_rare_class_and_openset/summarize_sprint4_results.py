import json

def main():
    with open("experiments/sprint4_rare_class_and_openset/sprint4a_rare_class_results.json") as f:
        d_4a = json.load(f)
    with open("experiments/sprint4_rare_class_and_openset/sprint4_temporal_calibration_openset_results.json") as f:
        d_4bcd = json.load(f)

    print("=== SPRINT 4A: IMBALANCE STRATEGY COMPARISON ===")
    print(f"{'Strategy':28s} | {'CV Macro-F1 (std)':19s} | {'CV Bal-Acc':10s} | {'Val Macro-F1':12s} | {'Val Bal-Acc':11s} | {'TDE F1':6s} | {'Other F1':8s} | {'Val Acc':7s}")
    print("-" * 115)
    for strat, cv_res in d_4a["cross_validation_results"].items():
        val_res = d_4a["validation_holdout_results"][strat]
        cv_str = f"{cv_res['cv_macro_f1_mean']:.4f} (+/-{cv_res['cv_macro_f1_std']:.4f})"
        tde_f1 = val_res["per_class"]["TDE"]["f1"]
        oe_f1 = val_res["per_class"]["OTHER_EXTRAGALACTIC"]["f1"]
        print(f"{strat:28s} | {cv_str:19s} | {cv_res['cv_bal_acc_mean']:10.4f} | {val_res['val_macro_f1']:12.4f} | {val_res['val_balanced_accuracy']:11.4f} | {tde_f1:6.4f} | {oe_f1:8.4f} | {val_res['val_accuracy']:7.4f}")

    print("\n=== SPRINT 4B: TEMPORAL SEQUENCE VS PHYSICAL FEATURE BASELINE ===")
    ast = d_4bcd["sprint4b_temporal_comparison"]["astromer"]
    brf = d_4bcd["sprint4b_temporal_comparison"]["balanced_rf"]
    print(f"{'Model':34s} | {'Macro-F1':8s} | {'Bal-Acc':8s} | {'Top-1 Acc':9s} | {'Latency (ms)':12s}")
    print("-" * 80)
    print(f"{ast['model_name']:34s} | {ast['macro_f1']:8.4f} | {ast['balanced_accuracy']:8.4f} | {ast['accuracy']:9.4f} | {ast['inference_latency_ms_per_sample']:12.2f}")
    print(f"{brf['model_name']:34s} | {brf['macro_f1']:8.4f} | {brf['balanced_accuracy']:8.4f} | {brf['accuracy']:9.4f} | {brf['inference_latency_ms_per_sample']:12.2f}")

    print("\n=== SPRINT 4C: RISK-COVERAGE & HONEST ABSTENTION ===")
    print(f"{'Rejection Conf Threshold':25s} | {'Coverage':9s} | {'Accepted Acc':12s} | {'Accepted Macro-F1':17s} | {'TDE Abstention Rate':19s} | {'Other Abstention Rate':21s}")
    print("-" * 115)
    for row in d_4bcd["sprint4c_calibration_and_abstention"]["risk_coverage_curve"]:
        tde_ar = row["rare_class_abstention"]["TDE"]["abstention_rate"] * 100
        oe_ar = row["rare_class_abstention"]["OTHER_EXTRAGALACTIC"]["abstention_rate"] * 100
        print(f"Conf >= {row['confidence_threshold']:<17.2f} | {row['coverage']*100:6.1f}%   | {row['accepted_accuracy']*100:9.1f}%  | {row['accepted_macro_f1']:17.4f} | {tde_ar:16.1f}%   | {oe_ar:18.1f}%")

    print("\n=== SPRINT 4D: TOP 10 UNCLASSIFIED DISCOVERIES (Track 2, N=4,365) ===")
    print(f"{'Rank':4s} | {'Candidate ID':26s} | {'RA (deg)':8s} | {'Dec (deg)':9s} | {'Score':6s} | {'IsoF':5s} | {'LOF':5s} | {'Mahal':5s} | {'Primary Anomaly Feature':26s} (Value, z-score)")
    print("-" * 125)
    for d in d_4bcd["sprint4d_track2_openset_discovery"]["top10_candidate_discoveries"]:
        print(f"#{d['rank']:<3d} | {d['candidate_id']:26s} | {d['ra_deg']:8.4f} | {d['dec_deg']:9.4f} | {d['composite_anomaly_score']:6.4f} | {d['isolation_forest_score']:5.3f} | {d['lof_score']:5.3f} | {d['mahalanobis_score']:5.3f} | {d['primary_anomalous_feature']:26s} ({d['feature_value']:8.2f}, z={d['feature_z_score']:4.1f})")

if __name__ == "__main__":
    main()
