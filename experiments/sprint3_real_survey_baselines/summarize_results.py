import json

def main():
    with open("experiments/sprint3_real_survey_baselines/sprint3_baseline_results.json") as f:
        d = json.load(f)

    rf_phys = d["configurations"]["COMBINED"]["physical_27_features"]["RandomForest"]["per_class"]
    hgbt_phys = d["configurations"]["COMBINED"]["physical_27_features"]["HistGradientBoosting"]["per_class"]

    print("=== PER-CLASS BREAKDOWN: RANDOM FOREST (COMBINED, Physical-27) ===")
    print(f"{'Class':24s} | {'Support':7s} | {'Precision':9s} (95% CI)       | {'Recall':6s} (95% CI)       | {'F1':6s}")
    print("-" * 85)
    for c, m in rf_phys.items():
        rare = " [RARE]" if m["is_rare_class"] else ""
        cname = c + rare
        p_ci = f"[{m['precision_ci_95'][0]:.2f}-{m['precision_ci_95'][1]:.2f}]"
        r_ci = f"[{m['recall_ci_95'][0]:.2f}-{m['recall_ci_95'][1]:.2f}]"
        print(f"{cname:24s} | {m['support']:7d} | {m['precision']:6.4f} {p_ci:14s} | {m['recall']:6.4f} {r_ci:14s} | {m['f1']:6.4f}")

    print("\n=== PER-CLASS BREAKDOWN: HIST GRADIENT BOOSTING (COMBINED, Physical-27) ===")
    print(f"{'Class':24s} | {'Support':7s} | {'Precision':9s} (95% CI)       | {'Recall':6s} (95% CI)       | {'F1':6s}")
    print("-" * 85)
    for c, m in hgbt_phys.items():
        rare = " [RARE]" if m["is_rare_class"] else ""
        cname = c + rare
        p_ci = f"[{m['precision_ci_95'][0]:.2f}-{m['precision_ci_95'][1]:.2f}]"
        r_ci = f"[{m['recall_ci_95'][0]:.2f}-{m['recall_ci_95'][1]:.2f}]"
        print(f"{cname:24s} | {m['support']:7d} | {m['precision']:6.4f} {p_ci:14s} | {m['recall']:6.4f} {r_ci:14s} | {m['f1']:6.4f}")

    print("\n=== TOP 10 FEATURE IMPORTANCES (RandomForest, COMBINED, Physical-27) ===")
    top_feats = d["configurations"]["COMBINED"]["physical_27_features"]["RandomForest"]["feature_importances_top10"]
    for f_name, imp in top_feats.items():
        print(f"  {f_name:28s}: {imp:.5f}")

    print("\n=== CONFUSION MATRIX (RandomForest, COMBINED, Physical-27) ===")
    classes = d["taxonomy"]
    print("True \\ Pred: " + " ".join([f"{c[:6]:>7s}" for c in classes]))
    cm = d["configurations"]["COMBINED"]["physical_27_features"]["RandomForest"]["confusion_matrix"]
    for i, row in enumerate(cm):
        row_str = " ".join([f"{val:7d}" for val in row])
        print(f"{classes[i][:11]:11s} {row_str}")

if __name__ == "__main__":
    main()
