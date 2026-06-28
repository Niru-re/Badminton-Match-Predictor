"""Export sklearn models to pure JSON so the Vercel API needs no sklearn."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR = ROOT / "models"
DISCIPLINES = ["MS", "WS", "MD", "WD", "XD"]


def export_tree(tree, n_features):
    """Recursively export a sklearn DecisionTree to a plain dict."""
    t = tree.tree_
    def node(i):
        if t.children_left[i] == -1:  # leaf
            vals = t.value[i][0]
            total = vals.sum()
            return {"leaf": True, "prob": float(vals[1] / total) if total > 0 else 0.5}
        return {
            "leaf": False,
            "feature": int(t.feature[i]),
            "threshold": float(t.threshold[i]),
            "left": node(t.children_left[i]),
            "right": node(t.children_right[i]),
        }
    return node(0)


def export_rf(model):
    return {
        "type": "RandomForest",
        "trees": [export_tree(est, model.n_features_in_) for est in model.estimators_],
    }


def export_gb(model):
    """GradientBoosting: export all trees + learning_rate + init prior."""
    init_prob = float(np.mean(model.init_.class_prior_))
    import math
    init_log_odds = math.log(init_prob / (1 - init_prob)) if 0 < init_prob < 1 else 0.0

    def gb_tree(est):
        t = est[0].tree_
        def node(i):
            if t.children_left[i] == -1:
                return {"leaf": True, "val": float(t.value[i][0][0])}
            return {
                "leaf": False,
                "feature": int(t.feature[i]),
                "threshold": float(t.threshold[i]),
                "left": node(t.children_left[i]),
                "right": node(t.children_right[i]),
            }
        return node(0)

    return {
        "type": "GradientBoosting",
        "learning_rate": float(model.learning_rate),
        "init_log_odds": init_log_odds,
        "trees": [gb_tree(stage) for stage in model.estimators_],
    }


def export_all():
    for d in DISCIPLINES:
        pkl_path = MODELS_DIR / f"best_model_{d.lower()}.pkl"
        if not pkl_path.exists():
            print(f"SKIP {d}: no pkl")
            continue
        bundle = pickle.load(open(pkl_path, "rb"))
        model = bundle["model"]
        mtype = type(model).__name__

        if "RandomForest" in mtype:
            data = export_rf(model)
        elif "GradientBoosting" in mtype:
            data = export_gb(model)
        else:
            print(f"SKIP {d}: unsupported model {mtype}")
            continue

        data["features"] = bundle["features"]
        data["discipline"] = d
        out = MODELS_DIR / f"model_{d.lower()}.json"
        with open(out, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        size_mb = out.stat().st_size / 1024 / 1024
        print(f"{d}: exported {mtype} -> {out.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    export_all()
