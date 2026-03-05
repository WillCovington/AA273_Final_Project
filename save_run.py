import numpy as np
import json
from pathlib import Path
from datetime import datetime

def save_ekf_run(out_dir, ts, X_truth, Xhat, Phat, meta: dict):
    # takes one of our EKF runs and saves it for later analysis
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = np.asarray(ts, dtype=np.float64)
    X_truth = np.asarray(X_truth, dtype=np.float64)
    Xhat = np.asarray(Xhat, dtype=np.float64)
    P = np.stack(Phat, axis=0).astype(np.float64)   # (K,6,6)

    meta = dict(meta)
    meta["saved_utc"] = datetime.utcnow().isoformat() + "Z"

    fname = meta.get("run_name", None)
    if fname is None:
        fname = datetime.utcnow().strftime("ekf_run_%Y%m%d_%H%M%S")

    path = out_dir / f"{fname}.npz"

    np.savez_compressed(
        path,
        ts=ts,
        X_truth=X_truth,
        Xhat=Xhat,
        Phat=P,
        meta_json=json.dumps(meta, indent=2),
    )

    return str(path)