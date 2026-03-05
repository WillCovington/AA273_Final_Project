import numpy as np
import json
from pathlib import Path

def save_ekf_run(date_str, runname, ts, X_truth, Xhat, Phat, meta: dict, out_root="runs"):
    """
    Saves:
      runs/<date_str>/<runname>/<runname>.npz
      runs/<date_str>/<runname>/figures/   (created empty)
    """
    out_root = Path(out_root)
    run_dir = out_root / date_str / runname
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    ts = np.asarray(ts, dtype=np.float64)
    X_truth = np.asarray(X_truth, dtype=np.float64)
    Xhat = np.asarray(Xhat, dtype=np.float64)

    # ✅ Important: store Phat as (K,6,6) array
    P = np.stack(Phat, axis=0).astype(np.float64)

    meta = dict(meta)
    meta["date_dir"] = date_str
    meta["runname"] = runname

    npz_path = run_dir / f"{runname}.npz"
    np.savez_compressed(
        npz_path,
        ts=ts,
        X_truth=X_truth,
        Xhat=Xhat,
        Phat=P,
        meta_json=json.dumps(meta, indent=2),
    )

    return str(npz_path), str(fig_dir)