import numpy as np
import json
from pathlib import Path

def save_ekf_run(date_str, runname, ts, X_truth, Xhat, Phat, meta: dict, timing = None, out_root="runs"):
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

    # checks the shape of Phat
    Phat = np.asarray(Phat, dtype=np.float64)
    if Phat.ndim == 3:
        P = Phat
    else:
        P = np.stack(Phat, axis=0).astype(np.float64)

    meta = dict(meta)
    meta["date_dir"] = date_str
    meta["runname"] = runname

    npz_path = run_dir / f"{runname}.npz"

    # Base payload
    npz_kwargs = dict(
        ts=ts,
        X_truth=X_truth,
        Xhat=Xhat,
        Phat=P,
        meta_json=json.dumps(meta, indent=2),
    )

    if timing is not None:
        for k, v in timing.items():
            npz_kwargs[f"timing_{k}"] = np.asarray(v, dtype=np.float64)

    np.savez_compressed(npz_path, **npz_kwargs)

    meta_path = run_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return str(npz_path), str(fig_dir)