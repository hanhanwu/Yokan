"""Shared data and DQN helpers for the two RCA experiments."""

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


REVISION = "8b6c06a7a15bcecdbf70b611a13b789c5925dc17"
URL = f"https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/{REVISION}"


def download(data, relative_path):
    target = data / relative_path
    if target.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "2",
             "--max-time", "180", "--output", str(temporary), f"{URL}/{relative_path}"],
            check=True,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def case_index(data, dataset):
    download(data, "cases.parquet")
    index = pd.read_parquet(data / "cases.parquet")
    return index.loc[index["dataset"].eq(dataset)].sort_values("case")


def download_many(data, paths):
    with ThreadPoolExecutor(max_workers=12) as workers:
        for count, _ in enumerate(workers.map(partial(download, data), paths), start=1):
            if count % 25 == 0 or count == len(paths):
                print(f"{count}/{len(paths)} files ready", flush=True)


def window_changes(frame, onset, columns, window=120):
    """Return bounded signed shifts and flags for usable metric pairs."""
    times = frame["time"]
    normal = frame.loc[(onset - window <= times) & (times < onset)]
    faulty = frame.loc[(onset <= times) & (times < onset + window)]
    before = normal.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
    after = faulty.reindex(columns=columns).replace([np.inf, -np.inf], np.nan)
    center = before.median()
    scale = (before - center).abs().median() * 1.4826
    scale = scale.where(scale > 0, before.std()).fillna(0)
    scale = np.maximum(scale, (center.abs() + 1) * 1e-6)
    present = (before.count() >= 10) & (after.count() >= 10)
    shift = (after.median() - center) / scale
    bounded = (np.clip(np.arcsinh(shift), -10, 10) / 10).where(present, 0)
    return np.column_stack((bounded, present)).astype(np.float32)


def train(env, steps, seed, directory, name, callback=None):
    check_env(env, warn=True)
    monitored = Monitor(env, str(directory / f"{name}-{seed}.monitor.csv"))
    model = DQN(
        "MlpPolicy", monitored, seed=seed, device="cpu",
        policy_kwargs={"net_arch": [64, 64]},
        learning_rate=1e-3, buffer_size=20_000, learning_starts=256,
        gamma=1.0, exploration_fraction=0.5, exploration_final_eps=0.05,
        batch_size=64, train_freq=4, target_update_interval=250,
    )
    try:
        model.learn(total_timesteps=steps, callback=callback)
        model.save(directory / f"{name}-{seed}")
    finally:
        monitored.close()
    return model


