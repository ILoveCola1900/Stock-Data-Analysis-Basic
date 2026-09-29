"""Core utilities for leakage-aware rolling CEEMDAN experiments."""

from __future__ import annotations

import pickle
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SPLIT_DATE = pd.Timestamp("2024-02-29")
RANDOM_SEED = 42


def garman_klass(frame: pd.DataFrame) -> pd.Series:
    high = frame["High"].astype(float)
    low = frame["Low"].astype(float)
    close = frame["Close"].astype(float)
    open_ = frame["Open"].astype(float)
    hl = np.log(high / low)
    co = np.log(close / open_)
    rv = 0.5 * hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * co.pow(2)
    return rv.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0.0).rename("rv")


def fetch_eastmoney_rv(start: str = "2023-01-03", end: str = "2024-08-30") -> pd.Series:
    import json
    import urllib.parse
    import urllib.request

    params = {
        "secid": "1.510300",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",
        "fqt": "0",
        "beg": start.replace("-", ""),
        "end": end.replace("-", ""),
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = []
    for item in payload["data"]["klines"]:
        date, open_, close, high, low = item.split(",")[:5]
        rows.append(
            {
                "date": pd.Timestamp(date),
                "Open": float(open_),
                "Close": float(close),
                "High": float(high),
                "Low": float(low),
            }
        )
    return garman_klass(pd.DataFrame(rows).set_index("date").sort_index())


def fetch_akshare_rv(start: str = "2023-01-03", end: str = "2024-08-30") -> pd.Series:
    import akshare as ak

    frame = ak.fund_etf_hist_sina(symbol="sh510300")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame[
        (frame["date"] >= pd.Timestamp(start)) & (frame["date"] <= pd.Timestamp(end))
    ]
    frame = frame.set_index("date").sort_index()
    frame = frame.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
    return garman_klass(frame)


def synthetic_fallback_rv(n: int = 404, seed: int = RANDOM_SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-03", periods=n)
    regimes = np.r_[
        np.full(n // 3, 0.000035),
        np.full(n // 3, 0.000065),
        np.full(n - 2 * (n // 3), 0.000085),
    ]
    seasonal = 0.000025 * np.abs(np.sin(np.arange(n) * 2.0 * np.pi / 21.0))
    shocks = rng.gamma(shape=2.0, scale=0.000025, size=n)
    return pd.Series(np.clip(regimes + seasonal + shocks, 1e-8, None), index=dates, name="rv")


def load_rv_series(
    csv_path: Path | str,
    allow_network: bool = True,
    allow_synthetic_fallback: bool = True,
) -> Tuple[pd.Series, str]:
    path = Path(csv_path)
    if path.exists():
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
        series = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna()
        series.index = pd.to_datetime(series.index)
        series.name = "rv"
        return series.sort_index(), f"csv:{path}"

    errors: List[str] = []
    if allow_network:
        for fetcher, label in [(fetch_eastmoney_rv, "eastmoney"), (fetch_akshare_rv, "akshare_sina")]:
            try:
                series = fetcher()
                if len(series) >= 300:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    series.rename("rv").to_csv(path, index_label="date")
                    return series, label
            except Exception as exc:  # pragma: no cover
                errors.append(f"{label}: {exc}")
                warnings.warn(f"Failed to load data from {label}: {exc}", RuntimeWarning)
    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"Missing {path} and all network data sources failed. " + " | ".join(errors)
        )
    warnings.warn(
        "Using deterministic synthetic RV data because CSV and network sources failed.",
        RuntimeWarning,
        stacklevel=2,
    )
    return synthetic_fallback_rv(), "synthetic_fallback"


def validate_split(rv_series: Sequence[float] | pd.Series) -> Tuple[int, int]:
    index = rv_series.index if isinstance(rv_series, pd.Series) else pd.RangeIndex(len(rv_series))
    index = pd.to_datetime(index)
    train_end_idx = int(np.searchsorted(index, SPLIT_DATE, side="right"))
    return train_end_idx, len(index) - train_end_idx


def decompose_ceemdan(
    history: np.ndarray,
    *,
    trials: int = 100,
    parallel: bool = False,
    processes: Optional[int] = None,
    max_imf: int = 20,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    from PyEMD import CEEMDAN

    np.random.seed(seed)
    kwargs: Dict[str, Any] = {
        "trials": int(trials),
        "epsilon": 0.005,
        "parallel": bool(parallel),
        "max_imf": int(max_imf),
        "seed": int(seed),
    }
    if processes is not None:
        kwargs["processes"] = int(processes)
    ceemdan = CEEMDAN(**kwargs)
    imfs = np.asarray(ceemdan(history.astype(float, copy=False), progress=False))
    if imfs.ndim == 1:
        imfs = imfs[np.newaxis, :]
    if imfs.ndim != 2 or imfs.shape[1] != len(history):
        raise ValueError(f"Unexpected CEEMDAN output shape: {imfs.shape}")
    return imfs


def rolling_ceemdan(
    rv_series: Sequence[float] | pd.Series,
    train_end_idx: int,
    window: Optional[int] = None,
    *,
    max_steps: Optional[int] = None,
    trials: int = 100,
    parallel: bool = False,
    processes: Optional[int] = None,
    max_imf: int = 20,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """For test day t, decompose rv_series[:t] and keep every raw result."""
    values = np.asarray(rv_series, dtype=float)
    if values.ndim != 1:
        raise ValueError("rv_series must be one-dimensional")
    if not 0 < int(train_end_idx) < len(values):
        raise ValueError("train_end_idx must be between 1 and len(rv_series) - 1")
    if window is not None and int(window) <= 0:
        raise ValueError("window must be a positive integer or None")

    target_indices = list(range(int(train_end_idx), len(values)))
    if max_steps is not None:
        target_indices = target_indices[: int(max_steps)]

    steps: List[Dict[str, Any]] = []
    started_all = time.perf_counter()
    for step_no, target_idx in enumerate(target_indices, start=1):
        history_end = target_idx
        history_start = 0 if window is None else max(0, history_end - int(window))
        history = values[history_start:history_end]
        started = time.perf_counter()
        error = None
        imfs: Optional[np.ndarray] = None
        try:
            if len(history) < 64:
                raise ValueError("History is too short for CEEMDAN (<64 observations)")
            imfs = decompose_ceemdan(
                history,
                trials=trials,
                parallel=parallel,
                processes=processes,
                max_imf=max_imf,
                seed=seed + step_no - 1,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"[WARN] step={step_no:02d} target_idx={target_idx}: {error}")

        elapsed = time.perf_counter() - started
        if imfs is None:
            energy = np.full(3, np.nan)
            n_imfs = 0
        else:
            energy = np.array([np.sum(component**2) for component in imfs[:3]])
            if energy.size < 3:
                energy = np.pad(energy, (0, 3 - energy.size), constant_values=np.nan)
            n_imfs = int(imfs.shape[0])

        step_data = {
            "step": step_no,
            "target_idx": int(target_idx),
            "target_date": (
                pd.Timestamp(rv_series.index[target_idx]).isoformat()
                if isinstance(rv_series, pd.Series)
                else None
            ),
            "history_start_idx": int(history_start),
            "history_end_idx": int(history_end),
            "history_len": int(len(history)),
            "n_imfs": n_imfs,
            "elapsed_sec": float(elapsed),
            "energy_first3": energy,
            "error": error,
            "raw_imfs": None if imfs is None else imfs.copy(),
        }
        steps.append(step_data)

        if verbose:
            energy_text = ",".join(f"{x:.4e}" for x in energy)
            print(
                f"[{step_no:02d}/{len(target_indices):02d}] "
                f"{step_data['target_date'] or target_idx} "
                f"history={len(history):3d} n_imfs={n_imfs:2d} "
                f"time={elapsed:6.2f}s energy={energy_text}"
            )

    max_imfs = max((item["n_imfs"] for item in steps), default=0)
    max_history_len = max((item["history_len"] for item in steps), default=0)
    aligned = np.full((len(steps), max_imfs, max_history_len), np.nan, dtype=float)
    for step_idx, item in enumerate(steps):
        raw = item["raw_imfs"]
        if raw is not None:
            aligned[step_idx, : raw.shape[0], : raw.shape[1]] = raw

    total_elapsed = time.perf_counter() - started_all
    counts = Counter(item["n_imfs"] for item in steps)
    metadata: Dict[str, Any] = {
        "method": "rolling_ceemdan",
        "train_end_idx": int(train_end_idx),
        "window": None if window is None else int(window),
        "requested_steps": int(len(target_indices)),
        "completed_steps": int(sum(item["error"] is None for item in steps)),
        "failed_steps": int(sum(item["error"] is not None for item in steps)),
        "trials": int(trials),
        "parallel": bool(parallel),
        "processes": None if processes is None else int(processes),
        "seed": int(seed),
        "max_history_len": int(max_history_len),
        "max_n_imfs": int(max_imfs),
        "imf_count_distribution": dict(sorted(counts.items())),
        "average_elapsed_sec": float(np.mean([item["elapsed_sec"] for item in steps]))
        if steps
        else 0.0,
        "min_elapsed_sec": float(np.min([item["elapsed_sec"] for item in steps]))
        if steps
        else 0.0,
        "max_elapsed_sec": float(np.max([item["elapsed_sec"] for item in steps]))
        if steps
        else 0.0,
        "total_elapsed_sec": float(total_elapsed),
        "steps": steps,
    }
    return aligned, metadata


def select_first_k(imfs: Optional[np.ndarray], k: int) -> np.ndarray:
    if imfs is None:
        return np.zeros((0, 0), dtype=float)
    array = np.asarray(imfs, dtype=float)
    if array.ndim != 2:
        raise ValueError("imfs must have shape (n_imfs, n_samples)")
    result = np.zeros_like(array, dtype=float)
    usable = min(int(k), array.shape[0])
    if usable:
        result[:usable] = array[:usable]
    return result


def _arima_forecast(component: np.ndarray, n_steps: int) -> np.ndarray:
    """Fit ARIMA on a centered/scaled component and restore its physical scale."""
    component = np.asarray(component, dtype=float).reshape(-1)
    n_steps = int(n_steps)
    if component.size == 0 or n_steps <= 0:
        return np.zeros(max(0, n_steps), dtype=float)
    finite = component[np.isfinite(component)]
    if finite.size == 0:
        return np.zeros(n_steps, dtype=float)
    center = float(np.mean(finite))
    scale = float(np.std(finite))
    if not np.isfinite(scale) or scale <= 1e-15:
        return np.full(n_steps, center, dtype=float)
    scaled = (component - center) / scale
    try:
        from statsmodels.tsa.arima.model import ARIMA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = ARIMA(scaled, order=(1, 1, 1)).fit().forecast(steps=n_steps)
        forecast = np.asarray(forecast, dtype=float).reshape(-1)
        if forecast.size < n_steps:
            forecast = np.pad(forecast, (0, n_steps - forecast.size), mode="edge")
        restored = forecast[:n_steps] * scale + center
        if not np.all(np.isfinite(restored)):
            raise ValueError("ARIMA returned non-finite forecast")
        return restored
    except Exception:
        return np.full(n_steps, float(finite[-1]), dtype=float)


def forecast_component(
    component: np.ndarray,
    *,
    model: str = "arima",
    n_high: int = 2,
    component_idx: int = 0,
) -> float:
    component = np.asarray(component, dtype=float)
    if component.size == 0:
        return 0.0
    if model == "persistence":
        return float(component[-1])
    if component_idx < int(n_high):
        return float(np.mean(component))
    return float(_arima_forecast(component, 1)[0])


def rolling_reconstruction(
    rolling_metadata: Dict[str, Any],
    *,
    k_values: Iterable[int] = (1, 3, 5),
    forecast_model: str = "arima",
    n_high: int = 2,
    fallback_value: Optional[float] = None,
) -> Dict[int, np.ndarray]:
    result: Dict[int, List[float]] = {int(k): [] for k in k_values}
    for step in rolling_metadata["steps"]:
        raw = step.get("raw_imfs")
        if raw is None:
            value = float(fallback_value) if fallback_value is not None else np.nan
            for k in result:
                result[k].append(value)
            continue
        forecasts = [
            forecast_component(
                component, model=forecast_model, n_high=n_high, component_idx=idx
            )
            for idx, component in enumerate(raw)
        ]
        for k in result:
            result[k].append(float(np.sum(forecasts[: int(k)])))
    return {k: np.asarray(values, dtype=float) for k, values in result.items()}


def forecast_component_multi(
    component: np.ndarray,
    n_steps: int,
    *,
    model: str = "arima",
    n_high: int = 2,
    component_idx: int = 0,
) -> np.ndarray:
    """Forecast one IMF for multiple future steps with the Day-2 convention."""
    component = np.asarray(component, dtype=float)
    n_steps = int(n_steps)
    if component.size == 0 or n_steps <= 0:
        return np.zeros(max(0, n_steps), dtype=float)
    if model == "persistence":
        return np.full(n_steps, float(component[-1]), dtype=float)
    if component_idx < int(n_high):
        return np.full(n_steps, float(np.mean(component)), dtype=float)
    return _arima_forecast(component, n_steps)


def decompose_full_and_fragment(
    rv_series: pd.Series,
    train_end_idx: int,
    *,
    n_test: Optional[int] = None,
    trials: int = 100,
    parallel: bool = False,
    processes: Optional[int] = None,
    max_imf: int = 20,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Compute the existing leakage and fragment-plus-ARIMA baselines."""
    values = np.asarray(rv_series, dtype=float)
    n_test = len(values) - int(train_end_idx) if n_test is None else int(n_test)
    train_values = values[: int(train_end_idx)]

    started = time.perf_counter()
    imfs_full = decompose_ceemdan(
        values,
        trials=trials,
        parallel=parallel,
        processes=processes,
        max_imf=max_imf,
        seed=seed,
    )
    full_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    imfs_train = decompose_ceemdan(
        train_values,
        trials=trials,
        parallel=parallel,
        processes=processes,
        max_imf=max_imf,
        seed=seed + 100000,
    )
    fragment_elapsed = time.perf_counter() - started

    extrapolated = np.zeros((imfs_train.shape[0], n_test), dtype=float)
    for idx, component in enumerate(imfs_train):
        extrapolated[idx, :] = forecast_component_multi(
            component,
            n_test,
            model="arima",
            n_high=2,
            component_idx=idx,
        )

    return {
        "imfs_full": imfs_full,
        "imfs_train": imfs_train,
        "imfs_fragment_extrapolated": extrapolated,
        "n_train": int(train_end_idx),
        "n_test": int(n_test),
        "full_elapsed_sec": float(full_elapsed),
        "fragment_elapsed_sec": float(fragment_elapsed),
    }


def reconstruction_rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(prediction)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((truth[mask] - prediction[mask]) ** 2)))


def save_pickle(payload: Any, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def stage1_metadata_frame(metadata: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for step in metadata["steps"]:
        energy = np.asarray(step["energy_first3"], dtype=float)
        rows.append(
            {
                "step": step["step"],
                "target_date": step["target_date"],
                "history_len": step["history_len"],
                "n_imfs": step["n_imfs"],
                "elapsed_sec": step["elapsed_sec"],
                "energy_imf1": energy[0] if energy.size > 0 else np.nan,
                "energy_imf2": energy[1] if energy.size > 1 else np.nan,
                "energy_imf3": energy[2] if energy.size > 2 else np.nan,
                "error": step["error"],
            }
        )
    return pd.DataFrame(rows)


def run_stage1(
    rv_series: Optional[pd.Series] = None,
    *,
    project_root: Path | str = ".",
    n_days: int = 10,
    window: Optional[int] = None,
    trials: int = 100,
    parallel: bool = False,
    processes: Optional[int] = None,
    save_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    project_root = Path(project_root).resolve()
    source = "provided_series"
    if rv_series is None:
        rv_series, source = load_rv_series(project_root / "data" / "rv_series.csv")
    train_end_idx, n_test = validate_split(rv_series)
    print("=" * 88)
    print("STAGE 1: 10-DAY ROLLING CEEMDAN VALIDATION")
    print("=" * 88)
    print(f"Project root     : {project_root}")
    print(f"RV source        : {source}")
    print(f"Series length    : {len(rv_series)}")
    print(f"Train/test split : {train_end_idx} / {n_test}")
    print(f"Test start       : {rv_series.index[train_end_idx].date()}")
    print(f"Rolling window   : {'all history' if window is None else window}")
    print(f"CEEMDAN trials   : {trials}")
    print(f"Parallel         : {parallel}")
    print()
    aligned, metadata = rolling_ceemdan(
        rv_series=rv_series,
        train_end_idx=train_end_idx,
        window=window,
        max_steps=n_days,
        trials=trials,
        parallel=parallel,
        processes=processes,
        verbose=True,
    )
    frame = stage1_metadata_frame(metadata)
    print("\nMini result table")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(frame.to_string(index=False, float_format=lambda value: f"{value:.6e}"))
    print("\nIMF-count distribution:", metadata["imf_count_distribution"])
    print(f"Average decomposition time: {metadata['average_elapsed_sec']:.2f} s")
    print(f"Total stage-1 time       : {metadata['total_elapsed_sec'] / 60.0:.2f} min")
    print(f"Aligned cube shape       : {aligned.shape}")
    suspicious = frame[
        (frame["n_imfs"] < 3) | frame["error"].notna() | (frame["elapsed_sec"] > 30.0)
    ]
    if suspicious.empty:
        print("Anomaly check             : OK (no failed, short-IMF, or >30s step)")
    else:
        print("Anomaly check             : review required")
        print(suspicious.to_string(index=False))
    payload = {
        "stage": 1,
        "n_days": int(n_days),
        "source": source,
        "rv_series": rv_series,
        "train_end_idx": int(train_end_idx),
        "n_test": int(n_test),
        "imfs_rolling": aligned,
        "metadata": metadata,
        "metadata_frame": frame,
    }
    output = Path(save_path) if save_path is not None else project_root / "results" / "rolling_stage1.pkl"
    save_pickle(payload, output)
    metadata_csv = output.with_name("rolling_stage1_metadata.csv")
    frame.to_csv(metadata_csv, index=False)
    print(f"\nSaved pickle             : {output}")
    print(f"Saved metadata CSV       : {metadata_csv}")
    print("STOP: wait for confirmation before running stage 2.")
    return payload
