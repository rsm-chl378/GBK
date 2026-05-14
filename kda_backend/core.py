from __future__ import annotations

import warnings as py_warnings

import numpy as np
import pandas as pd

from .applicability import ALL_METHODS, method_applicability
from .methods import METHOD_REGISTRY, normalize_scores, rank_desc
from .plotting import driver_bar_chart
from .preprocessing import complete_cases, detect_var_type
from .schemas import KDAResult, MethodResult


MIN_ROWS = 10
VALID_Y_TYPES = {"continuous", "binary", "ordered", "nominal", "unknown"}


def _minimum_required_rows(x_vars: list[str], controls: list[str]) -> int:
    return max(MIN_ROWS, len(x_vars) + len(controls) + 2)


def _validate_inputs(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None,
    subgroup: str | None,
    y_type_override: str | None = None,
) -> None:
    controls = controls or []
    needed = [y_var, *x_vars, *controls]
    if subgroup:
        needed.append(subgroup)
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise ValueError(f"Variables not found in data: {', '.join(missing)}")

    unknown = [method for method in methods if method not in ALL_METHODS]
    if unknown:
        raise ValueError(f"Unknown methods: {', '.join(unknown)}")

    if len(set(x_vars)) != len(x_vars):
        raise ValueError("x_vars contains duplicate variables.")
    overlap = sorted(set(x_vars).intersection(controls))
    if overlap:
        raise ValueError(f"Variables cannot be both drivers and controls: {', '.join(overlap)}")
    if y_type_override is not None and y_type_override not in VALID_Y_TYPES:
        valid = ", ".join(sorted(VALID_Y_TYPES))
        raise ValueError(f"Invalid y_type_override '{y_type_override}'. Expected one of: {valid}.")


def _nan_method(method: str, x_vars: list[str], warning: str) -> MethodResult:
    return MethodResult(
        pd.Series(np.nan, index=x_vars, dtype=float),
        {"applicable": False},
        [warning],
    )


def _assemble_tables(
    x_vars: list[str],
    methods: list[str],
    method_scores: dict[str, pd.Series],
    method_warnings: dict[str, list[str]],
    method_intervals: dict[str, tuple[pd.Series, pd.Series]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    method_intervals = method_intervals or {}
    importance = pd.DataFrame({"driver": x_vars})
    for method in methods:
        scores = method_scores[method].reindex(x_vars)
        importance[method] = scores.to_numpy()
        if method in method_intervals:
            lower, upper = method_intervals[method]
            importance[f"{method}_ci_lower"] = lower.reindex(x_vars).to_numpy()
            importance[f"{method}_ci_upper"] = upper.reindex(x_vars).to_numpy()
        importance[f"{method}_index"] = normalize_scores(scores).to_numpy()
        importance[f"{method}_rank"] = rank_desc(scores).to_numpy()
        warnings = method_warnings.get(method, [])
        importance[f"{method}_warning"] = "; ".join(warnings) if warnings else ""

    index_cols = [f"{method}_index" for method in methods]
    rank_cols = [f"{method}_rank" for method in methods]
    importance["mean_method_index"] = importance[index_cols].mean(axis=1, skipna=True)
    importance["average_rank"] = importance[rank_cols].mean(axis=1, skipna=True)
    importance["median_rank"] = importance[rank_cols].median(axis=1, skipna=True)
    importance["top3_appearances"] = (importance[rank_cols] <= 3).sum(axis=1)

    ranking = importance[
        [
            "driver",
            "average_rank",
            "median_rank",
            "mean_method_index",
            "top3_appearances",
            *rank_cols,
        ]
    ].copy()
    ranking = ranking.sort_values(
        ["average_rank", "mean_method_index", "driver"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "overall_rank", np.arange(1, len(ranking) + 1))
    return importance, ranking


def _diagnostics(
    *,
    rows_input: int,
    rows_used: int,
    y_var: str,
    y_type: str,
    x_vars: list[str],
    controls: list[str],
    methods: list[str],
    subgroup: str | None,
    method_metadata: dict[str, dict],
) -> pd.DataFrame:
    rows = [
        ("rows_input", rows_input),
        ("rows_used", rows_used),
        ("rows_dropped_missing", rows_input - rows_used),
        ("y_var", y_var),
        ("y_type", y_type),
        ("x_vars", ", ".join(x_vars)),
        ("controls", ", ".join(controls)),
        ("methods", ", ".join(methods)),
        ("subgroup", subgroup or ""),
    ]
    for method, metadata in method_metadata.items():
        if "model_type" in metadata:
            rows.append((f"{method}_model_type", metadata["model_type"]))
        if "train_score" in metadata:
            rows.append((f"{method}_train_score", metadata["train_score"]))
    return pd.DataFrame(rows, columns=["metric", "value"])


def _bootstrap_intervals(
    model_data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    controls: list[str],
    y_type: str,
    methods: list[str],
    method_scores: dict[str, pd.Series],
    method_params: dict,
    bootstrap_methods: list[str] | None,
    bootstrap_params: dict | None,
) -> tuple[dict[str, tuple[pd.Series, pd.Series]], list[str]]:
    if not bootstrap_methods:
        return {}, []

    params = {
        "n_resamples": 200,
        "ci": 0.95,
        "random_state": 454,
        "min_rows": MIN_ROWS,
    }
    params.update(bootstrap_params or {})
    n_resamples = int(params["n_resamples"])
    ci = float(params["ci"])
    min_rows = int(params["min_rows"])
    alpha = (1 - ci) / 2
    rng = np.random.default_rng(int(params["random_state"]))

    intervals: dict[str, tuple[pd.Series, pd.Series]] = {}
    warnings: list[str] = []
    selected = set(methods)
    for method in bootstrap_methods:
        if method not in selected:
            warnings.append(f"{method} bootstrap skipped: method was not selected for the main analysis.")
            continue

        applicability = method_applicability(method, y_type)
        if not applicability.applicable:
            warnings.append(f"{method} bootstrap skipped: {applicability.warning or 'method is not applicable.'}")
            continue

        scores_by_driver = {driver: [] for driver in x_vars}
        for _ in range(n_resamples):
            sample_idx = rng.integers(0, len(model_data), size=len(model_data))
            sample = model_data.iloc[sample_idx].reset_index(drop=True)
            sample_complete = complete_cases(sample, y_var, x_vars, controls)
            if len(sample_complete) < min_rows:
                continue
            try:
                with py_warnings.catch_warnings():
                    py_warnings.simplefilter("ignore")
                    result = METHOD_REGISTRY[method](
                        sample_complete,
                        y_var=y_var,
                        x_vars=x_vars,
                        controls=controls,
                        y_type=y_type,
                        params=method_params.get(method, {}),
                    )
            except Exception:
                continue

            sample_scores = result.scores.reindex(x_vars)
            for driver, value in sample_scores.items():
                if np.isfinite(value):
                    scores_by_driver[driver].append(float(value))

        lower = pd.Series(np.nan, index=x_vars, dtype=float)
        upper = pd.Series(np.nan, index=x_vars, dtype=float)
        min_valid = int(params.get("min_valid_resamples", max(10, int(n_resamples * 0.2))))
        for driver, values in scores_by_driver.items():
            if len(values) < min_valid:
                continue
            lo = float(np.quantile(values, alpha))
            hi = float(np.quantile(values, 1 - alpha))
            score = method_scores[method].reindex(x_vars).loc[driver]
            if np.isfinite(score):
                lo = min(lo, float(score))
                hi = max(hi, float(score))
            lower.loc[driver] = lo
            upper.loc[driver] = hi

        if lower.notna().any() and upper.notna().any():
            intervals[method] = (lower, upper)
        else:
            warnings.append(
                f"{method} bootstrap skipped: insufficient valid bootstrap samples after {n_resamples} resamples."
            )

    return intervals, warnings


def run_kda(
    data: pd.DataFrame,
    y_var: str,
    x_vars: list[str],
    methods: list[str],
    controls: list[str] | None = None,
    subgroup: str | None = None,
    method_params: dict | None = None,
    bootstrap_methods: list[str] | None = None,
    bootstrap_params: dict | None = None,
    y_type_override: str | None = None,
    _run_subgroups: bool = True,
) -> KDAResult:
    controls = controls or []
    method_params = method_params or {}
    _validate_inputs(data, y_var, x_vars, methods, controls, subgroup, y_type_override)

    subgroup_results: dict[str, KDAResult] | None = None
    subgroup_summary: pd.DataFrame | None = None
    warnings: list[str] = []
    if subgroup and _run_subgroups:
        subgroup_results = {}
        subgroup_rows = []
        min_required_rows = _minimum_required_rows(x_vars, controls)
        for level, group in data.dropna(subset=[subgroup]).groupby(subgroup, sort=True):
            complete = complete_cases(group, y_var, x_vars, controls)
            if len(complete) < min_required_rows:
                reason = f"only {len(complete)} complete rows; at least {min_required_rows} are required."
                warnings.append(f"Skipped subgroup {level}: {reason}")
                subgroup_rows.append(
                    {
                        "subgroup_variable": subgroup,
                        "subgroup_level": str(level),
                        "rows_used": len(complete),
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            try:
                subgroup_results[str(level)] = run_kda(
                    group,
                    y_var,
                    x_vars,
                    methods,
                    controls=controls,
                    subgroup=None,
                    method_params=method_params,
                    bootstrap_methods=bootstrap_methods,
                    bootstrap_params=bootstrap_params,
                    y_type_override=y_type_override,
                    _run_subgroups=False,
                )
            except Exception as exc:
                reason = str(exc)
                warnings.append(f"Skipped subgroup {level}: {reason}")
                subgroup_rows.append(
                    {
                        "subgroup_variable": subgroup,
                        "subgroup_level": str(level),
                        "rows_used": len(complete),
                        "status": "skipped",
                        "reason": reason,
                    }
                )
                continue
            subgroup_rows.append(
                {
                    "subgroup_variable": subgroup,
                    "subgroup_level": str(level),
                    "rows_used": len(complete),
                    "status": "included",
                    "reason": "",
                }
            )
        subgroup_summary = pd.DataFrame(
            subgroup_rows,
            columns=["subgroup_variable", "subgroup_level", "rows_used", "status", "reason"],
        )

    model_data = complete_cases(data, y_var, x_vars, controls)
    min_required_rows = _minimum_required_rows(x_vars, controls)
    if len(model_data) < min_required_rows:
        raise ValueError(
            "Insufficient complete rows for KDA analysis: "
            f"{len(model_data)} rows after dropping missing values; at least {min_required_rows} are required."
        )

    y_type = y_type_override or detect_var_type(model_data[y_var], role="y")
    method_scores: dict[str, pd.Series] = {}
    method_metadata: dict[str, dict] = {}
    method_warnings: dict[str, list[str]] = {}

    for method in methods:
        applicability = method_applicability(method, y_type)
        if not applicability.applicable:
            result = _nan_method(method, x_vars, applicability.warning or f"{method} is not applicable.")
        else:
            params = method_params.get(method, {})
            try:
                result = METHOD_REGISTRY[method](
                    model_data,
                    y_var=y_var,
                    x_vars=x_vars,
                    controls=controls,
                    y_type=y_type,
                    params=params,
                )
            except Exception as exc:
                result = _nan_method(method, x_vars, f"{method} failed: {exc}")

        metadata = {"applicable": applicability.applicable, "model_hint": applicability.model_hint}
        metadata.update(result.metadata)
        method_scores[method] = result.scores.reindex(x_vars)
        method_metadata[method] = metadata
        method_warnings[method] = result.warnings
        warnings.extend([f"{method}: {warning}" for warning in result.warnings])

    method_intervals, bootstrap_warnings = _bootstrap_intervals(
        model_data=model_data,
        y_var=y_var,
        x_vars=x_vars,
        controls=controls,
        y_type=y_type,
        methods=methods,
        method_scores=method_scores,
        method_params=method_params,
        bootstrap_methods=bootstrap_methods,
        bootstrap_params=bootstrap_params,
    )
    warnings.extend(bootstrap_warnings)

    importance, ranking = _assemble_tables(x_vars, methods, method_scores, method_warnings, method_intervals)
    diagnostics = _diagnostics(
        rows_input=len(data),
        rows_used=len(model_data),
        y_var=y_var,
        y_type=y_type,
        x_vars=x_vars,
        controls=controls,
        methods=methods,
        subgroup=subgroup,
        method_metadata=method_metadata,
    )
    chart = driver_bar_chart(ranking)
    return KDAResult(
        importance_table=importance,
        ranking_table=ranking,
        diagnostics=diagnostics,
        bar_chart=chart,
        subgroup_results=subgroup_results,
        subgroup_summary=subgroup_summary,
        method_metadata=method_metadata,
        warnings=warnings,
    )
