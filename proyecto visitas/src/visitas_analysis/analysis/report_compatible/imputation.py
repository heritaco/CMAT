from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SalonKey = tuple[int, str, int, str]


@dataclass
class StageContext:
    salones: dict[SalonKey, pd.DataFrame]
    ultramerge: pd.DataFrame
    ultramerge_means: pd.DataFrame | None = None


def _add_mean_imputation(sesion_materias: pd.DataFrame) -> pd.DataFrame:
    sesion_materias = sesion_materias.copy()
    numeric_calificaciones = pd.to_numeric(sesion_materias["CALIFICACION"], errors="coerce")
    mean_calificacion = numeric_calificaciones[numeric_calificaciones <= 7.5].mean()
    sesion_materias["IMPMEAN"] = numeric_calificaciones.fillna(mean_calificacion)
    mean = sesion_materias["IMPMEAN"].mean()
    std = sesion_materias["IMPMEAN"].std()
    sesion_materias["IMPMEAN_Z"] = (sesion_materias["IMPMEAN"] - mean) / std
    return sesion_materias


def _silverman_bandwidth(sample):
    sample = np.asarray(sample, dtype=float)
    n = len(sample)
    if n < 2:
        return 0.1
    sd = np.std(sample, ddof=1)
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    scale = min(sd, iqr / 1.34) if iqr > 0 else sd
    bandwidth = 0.9 * scale * n ** (-1.0 / 5.0)
    return max(bandwidth, 1e-3)


def _sample_kde_truncated(x_obs, size, low=0.0, high=7.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x_obs = np.asarray(x_obs, dtype=float)
    bandwidth = _silverman_bandwidth(x_obs)
    out = np.empty(size, dtype=float)
    index = 0
    while index < size:
        batch = size - index
        drawn_idx = rng.integers(0, len(x_obs), size=batch)
        eps = rng.normal(0.0, bandwidth, size=batch)
        draw = x_obs[drawn_idx] + eps
        ok = (draw >= low) & (draw <= high)
        kept = draw[ok]
        take = min(len(kept), batch)
        if take:
            out[index:index + take] = kept[:take]
            index += take
    return out


def _sample_empirical(x_obs, size, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x_obs = np.asarray(x_obs, dtype=float)
    return rng.choice(x_obs, size=size, replace=True)


def legacy_impute_nans_from_pre75_kde_df(
    df,
    value_col="CALIFICACION",
    out_col="IMPKDE",
    low=0.0,
    high=7.5,
    min_kde_n=20,
    seed=42,
):
    rng = np.random.default_rng(seed)
    df = df.copy()

    num = pd.to_numeric(df[value_col], errors="coerce")
    df[out_col] = num
    nan_mask = num.isna()
    need = int(nan_mask.sum())
    if need == 0:
        return df

    obs_pre = num[~num.isna() & (num <= high)]
    if len(obs_pre) >= min_kde_n:
        draws = _sample_kde_truncated(obs_pre.values, need, low=low, high=high, rng=rng)
    elif len(obs_pre) > 0:
        draws = _sample_empirical(obs_pre.values, need, rng=rng)
        draws = np.clip(draws, low, high)
    else:
        draws = np.full(need, min(6.5, high), dtype=float)

    df.loc[nan_mask, out_col] = draws
    df.loc[nan_mask, out_col] = df.loc[nan_mask, out_col].clip(lower=low, upper=high)
    return df


def corrected_impute_nans_from_pre75_kde_df(
    df,
    value_col="CALIFICACION",
    out_col="IMPKDE",
    low=0.0,
    high=7.4,
    min_kde_n=20,
    seed=42,
    fallback="uniform",
    constant_value=6.5,
    global_source=None,
):
    rng = np.random.default_rng(seed)
    df = df.copy()

    num = pd.to_numeric(df[value_col], errors="coerce")
    df[out_col] = num
    nan_mask = num.isna()
    need = int(nan_mask.sum())
    if need == 0:
        return df

    obs_pre = num[~num.isna() & (num <= high)]

    def _borrow_pool():
        src = pd.to_numeric(global_source, errors="coerce") if global_source is not None else num
        return src[(~src.isna()) & (src <= high)]

    if len(obs_pre) >= min_kde_n:
        draws = _sample_kde_truncated(obs_pre.values, need, low=low, high=high, rng=rng)
    elif 0 < len(obs_pre) < min_kde_n:
        draws = np.clip(_sample_empirical(obs_pre.values, need, rng=rng), low, high)
    else:
        if fallback == "global":
            pool = _borrow_pool()
            if len(pool) >= min_kde_n:
                draws = _sample_kde_truncated(pool.values, need, low=low, high=high, rng=rng)
            elif len(pool) > 0:
                draws = np.clip(_sample_empirical(pool.values, need, rng=rng), low, high)
            else:
                draws = np.full(need, min(constant_value, high))
        elif fallback == "parametric":
            obs_all = num[~num.isna()]
            mu = float(obs_all.mean()) if len(obs_all) else 6.5
            sd = float(obs_all.std(ddof=1)) if len(obs_all) > 1 else 0.5
            sd = max(sd, 1e-3)
            index, draws = 0, np.empty(need)
            while index < need:
                batch = need - index
                cand = rng.normal(mu, sd, size=batch)
                ok = (cand >= low) & (cand <= high)
                take = min(ok.sum(), batch)
                if take:
                    draws[index:index + take] = cand[ok][:take]
                    index += take
        elif fallback == "uniform":
            ks = np.arange(1, need + 1, dtype=float)
            draws = low + (ks / (need + 1.0)) * (high - low)
        elif fallback == "leave":
            return df
        elif fallback == "constant":
            draws = np.full(need, min(constant_value, high))
        else:
            raise ValueError("Unknown fallback")

    df.loc[nan_mask, out_col] = draws
    df.loc[nan_mask, out_col] = df.loc[nan_mask, out_col].clip(lower=low, upper=high)
    return df


def build_salones_mean_only(materias: pd.DataFrame, profes_ids) -> dict[SalonKey, pd.DataFrame]:
    salones: dict[SalonKey, pd.DataFrame] = {}
    for prof_id in profes_ids:
        profe_materias = materias[materias["CLAVEPROFESOR"] == prof_id]
        for materia in profe_materias["CLAVEVARIANTEMATERIA"].unique():
            materia_materias = profe_materias[profe_materias["CLAVEVARIANTEMATERIA"] == materia]
            for anio in materia_materias["anio"].unique():
                anio_materias = materia_materias[materia_materias["anio"] == anio]
                for sesion in anio_materias["CLAVESESION"].unique():
                    sesion_materias = anio_materias[anio_materias["CLAVESESION"] == sesion]
                    salones[(prof_id, materia, anio, sesion)] = _add_mean_imputation(sesion_materias)
    return salones


def build_salones_with_imputer(
    materias: pd.DataFrame,
    profes_ids,
    imputer,
    *,
    imputer_kwargs: dict | None = None,
) -> dict[SalonKey, pd.DataFrame]:
    salones: dict[SalonKey, pd.DataFrame] = {}
    imputer_kwargs = {} if imputer_kwargs is None else dict(imputer_kwargs)
    for prof_id in profes_ids:
        profe_materias = materias[materias["CLAVEPROFESOR"] == prof_id]
        for materia in profe_materias["CLAVEVARIANTEMATERIA"].unique():
            materia_materias = profe_materias[profe_materias["CLAVEVARIANTEMATERIA"] == materia]
            for anio in materia_materias["anio"].unique():
                anio_materias = materia_materias[materia_materias["anio"] == anio]
                for sesion in anio_materias["CLAVESESION"].unique():
                    sesion_materias = anio_materias[anio_materias["CLAVESESION"] == sesion]
                    sesion_materias = _add_mean_imputation(sesion_materias)
                    sesion_materias = imputer(
                        sesion_materias,
                        value_col="CALIFICACION",
                        out_col="IMPKDE",
                        **imputer_kwargs,
                    )
                    mean = sesion_materias["IMPKDE"].mean()
                    std = sesion_materias["IMPKDE"].std()
                    sesion_materias["IMPKDE_Z"] = (sesion_materias["IMPKDE"] - mean) / std
                    salones[(prof_id, materia, anio, sesion)] = sesion_materias
    return salones


def concat_salones(salones: dict[SalonKey, pd.DataFrame]) -> pd.DataFrame:
    ultramerge = pd.concat(salones.values(), ignore_index=True)
    ultramerge["CALIFICACION"] = pd.to_numeric(ultramerge["CALIFICACION"], errors="coerce")
    return ultramerge


def compute_ultramerge_means(ultramerge: pd.DataFrame, materias: pd.DataFrame) -> pd.DataFrame:
    ultramerge_means = ultramerge.groupby("CLAVEALUMNO")["IMPKDE_Z"].mean().reset_index()
    ultramerge_means.rename(columns={"IMPKDE_Z": "MEAN_IMPKDE_Z"}, inplace=True)
    ultramerge_means = ultramerge_means.merge(
        materias[["CLAVEALUMNO", "VISITAS"]].drop_duplicates(),
        on="CLAVEALUMNO",
        how="left",
    )
    return ultramerge_means
