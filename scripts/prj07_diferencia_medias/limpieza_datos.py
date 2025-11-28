import numpy as np
import pandas as pd

def clean_materias_df(materias: pd.DataFrame) -> pd.DataFrame:
    # IF CLAVEPROFESOR is nan and 'CLAVEALUMNO', 'CLAVEVARIANTEMATERIA', 'CALIFICACION' are the same, drop that row
    materias.sort_values(by=['CLAVEALUMNO', 'CLAVEVARIANTEMATERIA', 'CALIFICACION', 'CLAVEPROFESOR'], inplace=True)
    materias.drop_duplicates(subset=['CLAVEALUMNO', 'CLAVEVARIANTEMATERIA', 'CALIFICACION'], keep='first', inplace=True, ignore_index=True)

    # drop NUMORDEN column
    materias.drop(columns=['NUMORDEN'], inplace=True)

    # drop the rows where CLAVEPROFESOR is nan
    materias.dropna(subset=['CLAVEPROFESOR'], inplace=True)
    # make the CLAVEPROFESOR column integer
    materias['CLAVEPROFESOR'] = materias['CLAVEPROFESOR'].astype(int)
    return materias

def _silverman_bandwidth(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2: return 0.1
    sd = np.std(x, ddof=1)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    s = min(sd, iqr/1.34) if iqr > 0 else sd
    h = 0.9 * s * n**(-1/5)
    return max(h, 1e-3)

def _sample_kde_truncated(x_obs, size, low=0.0, high=7.5, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x_obs = np.asarray(x_obs, dtype=float)
    h = _silverman_bandwidth(x_obs)
    out = np.empty(size, dtype=float); k = 0
    while k < size:
        batch = size - k
        idx = rng.integers(0, len(x_obs), size=batch)
        eps = rng.normal(0.0, h, size=batch)
        draw = x_obs[idx] + eps
        ok = (draw >= low) & (draw <= high)
        take = min(ok.sum(), batch)
        if take:
            out[k:k+take] = draw[ok][:take]
            k += take
    return out

def _sample_empirical(x_obs, size, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x_obs = np.asarray(x_obs, dtype=float)
    return rng.choice(x_obs, size=size, replace=True)

def impute_nans_from_pre75_kde_df(
    df, value_col='CALIFICACION', out_col='IMPKDE',
    low=0.0, high=7.4, min_kde_n=20, seed=42,
    fallback='uniform',           # 'constant' | 'global' | 'parametric' | 'leave' | 'uniform'
    constant_value=6.5,
    prefer_empirical_if_small=True,
    global_source=None,
    debug=False
):
    rng = np.random.default_rng(seed)
    df = df.copy()

    num = pd.to_numeric(df[value_col], errors='coerce')
    df[out_col] = num
    nan_mask = num.isna()
    need = int(nan_mask.sum())
    if need == 0:
        return df

    obs_pre = num[~num.isna() & (num <= high)]

    def _borrow_pool():
        src = pd.to_numeric(global_source, errors='coerce') if global_source is not None else num
        return src[(~src.isna()) & (src <= high)]

    if len(obs_pre) >= min_kde_n:
        draws = _sample_kde_truncated(obs_pre.values, need, low=low, high=high, rng=rng)
    elif 0 < len(obs_pre) < min_kde_n:
        draws = np.clip(_sample_empirical(obs_pre.values, need, rng=rng), low, high)
    else:
        # No mass in [low, high]; choose fallback
        if fallback == 'global':
            pool = _borrow_pool()
            if len(pool) >= min_kde_n:
                draws = _sample_kde_truncated(pool.values, need, low=low, high=high, rng=rng)
            elif len(pool) > 0:
                draws = np.clip(_sample_empirical(pool.values, need, rng=rng), low, high)
            else:
                draws = np.full(need, min(constant_value, high))
        elif fallback == 'parametric':
            obs_all = num[~num.isna()]
            mu = float(obs_all.mean()) if len(obs_all) else 6.5
            sd = float(obs_all.std(ddof=1)) if len(obs_all) > 1 else 0.5
            sd = max(sd, 1e-3)
            k, draws = 0, np.empty(need)
            while k < need:
                batch = need - k
                cand = rng.normal(mu, sd, size=batch)
                ok = (cand >= low) & (cand <= high)
                take = min(ok.sum(), batch)
                if take:
                    draws[k:k+take] = cand[ok][:take]
                    k += take
        elif fallback == 'uniform':
            # Equally spaced interior points in [low, high]
            # order matches the order of NaNs in the DataFrame
            ks = np.arange(1, need + 1, dtype=float)
            draws = low + (ks / (need + 1.0)) * (high - low)
        elif fallback == 'leave':
            return df
        elif fallback == 'constant':
            draws = np.full(need, min(constant_value, high))
        else:
            raise ValueError("Unknown fallback")

    df.loc[nan_mask, out_col] = draws
    # hard bounds only on imputed entries
    vals = df.loc[nan_mask, out_col].to_numpy(dtype=float, copy=False)
    np.clip(vals, low, high, out=vals)
    df.loc[nan_mask, out_col] = vals
    return df



def get_salones_with_imputations(materias):
    salones = {}

    # group by the 4 keys instead of nested loops
    grouped = materias.groupby(
        ['CLAVEPROFESOR', 'CLAVEVARIANTEMATERIA', 'anio', 'CLAVESESION'],
        sort=False
    )

    for (prof_id, materia, anio, sesion), sesion_materias in grouped:
        # IMPORTANT: work on a copy so assignments are safe
        sesion_materias = sesion_materias.copy()

        # 1) mean-imputation truncated at 7.5
        numeric_calificaciones = pd.to_numeric(
            sesion_materias['CALIFICACION'], errors='coerce'
        )
        mean_calificacion = numeric_calificaciones[
            numeric_calificaciones <= 7.5
        ].mean()

        sesion_materias.loc[:, 'IMPMEAN'] = numeric_calificaciones.fillna(
            mean_calificacion
        )

        mean = sesion_materias['IMPMEAN'].mean()
        std = sesion_materias['IMPMEAN'].std()

        # avoid division by zero just in case
        if std == 0 or pd.isna(std):
            sesion_materias.loc[:, 'IMPMEAN_Z'] = 0.0
        else:
            sesion_materias.loc[:, 'IMPMEAN_Z'] = (
                sesion_materias['IMPMEAN'] - mean
            ) / std

        # 2) KDE-based imputation
        sesion_materias = impute_nans_from_pre75_kde_df(
            sesion_materias,
            value_col='CALIFICACION',
            out_col='IMPKDE',
            constant_value=6.5,
            fallback='uniform',
        )

        mean_kde = sesion_materias['IMPKDE'].mean()
        std_kde = sesion_materias['IMPKDE'].std()

        if std_kde == 0 or pd.isna(std_kde):
            sesion_materias.loc[:, 'IMPKDE_Z'] = 0.0
        else:
            sesion_materias.loc[:, 'IMPKDE_Z'] = (
                sesion_materias['IMPKDE'] - mean_kde
            ) / std_kde

        # store the processed DataFrame
        salones[(prof_id, materia, anio, sesion)] = sesion_materias

    return salones


def limpieza_datos():

    materias = pd.read_excel('data/onedrive/Archivos2024/Materias estudiantes-profesores 2019-2025 P y O.xlsx')
    materias_copy = materias.copy()

    materias = clean_materias_df(materias)
    
    asesorias = pd.read_excel('data/onedrive/Archivos2024/Asesorias2024.xlsx')

    # count how many times each id appears in 'id' column
    asesoria_counts = asesorias['id'].value_counts()

    # if id in materias['CLAVEALUMNO'] but not in asesoria_counts, assign 0 in materias_counts
    asesoria_counts = asesoria_counts.reindex(materias['CLAVEALUMNO'].unique(), fill_value=0)


    # if the id of asesorias_counts is in materias['CLAVEALUMNO'], map the count to a new column 'VISITAS' in materias
    materias['VISITAS'] = materias['CLAVEALUMNO'].map(asesoria_counts)

    # add the timmes each id appears as a new column 'asesoria_count' in asesorias
    materias['VISITAS'] = materias['VISITAS'].fillna(0).astype(int)

    salones = get_salones_with_imputations(materias)

    ultramerge = pd.concat(salones.values(), ignore_index=True)
    ultramerge['CALIFICACION'] = pd.to_numeric(ultramerge['CALIFICACION'], errors='coerce')

    # get the mean of each CLAVEALUMNO in ultramerge
    ultramerge_means = ultramerge.groupby('CLAVEALUMNO')['IMPKDE_Z'].mean().reset_index()
    ultramerge_means.rename(columns={'IMPKDE_Z': 'MEAN_IMPKDE_Z'}, inplace=True)
    ultramerge_means = ultramerge_means.merge(materias[['CLAVEALUMNO', 'VISITAS']].drop_duplicates(), on='CLAVEALUMNO', how='left')
    ultramerge['CALIFICACION'] = pd.to_numeric(ultramerge['CALIFICACION'], errors='coerce')
    # ultramerge_means.sort_values(by='MEAN_IMPKDE_Z', ascending=False).head(10)

    return ultramerge, ultramerge_means