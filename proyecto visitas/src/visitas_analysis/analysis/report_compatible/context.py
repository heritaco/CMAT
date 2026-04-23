from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from visitas_analysis.analysis.cleaning import clean_materias_df


COMPARISON_SALON_KEY = (23453, "MAT1012", 2020, "PRIMAVERA")
OUTLIER_SALON_KEY = (22115, "MAT1052", 2024, "PRIMAVERA")


@dataclass
class BaseContext:
    materias: pd.DataFrame
    profes_value_counts: pd.Series
    profes_ids: np.ndarray
    colors: list
    viridis_by_prof: list
    magma_by_prof: list
    threshold: float
    mean_below: float
    mean_above: float


@dataclass
class VisitsContext:
    materias: pd.DataFrame
    asesoria_counts: pd.Series


def load_base_context(materias_path: Path) -> BaseContext:
    materias = pd.read_excel(materias_path)
    materias = clean_materias_df(materias)

    profes_value_counts = materias["CLAVEPROFESOR"].value_counts()
    profes_ids = materias["CLAVEPROFESOR"].unique()

    scaler = MinMaxScaler()
    counts = profes_value_counts.values.reshape(-1, 1)
    scaled_counts = scaler.fit_transform(counts)

    colors = [
        sns.color_palette("viridis_r", 100)[int(scaled_counts[i][0] * 99)]
        for i in range(len(scaled_counts))
    ]
    viridis_by_prof = [
        sns.color_palette("Blues_r", 100)[int(scaled_counts[i][0] * 50)]
        for i in range(len(scaled_counts))
    ]
    magma_by_prof = [
        sns.color_palette("Reds_r", 100)[int(scaled_counts[i][0] * 50)]
        for i in range(len(scaled_counts))
    ]

    threshold = 7.5
    below_threshold_percentages = []
    above_threshold_percentages = []
    for num in range(len(profes_ids)):
        profe = materias[materias["CLAVEPROFESOR"] == profes_ids[num]]
        profe_num = profe[pd.to_numeric(profe["CALIFICACION"], errors="coerce").notnull()]
        calificaciones = profe_num["CALIFICACION"].astype(float)
        total_count = len(calificaciones)
        if total_count == 0:
            below_threshold_percentages.append(0)
            above_threshold_percentages.append(0)
            continue
        below_count = (calificaciones < threshold).sum()
        above_count = (calificaciones >= threshold).sum()
        below_threshold_percentages.append(below_count / total_count)
        above_threshold_percentages.append(above_count / total_count)

    return BaseContext(
        materias=materias,
        profes_value_counts=profes_value_counts,
        profes_ids=profes_ids,
        colors=colors,
        viridis_by_prof=viridis_by_prof,
        magma_by_prof=magma_by_prof,
        threshold=threshold,
        mean_below=float(np.mean(below_threshold_percentages)),
        mean_above=float(np.mean(above_threshold_percentages)),
    )


def load_visits_context(asesorias_path: Path, materias: pd.DataFrame) -> VisitsContext:
    asesorias = pd.read_excel(asesorias_path)
    asesoria_counts = asesorias["id"].value_counts()
    asesoria_counts = asesoria_counts.reindex(materias["CLAVEALUMNO"].unique(), fill_value=0)

    materias = materias.copy()
    materias["VISITAS"] = materias["CLAVEALUMNO"].map(asesoria_counts)
    materias["VISITAS"] = materias["VISITAS"].fillna(0).astype(int)
    return VisitsContext(materias=materias, asesoria_counts=asesoria_counts)
