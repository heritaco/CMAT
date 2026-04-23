from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config.settings import get_settings
from student_cluster_analysis.analytics.paradoxical_group import compute_discrepancy_score, run_paradoxical_group_analysis
from student_cluster_analysis.clustering.target_cluster import select_target_cluster
from student_cluster_analysis.entities import ClusterCandidate
from student_cluster_analysis.pipeline.main_pipeline import run_student_cluster_pipeline
from student_cluster_analysis.preprocessing.cleaning import clean_materias_dataframe
from student_cluster_analysis.preprocessing.merging import build_merged_dataset


def _build_temp_inputs(tmp_path: Path) -> tuple[Path, Path]:
    materias_path = tmp_path / "materias.xlsx"
    examenes_path = tmp_path / "resultados_examenes.xlsx"

    materias_rows = []
    for student_id in range(1, 13):
        materias_rows.append(
            {
                "CLAVEALUMNO": student_id,
                "CLAVECARRERA": "LAD",
                "anio": 2024,
                "CLAVESESION": "PRIMAVERA",
                "NUMORDEN": 0,
                "CLAVEVARIANTEMATERIA": "MAT1012",
                "DESCRIBEMATERIA": "MATEMATICAS UNIVERSITARIAS",
                "CALIFICACION": 9.0 if student_id <= 6 else 8.0,
                "CLAVEPROFESOR": 111 if student_id <= 6 else 222,
            }
        )
    materias_df = pd.DataFrame(materias_rows)

    dmu_df = pd.DataFrame(
        {
            "ID": list(range(1, 13)),
            "Año": [2024] * 12,
            "Apellidos": ["X"] * 12,
            "Nombre": ["Y"] * 12,
            "Tipo_de_examen": ["DMU"] * 12,
            "Porcentaje_DMU": [20] * 6 + [70] * 6,
        }
    )
    gagb_df = pd.DataFrame(
        {
            "ID": list(range(1, 13)),
            "Año": [2024] * 12,
            "Apellidos": ["X"] * 12,
            "Nombre": ["Y"] * 12,
            "Tipo_de_examen": ["GA_GB"] * 12,
            "Porcentaje_GA_GB": [25] * 6 + [75] * 6,
        }
    )

    materias_df.to_excel(materias_path, index=False)
    with pd.ExcelWriter(examenes_path, engine="openpyxl") as writer:
        dmu_df.to_excel(writer, sheet_name="DMU", index=False)
        gagb_df.to_excel(writer, sheet_name="GA-GB", index=False)
    return materias_path, examenes_path


def test_required_column_validation_raises_on_missing_columns() -> None:
    broken_df = pd.DataFrame({"CLAVEALUMNO": [1], "anio": [2024]})
    with pytest.raises(ValueError):
        clean_materias_dataframe(broken_df)


def test_merge_uses_nearest_year_fallback() -> None:
    materias = pd.DataFrame(
        {
            "CLAVEALUMNO": pd.Series([10], dtype="Int64"),
            "CLAVECARRERA": ["LAD"],
            "anio": pd.Series([2025], dtype="Int64"),
            "CLAVESESION": ["PRIMAVERA"],
            "NUMORDEN": pd.Series([0], dtype="Int64"),
            "CLAVEVARIANTEMATERIA": ["MAT1012"],
            "DESCRIBEMATERIA": ["MATEMATICAS UNIVERSITARIAS"],
            "CALIFICACION_RAW": ["9.1"],
            "CALIFICACION": [9.1],
            "CLAVEPROFESOR": pd.Series([12345], dtype="Int64"),
        }
    )
    dmu = pd.DataFrame({"CLAVEALUMNO": pd.Series([10], dtype="Int64"), "anio": pd.Series([2024], dtype="Int64"), "Porcentaje_DMU": [18.0]})
    gagb = pd.DataFrame({"CLAVEALUMNO": pd.Series([10], dtype="Int64"), "anio": pd.Series([2024], dtype="Int64"), "Porcentaje_GA_GB": [22.0]})

    settings = get_settings().with_overrides(
        merge_year_tolerance=1,
        subjects=("MAT1012",),
    )
    merged = build_merged_dataset(materias, dmu, gagb, settings).merged_df
    assert float(merged.loc[0, "Porcentaje_DMU"]) == 18.0
    assert merged.loc[0, "match_type_dmu"] == "nearest_year"


def test_target_cluster_selection_prefers_high_grade_low_exam_cluster() -> None:
    candidate = ClusterCandidate(
        method="gmm",
        n_clusters=2,
        labels=pd.Series([0, 0, 1, 1]).to_numpy(),
        centers_scaled=pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "Porcentaje_DMU": [-1.0, 1.0],
                "Porcentaje_GA_GB": [-0.8, 1.2],
                "CALIFICACION": [1.5, -1.0],
            }
        ),
        centers_original=pd.DataFrame(
            {
                "cluster_label": [0, 1],
                "Porcentaje_DMU": [20.0, 70.0],
                "Porcentaje_GA_GB": [25.0, 80.0],
                "CALIFICACION": [9.2, 6.0],
            }
        ),
        metrics={},
        cluster_sizes={0: 2, 1: 2},
        is_valid=True,
    )
    analysis_df = pd.DataFrame(
        {
            "Porcentaje_DMU": [20, 21, 70, 71],
            "Porcentaje_GA_GB": [25, 26, 80, 81],
            "CALIFICACION": [9.0, 9.4, 6.2, 5.8],
        }
    )
    outcome = select_target_cluster(candidate, analysis_df, feature_columns=("Porcentaje_DMU", "Porcentaje_GA_GB", "CALIFICACION"))
    assert outcome.cluster_label == 0
    assert outcome.validation_grade_above_mean is True
    assert outcome.validation_low_exam_score is True


def test_discrepancy_score_rewards_high_grade_low_percentages() -> None:
    settings = get_settings()
    z_df = pd.DataFrame(
        {
            "subject_z_calificacion": [1.2, -0.4],
            "subject_z_dmu": [-1.0, 0.5],
            "subject_z_gagb": [-0.8, 0.7],
        }
    )
    scores = compute_discrepancy_score(z_df, settings)
    assert scores.iloc[0] > scores.iloc[1]


def test_paradoxical_gmm_and_baseline_identify_expected_synthetic_group() -> None:
    settings = get_settings().with_overrides(subjects=("MAT1012",), make_paradoxical_plots=False)
    df = pd.DataFrame(
        {
            "CLAVEALUMNO": range(1, 9),
            "CLAVEVARIANTEMATERIA": ["MAT1012"] * 8,
            "DESCRIBEMATERIA": ["MATEMATICAS"] * 8,
            "CLAVEPROFESOR": [111] * 4 + [222] * 4,
            "anio": [2024] * 8,
            "CLAVESESION": ["PRIMAVERA"] * 8,
            "Porcentaje_DMU": [20, 22, 24, 25, 80, 82, 84, 85],
            "Porcentaje_GA_GB": [18, 20, 22, 24, 78, 80, 82, 84],
            "CALIFICACION": [9.4, 9.2, 9.5, 9.3, 7.0, 7.2, 7.4, 7.1],
            "data_complete_r3": [True] * 8,
        }
    )
    result = run_paradoxical_group_analysis(df, settings)
    enriched = result.enriched_df
    selected_students = set(enriched.loc[enriched["is_paradoxical_group_main"], "CLAVEALUMNO"])
    baseline_students = set(enriched.loc[enriched["binary_group_baseline_40_40_8"] == 1, "CLAVEALUMNO"])
    assert selected_students == {1, 2, 3, 4}
    assert baseline_students == {1, 2, 3, 4}
    assert {"subject_z_dmu", "discrepancy_score", "binary_group_gmm"}.issubset(enriched.columns)


def test_pipeline_smoke_run_with_synthetic_inputs(tmp_path: Path) -> None:
    materias_path, examenes_path = _build_temp_inputs(tmp_path)
    output_root = tmp_path / "output_cluster_analisis"
    processed_data_root = tmp_path / "data" / "datos_procesados"
    settings = get_settings().with_overrides(
        materias_input_path=materias_path,
        examenes_input_path=examenes_path,
        output_root=output_root,
        processed_data_root=processed_data_root,
        subjects=("MAT1012",),
        k_values=(2, 3),
        make_ica_plots=False,
        make_plotly_plots=False,
        make_presentation_plots=False,
        make_paradoxical_plots=False,
        update_latex_report=False,
    )
    artifacts = run_student_cluster_pipeline(settings)
    assert artifacts["merged_dataset_path"].exists()
    assert artifacts["cluster_metrics_path"].exists()
    assert artifacts["target_students_path"].exists()
    assert artifacts["target_professor_roster_path"].exists()
    assert artifacts["target_professor_students_path"].exists()
    assert artifacts["binary_group_summary_by_subject_path"].exists()
    assert artifacts["merged_dataset_csv_path"].exists()
    assert artifacts["merged_dataset_xlsx_path"].exists()
    assert artifacts["analysis_dataset_csv_path"].exists()
    assert artifacts["paradoxical_group_dataset_xlsx_path"].exists()
    assert artifacts["professor_appendix_all_years_csv_path"].exists()
    assert artifacts["professor_appendix_by_period_xlsx_path"].exists()
    assert artifacts["processed_data_readme_path"].exists()
    assert artifacts["processed_data_dictionary_path"].exists()
    report_df = pd.read_csv(artifacts["professor_report_path"])
    merged_df = pd.read_csv(artifacts["merged_dataset_path"])
    period_appendix_df = pd.read_csv(artifacts["professor_appendix_by_period_csv_path"])
    target_students_df = pd.read_csv(artifacts["target_students_path"])
    target_professors_df = pd.read_csv(artifacts["target_professor_roster_path"])
    assert not report_df.empty
    assert not period_appendix_df.empty
    assert not target_students_df.empty
    assert not target_professors_df.empty
    assert {"CLAVEPROFESOR", "share_cluster_objetivo"}.issubset(report_df.columns)
    assert {"binary_group_gmm", "discrepancy_score", "is_paradoxical_group_main"}.issubset(merged_df.columns)
    assert {"anio", "CLAVESESION", "alumnos_benchmark_manual", "ranking_position"}.issubset(period_appendix_df.columns)
    assert {"CLAVEALUMNO", "CLAVEPROFESOR", "target_cluster_score"}.issubset(target_students_df.columns)
    assert {"CLAVEPROFESOR", "alumnos_cluster_objetivo_ids"}.issubset(target_professors_df.columns)
