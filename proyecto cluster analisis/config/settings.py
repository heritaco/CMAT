from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_tuple(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    project_root: Path
    repo_root: Path
    materias_input_path: Path
    examenes_input_path: Path
    examenes_dmu_sheet: str
    examenes_gagb_sheet: str
    output_root: Path
    subjects: tuple[str, ...]
    feature_columns: tuple[str, ...]
    clustering_method: str
    k_values: tuple[int, ...]
    selection_strategy: str
    preferred_min_clusters: int
    target_cluster_min_fraction: float
    minimum_grade_for_clustering: float
    random_state: int
    gmm_covariance_type: str
    merge_year_tolerance: int
    enable_unique_id_fallback: bool
    min_cluster_size: int
    min_cluster_fraction: float
    min_students_per_professor: int
    min_students_per_professor_relaxed: int
    high_data_loss_threshold: float
    minimum_rows_for_candidate: int
    make_ica_plots: bool
    make_plotly_plots: bool
    make_presentation_plots: bool
    enable_paradoxical_analysis: bool
    paradoxical_main_method: str
    compare_paradoxical_methods: bool
    make_paradoxical_plots: bool
    update_latex_report: bool
    paradoxical_score_cal_weight: float
    paradoxical_score_dmu_weight: float
    paradoxical_score_gagb_weight: float
    paradoxical_baseline_dmu_threshold: float
    paradoxical_baseline_gagb_threshold: float
    paradoxical_baseline_grade_threshold: float
    paradoxical_min_subject_rows: int
    paradoxical_min_group_size: int
    paradoxical_min_group_fraction: float
    paradoxical_max_group_fraction_warning: float
    paradoxical_top_n_professors: int
    paradoxical_top_k_overlap: int
    save_intermediate_files: bool
    figure_dpi: int
    presentation_top_n_professors: int
    quantiles: tuple[float, ...]
    merged_dataset_filename: str
    cluster_metrics_filename: str
    centroids_filename: str
    target_cluster_filename: str
    target_cluster_students_filename: str
    professor_report_filename: str
    target_professor_roster_filename: str
    target_professor_students_filename: str
    global_professor_ranking_filename: str
    data_quality_filename: str
    missingness_filename: str
    log_filename: str

    @property
    def output_data_clean_dir(self) -> Path:
        return self.output_root / "data_clean"

    @property
    def output_metrics_dir(self) -> Path:
        return self.output_root / "metrics"

    @property
    def output_centroids_dir(self) -> Path:
        return self.output_root / "centroids"

    @property
    def output_professor_reports_dir(self) -> Path:
        return self.output_root / "professor_reports"

    @property
    def output_plots_2d_dir(self) -> Path:
        return self.output_root / "plots_2d_ica"

    @property
    def output_plots_3d_dir(self) -> Path:
        return self.output_root / "plots_3d_plotly"

    @property
    def output_presentation_plots_dir(self) -> Path:
        return self.output_root / "presentation_plots"

    @property
    def output_paradoxical_root_dir(self) -> Path:
        return self.output_root / "paradoxical_analysis"

    @property
    def output_paradoxical_tables_dir(self) -> Path:
        return self.output_paradoxical_root_dir / "tables"

    @property
    def output_paradoxical_figures_dir(self) -> Path:
        return self.output_paradoxical_root_dir / "figures"

    @property
    def output_paradoxical_subject_figures_dir(self) -> Path:
        return self.output_paradoxical_figures_dir / "by_subject"

    @property
    def output_paradoxical_diagnostics_dir(self) -> Path:
        return self.output_paradoxical_root_dir / "diagnostics"

    @property
    def output_reports_dir(self) -> Path:
        return self.project_root / "reportes"

    @property
    def output_summaries_dir(self) -> Path:
        return self.output_root / "summaries"

    @property
    def output_diagnostics_dir(self) -> Path:
        return self.output_root / "diagnostics"

    @property
    def output_logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def output_directories(self) -> tuple[Path, ...]:
        return (
            self.output_root,
            self.output_data_clean_dir,
            self.output_metrics_dir,
            self.output_centroids_dir,
            self.output_professor_reports_dir,
            self.output_plots_2d_dir,
            self.output_plots_3d_dir,
            self.output_presentation_plots_dir,
            self.output_paradoxical_root_dir,
            self.output_paradoxical_tables_dir,
            self.output_paradoxical_figures_dir,
            self.output_paradoxical_subject_figures_dir,
            self.output_paradoxical_diagnostics_dir,
            self.output_summaries_dir,
            self.output_diagnostics_dir,
            self.output_logs_dir,
            self.output_reports_dir,
        )

    def with_overrides(self, **overrides: object) -> "Settings":
        return replace(self, **overrides)


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent

    return Settings(
        project_root=project_root,
        repo_root=repo_root,
        materias_input_path=_env_path(
            "SCA_MATERIAS_PATH",
            repo_root / "data" / "onedrive" / "Archivos2024" / "Materias estudiantes-profesores 2019-2025 P y O.xlsx",
        ),
        examenes_input_path=_env_path(
            "SCA_EXAMENES_PATH",
            repo_root / "onedrive" / "RicardoMJ" / "resultados_examenes.xlsx",
        ),
        examenes_dmu_sheet=os.environ.get("SCA_DMU_SHEET", "DMU"),
        examenes_gagb_sheet=os.environ.get("SCA_GAGB_SHEET", "GA-GB"),
        output_root=_env_path("SCA_OUTPUT_ROOT", project_root / "output_cluster_analisis"),
        subjects=(
            "MAT1012",
            "MAT1022",
            "MAT1032",
            "MAT1052",
            "MAT2012",
            "MAT2022",
        ),
        feature_columns=("Porcentaje_DMU", "Porcentaje_GA_GB", "CALIFICACION"),
        clustering_method=os.environ.get("SCA_CLUSTERING_METHOD", "gmm").strip().lower(),
        k_values=_env_int_tuple("SCA_K_VALUES", tuple(range(2, 13))),
        selection_strategy=os.environ.get("SCA_SELECTION_STRATEGY", "target_oriented").strip().lower(),
        preferred_min_clusters=int(os.environ.get("SCA_PREFERRED_MIN_CLUSTERS", "4")),
        target_cluster_min_fraction=float(os.environ.get("SCA_TARGET_CLUSTER_MIN_FRACTION", "0.01")),
        minimum_grade_for_clustering=float(os.environ.get("SCA_MINIMUM_GRADE_FOR_CLUSTERING", "7.5")),
        random_state=42,
        gmm_covariance_type=os.environ.get("SCA_GMM_COVARIANCE_TYPE", "full"),
        merge_year_tolerance=int(os.environ.get("SCA_MERGE_YEAR_TOLERANCE", "1")),
        enable_unique_id_fallback=_env_bool("SCA_ENABLE_UNIQUE_ID_FALLBACK", True),
        min_cluster_size=int(os.environ.get("SCA_MIN_CLUSTER_SIZE", "5")),
        min_cluster_fraction=float(os.environ.get("SCA_MIN_CLUSTER_FRACTION", "0.005")),
        min_students_per_professor=int(os.environ.get("SCA_MIN_STUDENTS_PER_PROFESSOR", "10")),
        min_students_per_professor_relaxed=int(os.environ.get("SCA_MIN_STUDENTS_PER_PROFESSOR_RELAXED", "5")),
        high_data_loss_threshold=float(os.environ.get("SCA_HIGH_DATA_LOSS_THRESHOLD", "0.30")),
        minimum_rows_for_candidate=int(os.environ.get("SCA_MINIMUM_ROWS_FOR_CANDIDATE", "6")),
        make_ica_plots=_env_bool("SCA_MAKE_ICA_PLOTS", True),
        make_plotly_plots=_env_bool("SCA_MAKE_PLOTLY_PLOTS", True),
        make_presentation_plots=_env_bool("SCA_MAKE_PRESENTATION_PLOTS", True),
        enable_paradoxical_analysis=_env_bool("SCA_ENABLE_PARADOXICAL_ANALYSIS", True),
        paradoxical_main_method=os.environ.get("SCA_PARADOXICAL_MAIN_METHOD", "gmm").strip().lower(),
        compare_paradoxical_methods=_env_bool("SCA_COMPARE_PARADOXICAL_METHODS", True),
        make_paradoxical_plots=_env_bool("SCA_MAKE_PARADOXICAL_PLOTS", True),
        update_latex_report=_env_bool("SCA_UPDATE_LATEX_REPORT", True),
        paradoxical_score_cal_weight=float(os.environ.get("SCA_PARADOXICAL_SCORE_CAL_WEIGHT", "1.0")),
        paradoxical_score_dmu_weight=float(os.environ.get("SCA_PARADOXICAL_SCORE_DMU_WEIGHT", "0.5")),
        paradoxical_score_gagb_weight=float(os.environ.get("SCA_PARADOXICAL_SCORE_GAGB_WEIGHT", "0.5")),
        paradoxical_baseline_dmu_threshold=float(os.environ.get("SCA_PARADOXICAL_BASELINE_DMU_THRESHOLD", "40")),
        paradoxical_baseline_gagb_threshold=float(os.environ.get("SCA_PARADOXICAL_BASELINE_GAGB_THRESHOLD", "40")),
        paradoxical_baseline_grade_threshold=float(os.environ.get("SCA_PARADOXICAL_BASELINE_GRADE_THRESHOLD", "8")),
        paradoxical_min_subject_rows=int(os.environ.get("SCA_PARADOXICAL_MIN_SUBJECT_ROWS", "6")),
        paradoxical_min_group_size=int(os.environ.get("SCA_PARADOXICAL_MIN_GROUP_SIZE", "5")),
        paradoxical_min_group_fraction=float(os.environ.get("SCA_PARADOXICAL_MIN_GROUP_FRACTION", "0.01")),
        paradoxical_max_group_fraction_warning=float(os.environ.get("SCA_PARADOXICAL_MAX_GROUP_FRACTION_WARNING", "0.80")),
        paradoxical_top_n_professors=int(os.environ.get("SCA_PARADOXICAL_TOP_N_PROFESSORS", "12")),
        paradoxical_top_k_overlap=int(os.environ.get("SCA_PARADOXICAL_TOP_K_OVERLAP", "10")),
        save_intermediate_files=_env_bool("SCA_SAVE_INTERMEDIATE_FILES", True),
        figure_dpi=int(os.environ.get("SCA_FIGURE_DPI", "160")),
        presentation_top_n_professors=int(os.environ.get("SCA_PRESENTATION_TOP_N_PROFESSORS", "12")),
        quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
        merged_dataset_filename="merged_dataset.csv",
        cluster_metrics_filename="cluster_metrics_por_materia.csv",
        centroids_filename="centroides_por_materia.csv",
        target_cluster_filename="cluster_objetivo_por_materia.csv",
        target_cluster_students_filename="alumnos_cluster_objetivo.csv",
        professor_report_filename="profesores_por_materia.csv",
        target_professor_roster_filename="profesores_cluster_objetivo_detalle.csv",
        target_professor_students_filename="alumnos_profesores_cluster_objetivo.csv",
        global_professor_ranking_filename="ranking_profesores_global.csv",
        data_quality_filename="data_quality_report.csv",
        missingness_filename="missingness_por_materia.csv",
        log_filename="pipeline.log",
    )
