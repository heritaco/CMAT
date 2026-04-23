from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from visitas_analysis.paths import DEFAULT_ASESORIAS_PATH, DEFAULT_MATERIAS_PATH, OUTPUT_ROOT
from visitas_analysis.visualization import style

from .context import load_base_context, load_visits_context
from .figures_clusters import (
    assign_notebook_clusters,
    build_cluster_context,
    plot_cluster_distributions,
    plot_cluster_distributions_with_ci,
    plot_cluster_heatmap,
    plot_cluster_selection,
)
from .figures_imputation import (
    plot_global_imputation_phase1,
    plot_global_imputation_phase2,
    plot_outlier_phase1,
    plot_outlier_phase2,
    plot_single_classroom_comparison,
)
from .figures_professors import (
    plot_all_professors_png,
    plot_imputed_professors_split,
    plot_reported_professors_split,
    plot_yearly_professor_variance,
)
from .figures_tests import (
    plot_ecdf_salon,
    plot_ecdf_student,
    plot_nonparametric_salon,
    plot_nonparametric_student,
    plot_parametric_salon,
    plot_parametric_student,
)
from .figures_visits import plot_mean_z_by_visits, plot_salon_scatter, plot_student_scatter, plot_visit_histograms
from .imputation import (
    build_salones_mean_only,
    build_salones_with_imputer,
    compute_ultramerge_means,
    concat_salones,
    corrected_impute_nans_from_pre75_kde_df,
    legacy_impute_nans_from_pre75_kde_df,
)
from .plot_helpers import build_output_layout, ensure_output_dirs


DEFAULT_MATERIAS = DEFAULT_MATERIAS_PATH
DEFAULT_ASESORIAS = DEFAULT_ASESORIAS_PATH
DEFAULT_OUTPUT_ROOT = OUTPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera las figuras de reporte compatibles con la notebook 00 Raw.")
    parser.add_argument("--materias-path", type=Path, default=DEFAULT_MATERIAS)
    parser.add_argument("--asesorias-path", type=Path, default=DEFAULT_ASESORIAS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def run_raw_report_figures(
    *,
    materias_path: Path = DEFAULT_MATERIAS,
    asesorias_path: Path = DEFAULT_ASESORIAS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    style.mpl_apply()

    layout = build_output_layout(output_root)
    ensure_output_dirs(layout)

    base = load_base_context(materias_path)
    plot_yearly_professor_variance(base, layout)
    plot_all_professors_png(base, layout)

    mean_only_salones = build_salones_mean_only(base.materias, base.profes_ids)
    plot_single_classroom_comparison(mean_only_salones, layout)

    phase1_salones = build_salones_with_imputer(base.materias, base.profes_ids, legacy_impute_nans_from_pre75_kde_df)
    plot_global_imputation_phase1(phase1_salones, layout)

    visits = load_visits_context(asesorias_path, base.materias)
    plot_visit_histograms(visits.asesoria_counts, layout)

    legacy_visit_salones = build_salones_with_imputer(
        visits.materias,
        base.profes_ids,
        legacy_impute_nans_from_pre75_kde_df,
    )
    ultramerge_legacy = concat_salones(legacy_visit_salones)
    ultramerge_means_legacy = compute_ultramerge_means(ultramerge_legacy, visits.materias)
    plot_outlier_phase1(legacy_visit_salones, layout)

    corrected_visit_salones = build_salones_with_imputer(
        visits.materias,
        base.profes_ids,
        corrected_impute_nans_from_pre75_kde_df,
        imputer_kwargs={"constant_value": 6.5, "fallback": "uniform"},
    )
    ultramerge_corrected = concat_salones(corrected_visit_salones)
    plot_outlier_phase2(corrected_visit_salones, layout)
    plot_global_imputation_phase2(ultramerge_corrected, layout)

    plot_salon_scatter(ultramerge_corrected, layout)
    plot_student_scatter(ultramerge_means_legacy, layout)
    plot_ecdf_salon(ultramerge_corrected, layout)
    plot_ecdf_student(ultramerge_means_legacy, layout)
    plot_parametric_student(ultramerge_means_legacy, layout)
    plot_parametric_salon(ultramerge_corrected, layout)
    plot_nonparametric_student(ultramerge_means_legacy, layout)
    plot_nonparametric_salon(ultramerge_corrected, layout)
    plot_mean_z_by_visits(ultramerge_means_legacy, layout)

    plot_reported_professors_split(base, ultramerge_corrected, layout)
    plot_imputed_professors_split(base, ultramerge_corrected, layout)

    cluster_ctx = build_cluster_context(ultramerge_corrected)
    plot_cluster_heatmap(cluster_ctx, layout)
    plot_cluster_selection(cluster_ctx, layout)
    ultramerge_clustered = assign_notebook_clusters(ultramerge_corrected, cluster_ctx)
    plot_cluster_distributions(base, ultramerge_clustered, layout)
    plot_cluster_distributions_with_ci(base, ultramerge_clustered, layout)
    return {
        "output_root": output_root,
        "raw_report_figures_dir": layout.pdf_dir,
        "professor_distributions_dir": layout.professor_dir,
        "imputation_dir": layout.imputation_dir,
    }


def main() -> int:
    args = parse_args()
    run_raw_report_figures(
        materias_path=args.materias_path,
        asesorias_path=args.asesorias_path,
        output_root=args.output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
