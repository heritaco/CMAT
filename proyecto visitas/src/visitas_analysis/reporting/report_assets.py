from __future__ import annotations

from pathlib import Path

from visitas_analysis.paths import (
    DEFAULT_ASESORIAS_PATH,
    DEFAULT_MATERIAS_PATH,
    PROJECT_ROOT,
    REPORT_ASSETS_DIR,
)
from visitas_analysis.reporting.metrics import (
    compute_classroom_size_distribution,
    compute_classroom_unit_summary,
    compute_concentration_outputs,
    compute_grade_variable_summary,
    compute_non_numeric_grade_tokens,
    compute_professor_summary,
    compute_source_data_overview,
    compute_student_summary,
    compute_student_visit_distribution,
    compute_subject_summary,
    compute_summary_json,
    compute_threshold_summaries,
    compute_top_students_by_visits,
    compute_year_summary,
)
from visitas_analysis.reporting.descriptive_pipeline import build_analytical_bundle
from visitas_analysis.reporting.plots import generate_figures
from visitas_analysis.reporting.render import write_csv_tables, write_json_summary, write_readme, write_tex_snippets


def _reset_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _log_paths(label: str, paths: list[Path], project_root: Path) -> None:
    print(f"{label}: {len(paths)}")
    for path in paths:
        print(f"  - {path.relative_to(project_root)}")


def generate_report_assets(
    *,
    project_root: Path = PROJECT_ROOT,
    materias_path: Path = DEFAULT_MATERIAS_PATH,
    asesorias_path: Path = DEFAULT_ASESORIAS_PATH,
    output_dir: Path = REPORT_ASSETS_DIR,
) -> dict[str, object]:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tex_dir = output_dir / "tex"
    summary_path = output_dir / "summary.json"
    readme_path = output_dir / "README.md"

    _reset_output_dir(tables_dir)
    _reset_output_dir(figures_dir)
    _reset_output_dir(tex_dir)

    bundle = build_analytical_bundle(
        project_root=project_root,
        materias_path=materias_path,
        asesorias_path=asesorias_path,
    )

    source_overview = compute_source_data_overview(bundle)
    visit_exact, visit_tail, visit_thresholds, visit_summary = compute_student_visit_distribution(bundle.student_visits)
    year_summary, asesorias_raw_year_summary = compute_year_summary(bundle)
    classroom_summary = compute_classroom_unit_summary(bundle)
    classroom_size_distribution = compute_classroom_size_distribution(classroom_summary)
    professor_summary = compute_professor_summary(bundle, classroom_summary)
    subject_summary = compute_subject_summary(bundle, classroom_summary)
    student_summary = compute_student_summary(bundle)
    grade_summary = compute_grade_variable_summary(bundle)
    non_numeric_grade_tokens = compute_non_numeric_grade_tokens(bundle)
    threshold_summaries = compute_threshold_summaries(bundle)
    top_students_by_visits = compute_top_students_by_visits(student_summary)
    concentration_summary, lorenz_visits, classroom_units_per_student_distribution = compute_concentration_outputs(
        bundle,
        student_summary,
        year_summary,
    )
    summary = compute_summary_json(
        bundle=bundle,
        visit_summary=visit_summary,
        year_summary=year_summary,
        grade_summary=grade_summary,
        concentration_summary=concentration_summary,
    )

    tables = {
        "source_data_overview": source_overview,
        "cleaning_summary": bundle.cleaning_summary,
        "student_visit_distribution_exact": visit_exact,
        "student_visit_distribution_tail": visit_tail,
        "student_visit_thresholds": visit_thresholds,
        "visit_summary": visit_summary,
        "visits_by_year": year_summary,
        "asesorias_raw_year_summary": asesorias_raw_year_summary,
        "classroom_unit_summary": classroom_summary,
        "classroom_size_distribution": classroom_size_distribution,
        "professor_summary": professor_summary,
        "subject_summary": subject_summary,
        "student_summary": student_summary,
        "grade_missingness_summary": grade_summary,
        "raw_grade_non_numeric_tokens": non_numeric_grade_tokens,
        "threshold_summaries": threshold_summaries,
        "concentration_summary": concentration_summary,
        "lorenz_visits": lorenz_visits,
        "student_classroom_units_per_student_distribution": classroom_units_per_student_distribution,
        "top_students_by_visits": top_students_by_visits,
    }

    table_paths = write_csv_tables(tables, tables_dir)
    figure_paths = generate_figures(
        student_visits=bundle.student_visits,
        visit_tail=visit_tail,
        student_year_visits=bundle.student_year_visits,
        classroom_summary=classroom_summary,
        lorenz=lorenz_visits,
        output_dir=figures_dir,
    )
    tex_paths = write_tex_snippets(
        summary=summary,
        source_overview=source_overview,
        cleaning_summary=bundle.cleaning_summary,
        visit_thresholds=visit_thresholds,
        year_summary=year_summary,
        concentration_summary=concentration_summary,
        grade_summary=grade_summary,
        non_numeric_grade_tokens=non_numeric_grade_tokens,
        output_dir=tex_dir,
    )
    write_json_summary(summary, summary_path)
    write_readme(
        path=readme_path,
        summary=summary,
        source_overview=source_overview,
        cleaning_summary=bundle.cleaning_summary,
        visit_summary=visit_summary,
        visit_thresholds=visit_thresholds,
        year_summary=year_summary,
        classroom_summary=classroom_summary,
        professor_summary=professor_summary,
        grade_summary=grade_summary,
        concentration_summary=concentration_summary,
        table_paths=table_paths,
        figure_paths=figure_paths,
        tex_paths=tex_paths,
    )

    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "readme_path": readme_path,
        "tables_dir": tables_dir,
        "figures_dir": figures_dir,
        "tex_dir": tex_dir,
        "table_paths": table_paths,
        "figure_paths": figure_paths,
        "tex_paths": tex_paths,
    }


def main() -> None:
    artifacts = generate_report_assets()
    print(f"Summary JSON: {artifacts['summary_path'].relative_to(PROJECT_ROOT)}")
    print(f"README: {artifacts['readme_path'].relative_to(PROJECT_ROOT)}")
    _log_paths("Tables", artifacts["table_paths"], PROJECT_ROOT)
    _log_paths("Figures", artifacts["figure_paths"], PROJECT_ROOT)
    _log_paths("TeX", artifacts["tex_paths"], PROJECT_ROOT)


if __name__ == "__main__":
    main()
