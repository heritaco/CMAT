from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import Settings
from student_cluster_analysis.analytics.diagnostics import (
    build_data_quality_report,
    build_missingness_by_subject,
)
from student_cluster_analysis.analytics.method_comparison import (
    build_binary_group_summary_by_subject,
    build_method_comparison_by_subject,
    build_overlap_between_methods,
    build_paradoxical_diagnostics,
    build_professor_paradoxical_global_ranking,
    build_professor_paradoxical_summary,
    build_professor_ranking_stability,
)
from student_cluster_analysis.analytics.manual_threshold_group import run_manual_threshold_analysis
from student_cluster_analysis.analytics.paradoxical_group import run_paradoxical_group_analysis
from student_cluster_analysis.analytics.professor_stats import (
    build_global_professor_ranking,
    build_subject_professor_report,
)
from student_cluster_analysis.analytics.processed_data import (
    build_analysis_dataset,
    build_professor_appendix_tables,
    build_subject_period_summary,
    build_subject_summary_dataset,
)
from student_cluster_analysis.analytics.summaries import (
    build_centroids_table,
    build_cluster_metrics_table,
    build_target_cluster_table,
)
from student_cluster_analysis.analytics.target_details import (
    build_students_for_target_professors,
    build_target_cluster_students,
    build_target_professor_roster,
)
from student_cluster_analysis.clustering.selection import select_best_clustering
from student_cluster_analysis.clustering.target_cluster import select_target_cluster
from student_cluster_analysis.entities import SubjectAnalysisResult
from student_cluster_analysis.features.subject_filter import (
    build_subject_frames,
    filter_supported_subjects,
    mark_clustering_eligibility,
    mark_complete_cases,
    select_merged_output_columns,
)
from student_cluster_analysis.io.readers import load_raw_inputs
from student_cluster_analysis.io.writers import (
    ensure_output_structure,
    save_matplotlib_figure,
    save_plotly_figure,
    write_dataframe_csv_and_excel,
    write_dataframe,
)
from student_cluster_analysis.preprocessing.cleaning import clean_exam_dataframe, clean_materias_dataframe
from student_cluster_analysis.preprocessing.merging import build_merged_dataset
from student_cluster_analysis.reporting.data_documentation import write_processed_data_documentation
from student_cluster_analysis.reporting.latex_report import (
    write_manual_mat1012_latex_report,
    write_paradoxical_latex_section,
    write_professor_appendix_latex,
    write_subject_professor_appendix_latex,
)
from student_cluster_analysis.visualization.plots_2d import create_ica_plot
from student_cluster_analysis.visualization.plots_3d import create_plotly_3d
from student_cluster_analysis.visualization.paradoxical_plots import create_paradoxical_figures
from student_cluster_analysis.visualization.presentation_plots import create_presentation_plots


def _configure_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger("student_cluster_analysis")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger

    ensure_output_structure(settings)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(settings.output_logs_dir / settings.log_filename, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _append_subject_warning(result: SubjectAnalysisResult, message: str) -> None:
    if message not in result.warnings:
        result.warnings.append(message)


def run_student_cluster_pipeline(settings: Settings) -> dict[str, Path]:
    ensure_output_structure(settings)
    logger = _configure_logging(settings)
    logger.info("Starting student cluster analysis pipeline.")

    raw_inputs = load_raw_inputs(settings)
    logger.info("Cleaning materias and exam sheets.")
    materias_clean_df = clean_materias_dataframe(raw_inputs.materias_df)
    dmu_clean_df = clean_exam_dataframe(
        raw_inputs.dmu_df,
        percentage_column="Porcentaje_DMU",
        exam_label=settings.examenes_dmu_sheet,
    )
    gagb_clean_df = clean_exam_dataframe(
        raw_inputs.gagb_df,
        percentage_column="Porcentaje_GA_GB",
        exam_label=settings.examenes_gagb_sheet,
    )

    materias_filtered_df = filter_supported_subjects(materias_clean_df, settings.subjects)
    logger.info("Building merged dataset with hierarchical exam matching.")
    merge_result = build_merged_dataset(materias_filtered_df, dmu_clean_df, gagb_clean_df, settings)
    merged_df = mark_complete_cases(merge_result.merged_df, settings.feature_columns)
    merged_df = mark_clustering_eligibility(
        merged_df,
        minimum_grade=settings.minimum_grade_for_clustering,
    )
    merged_base_df = merged_df.copy()

    paradoxical_metadata = []
    if settings.enable_paradoxical_analysis:
        logger.info("Running binary paradoxical-group analysis on complete R^3 rows.")
        paradoxical_result = run_paradoxical_group_analysis(merged_df, settings)
        merged_df = paradoxical_result.enriched_df
        paradoxical_metadata = paradoxical_result.subject_metadata

    logger.info("Running strict manual 50/50/8 analysis on complete R^3 rows.")
    manual_result = run_manual_threshold_analysis(merged_df, settings)
    merged_df = manual_result.enriched_df

    merged_output_df = select_merged_output_columns(merged_df)
    merged_dataset_path = settings.output_data_clean_dir / settings.merged_dataset_filename
    write_dataframe(merged_output_df, merged_dataset_path)
    logger.info("Merged dataset written to %s", merged_dataset_path)

    subject_frames = build_subject_frames(merged_df, settings.subjects)
    subject_results: list[SubjectAnalysisResult] = []
    professor_reports: list[pd.DataFrame] = []

    for subject_code, subject_df in subject_frames.items():
        subject_name = (
            subject_df["DESCRIBEMATERIA"].dropna().astype(str).iloc[0]
            if not subject_df.empty and not subject_df["DESCRIBEMATERIA"].dropna().empty
            else None
        )
        logger.info("Processing subject %s with %s rows.", subject_code, len(subject_df))
        analysis_df = subject_df[subject_df["eligible_for_clustering"]].copy().reset_index(drop=True)
        total_rows = len(subject_df)
        complete_r3_rows = int(subject_df["data_complete_r3"].sum())
        excluded_low_grade_rows = int(
            (subject_df["data_complete_r3"] & ~subject_df["passes_minimum_grade_for_clustering"]).sum()
        )
        complete_rows = len(analysis_df)
        excluded_rows = total_rows - complete_rows
        loss_fraction = excluded_rows / total_rows if total_rows else 0.0

        result = SubjectAnalysisResult(
            subject_code=subject_code,
            subject_name=subject_name,
            full_subject_df=subject_df,
            analysis_df=pd.DataFrame(),
            selection=None,
            target_cluster=None,
            professor_stats=pd.DataFrame(),
            total_rows=total_rows,
            complete_rows=complete_rows,
            excluded_rows=excluded_rows,
            loss_fraction=loss_fraction,
            status="pending",
            warnings=[],
            complete_r3_rows=complete_r3_rows,
            excluded_low_grade_rows=excluded_low_grade_rows,
            minimum_grade_for_clustering=settings.minimum_grade_for_clustering,
        )
        if loss_fraction > settings.high_data_loss_threshold:
            _append_subject_warning(
                result,
                (
                    f"Data loss after complete R^3 and CALIFICACION >= "
                    f"{settings.minimum_grade_for_clustering:g} filtering is {loss_fraction:.2%}, above the "
                    f"configured threshold of {settings.high_data_loss_threshold:.2%}."
                ),
            )
        if excluded_low_grade_rows:
            _append_subject_warning(
                result,
                (
                    f"{excluded_low_grade_rows} complete R^3 observations were excluded because "
                    f"CALIFICACION < {settings.minimum_grade_for_clustering:g}."
                ),
            )

        if analysis_df.empty or len(analysis_df) < settings.minimum_rows_for_candidate:
            result.status = "skipped_insufficient_data"
            _append_subject_warning(result, "Insufficient eligible rows to run clustering.")
            subject_results.append(result)
            continue

        selection = select_best_clustering(
            analysis_df,
            subject_code=subject_code,
            feature_columns=settings.feature_columns,
            settings=settings,
        )
        result.selection = selection
        if selection.selected is None:
            result.status = "skipped_model_selection"
            result.warnings.extend(selection.notes)
            subject_results.append(result)
            continue

        selected_candidate = selection.selected
        analysis_df = analysis_df.copy()
        analysis_df["cluster_label"] = selected_candidate.labels
        target_cluster = select_target_cluster(
            selected_candidate,
            analysis_df,
            feature_columns=settings.feature_columns,
        )
        analysis_df["is_target_cluster"] = analysis_df["cluster_label"] == target_cluster.cluster_label

        result.analysis_df = analysis_df
        result.target_cluster = target_cluster
        result.status = "clustered"
        result.warnings.extend(selection.notes)
        result.warnings.extend(target_cluster.notes)

        professor_report = build_subject_professor_report(
            analysis_df,
            subject_df,
            target_cluster_label=target_cluster.cluster_label,
            settings=settings,
        )
        result.professor_stats = professor_report
        professor_reports.append(professor_report)

        if settings.make_ica_plots:
            try:
                fig = create_ica_plot(
                    analysis_df,
                    subject_code=subject_code,
                    subject_name=subject_name,
                    feature_columns=settings.feature_columns,
                    target_cluster_label=target_cluster.cluster_label,
                    centroids_scaled=selected_candidate.centers_scaled,
                    random_state=settings.random_state,
                    dpi=settings.figure_dpi,
                )
                save_matplotlib_figure(fig, settings.output_plots_2d_dir / f"{subject_code}_ica.png")
            except Exception as exc:  # pragma: no cover - visualization fallback
                _append_subject_warning(result, f"2D ICA plot failed: {exc}")

        if settings.make_plotly_plots:
            try:
                fig_3d = create_plotly_3d(
                    analysis_df,
                    subject_code=subject_code,
                    subject_name=subject_name,
                    target_cluster_label=target_cluster.cluster_label,
                    centroids_original=selected_candidate.centers_original,
                )
                save_plotly_figure(fig_3d, settings.output_plots_3d_dir / f"{subject_code}_clusters_3d.html")
            except Exception as exc:  # pragma: no cover - visualization fallback
                _append_subject_warning(result, f"3D Plotly plot failed: {exc}")

        subject_results.append(result)

    logger.info("Building summary tables and reports.")
    data_quality_df = build_data_quality_report(
        raw_inputs=raw_inputs,
        materias_clean_df=materias_clean_df,
        materias_filtered_df=materias_filtered_df,
        dmu_clean_df=dmu_clean_df,
        gagb_clean_df=gagb_clean_df,
        merge_audit_df=merge_result.merge_audit_df,
        merged_df=merged_df,
        subject_results=subject_results,
        settings=settings,
    )
    missingness_df = build_missingness_by_subject(merged_df, settings)
    cluster_metrics_df = build_cluster_metrics_table(subject_results)
    centroids_df = build_centroids_table(subject_results, settings.feature_columns)
    target_cluster_df = build_target_cluster_table(subject_results)
    professor_report_df = pd.concat(professor_reports, ignore_index=True) if professor_reports else pd.DataFrame()
    target_students_df = build_target_cluster_students(subject_results)
    target_professor_roster_df = build_target_professor_roster(subject_results, settings)
    target_professor_students_df = build_students_for_target_professors(subject_results)
    global_ranking_df = build_global_professor_ranking(subject_results, settings)

    paradoxical_artifacts: dict[str, Path] = {}
    paradoxical_plot_paths: list[Path] = []
    binary_summary_df = pd.DataFrame()
    method_comparison_df = pd.DataFrame()
    overlap_df = pd.DataFrame()
    professor_paradoxical_df = pd.DataFrame()
    professor_paradoxical_global_df = pd.DataFrame()
    ranking_stability_df = pd.DataFrame()
    if settings.enable_paradoxical_analysis:
        logger.info("Building binary paradoxical-group tables.")
        binary_summary_df = build_binary_group_summary_by_subject(merged_df, settings)
        method_comparison_df = build_method_comparison_by_subject(merged_df)
        overlap_df = build_overlap_between_methods(merged_df)
        paradoxical_diagnostics_df = build_paradoxical_diagnostics(merged_df, paradoxical_metadata, settings)
        professor_paradoxical_df = build_professor_paradoxical_summary(merged_df, settings)
        professor_paradoxical_global_df = build_professor_paradoxical_global_ranking(
            professor_paradoxical_df,
            settings,
        )
        ranking_stability_df = build_professor_ranking_stability(
            professor_paradoxical_df,
            professor_paradoxical_global_df,
            settings,
        )

        paradoxical_tables = {
            "binary_group_summary_by_subject_path": (
                settings.output_paradoxical_tables_dir / "binary_group_summary_by_subject.csv",
                binary_summary_df,
            ),
            "method_comparison_by_subject_path": (
                settings.output_paradoxical_tables_dir / "method_comparison_by_subject.csv",
                method_comparison_df,
            ),
            "overlap_between_methods_path": (
                settings.output_paradoxical_tables_dir / "overlap_between_methods.csv",
                overlap_df,
            ),
            "professor_paradoxical_summary_path": (
                settings.output_paradoxical_tables_dir / "professor_paradoxical_summary.csv",
                professor_paradoxical_df,
            ),
            "professor_paradoxical_global_ranking_path": (
                settings.output_paradoxical_tables_dir / "professor_paradoxical_global_ranking.csv",
                professor_paradoxical_global_df,
            ),
            "professor_ranking_stability_path": (
                settings.output_paradoxical_tables_dir / "professor_ranking_stability.csv",
                ranking_stability_df,
            ),
            "paradoxical_group_diagnostics_path": (
                settings.output_paradoxical_diagnostics_dir / "paradoxical_group_diagnostics.csv",
                paradoxical_diagnostics_df,
            ),
        }
        for artifact_name, (path, dataframe) in paradoxical_tables.items():
            write_dataframe(dataframe, path)
            paradoxical_artifacts[artifact_name] = path

        if settings.make_paradoxical_plots:
            logger.info("Creating binary paradoxical-group figures.")
            try:
                paradoxical_plot_paths = create_paradoxical_figures(
                    enriched_df=merged_df,
                    summary_df=binary_summary_df,
                    overlap_df=overlap_df,
                    professor_summary_df=professor_paradoxical_df,
                    ranking_stability_df=ranking_stability_df,
                    settings=settings,
                )
            except Exception as exc:  # pragma: no cover - visualization fallback
                logger.exception("Paradoxical analysis figures failed: %s", exc)

        if settings.update_latex_report:
            try:
                paradoxical_artifacts["paradoxical_latex_section_path"] = write_paradoxical_latex_section(
                    summary_df=binary_summary_df,
                    overlap_df=overlap_df,
                    professor_summary_df=professor_paradoxical_df,
                    stability_df=ranking_stability_df,
                    settings=settings,
                )
            except Exception as exc:  # pragma: no cover - reporting fallback
                logger.exception("Paradoxical LaTeX section generation failed: %s", exc)

    analysis_dataset_df = build_analysis_dataset(subject_results)
    subject_summary_df = build_subject_summary_dataset(
        target_cluster_df,
        binary_summary_df if not binary_summary_df.empty else None,
    )
    subject_period_summary_df = build_subject_period_summary(merged_df)
    professor_appendix_all_years_df, professor_appendix_by_period_df = build_professor_appendix_tables(
        merged_df,
        settings,
    )

    data_quality_path = settings.output_diagnostics_dir / settings.data_quality_filename
    missingness_path = settings.output_diagnostics_dir / settings.missingness_filename
    cluster_metrics_path = settings.output_metrics_dir / settings.cluster_metrics_filename
    centroids_path = settings.output_centroids_dir / settings.centroids_filename
    target_cluster_path = settings.output_summaries_dir / settings.target_cluster_filename
    target_students_path = settings.output_data_clean_dir / settings.target_cluster_students_filename
    professor_report_path = settings.output_professor_reports_dir / settings.professor_report_filename
    target_professor_roster_path = settings.output_professor_reports_dir / settings.target_professor_roster_filename
    target_professor_students_path = settings.output_professor_reports_dir / settings.target_professor_students_filename
    global_ranking_path = settings.output_professor_reports_dir / settings.global_professor_ranking_filename

    write_dataframe(data_quality_df, data_quality_path)
    write_dataframe(missingness_df, missingness_path)
    write_dataframe(cluster_metrics_df, cluster_metrics_path)
    write_dataframe(centroids_df, centroids_path)
    write_dataframe(target_cluster_df, target_cluster_path)
    write_dataframe(target_students_df, target_students_path)
    write_dataframe(professor_report_df, professor_report_path)
    write_dataframe(target_professor_roster_df, target_professor_roster_path)
    write_dataframe(target_professor_students_df, target_professor_students_path)
    write_dataframe(global_ranking_df, global_ranking_path)

    processed_dataframes = {
        "merged_dataset": select_merged_output_columns(merged_base_df),
        "analysis_dataset": analysis_dataset_df,
        "paradoxical_group_dataset": select_merged_output_columns(merged_df),
        "subject_summary": subject_summary_df,
        "subject_period_summary": subject_period_summary_df,
        "professor_summary_all_years": professor_appendix_all_years_df,
        "professor_summary_by_period": professor_appendix_by_period_df,
        "professor_appendix_all_years": professor_appendix_all_years_df,
        "professor_appendix_by_period": professor_appendix_by_period_df,
        "manual_50_50_8_students": manual_result.students_df,
        "manual_50_50_8_subject_period_summary": manual_result.subject_period_summary_df,
        "manual_50_50_8_professor_summary_by_period": manual_result.professor_summary_by_period_df,
        "manual_50_50_8_professor_summary_all_years": manual_result.professor_summary_all_years_df,
    }
    processed_artifacts: dict[str, Path] = {}
    logger.info("Writing processed dataframes to %s.", settings.processed_data_dir)
    for dataframe_name, dataframe in processed_dataframes.items():
        csv_path = settings.processed_data_dir / f"{dataframe_name}.csv"
        xlsx_path = settings.processed_data_dir / f"{dataframe_name}.xlsx"
        write_dataframe_csv_and_excel(dataframe, csv_path, xlsx_path, sheet_name=dataframe_name)
        processed_artifacts[f"{dataframe_name}_csv_path"] = csv_path
        processed_artifacts[f"{dataframe_name}_xlsx_path"] = xlsx_path
    processed_artifacts.update(write_processed_data_documentation(settings, processed_dataframes))

    manual_copy_dataframes = {
        "manual_50_50_8_students": manual_result.students_df,
        "manual_50_50_8_subject_period_summary": manual_result.subject_period_summary_df,
        "manual_50_50_8_professor_summary_by_period": manual_result.professor_summary_by_period_df,
        "manual_50_50_8_professor_summary_all_years": manual_result.professor_summary_all_years_df,
    }
    for dataframe_name, dataframe in manual_copy_dataframes.items():
        csv_path = settings.output_manual_tables_dir / f"{dataframe_name}.csv"
        xlsx_path = settings.output_manual_tables_dir / f"{dataframe_name}.xlsx"
        write_dataframe_csv_and_excel(dataframe, csv_path, xlsx_path, sheet_name=dataframe_name)
        processed_artifacts[f"{dataframe_name}_copy_csv_path"] = csv_path
        processed_artifacts[f"{dataframe_name}_copy_xlsx_path"] = xlsx_path

    if settings.update_latex_report:
        try:
            processed_artifacts["professor_appendix_latex_path"] = write_professor_appendix_latex(
                all_years_df=professor_appendix_all_years_df,
                by_period_df=professor_appendix_by_period_df,
                settings=settings,
            )
            for subject_code in ("MAT1012", "MAT1022"):
                processed_artifacts[f"professor_appendix_{subject_code}_latex_path"] = (
                    write_subject_professor_appendix_latex(
                        all_years_df=professor_appendix_all_years_df,
                        by_period_df=professor_appendix_by_period_df,
                        subject_code=subject_code,
                        settings=settings,
                    )
                )
            processed_artifacts["manual_50_50_8_mat1012_latex_path"] = write_manual_mat1012_latex_report(
                students_df=manual_result.students_df,
                subject_period_summary_df=manual_result.subject_period_summary_df,
                professor_summary_by_period_df=manual_result.professor_summary_by_period_df,
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - reporting fallback
            logger.exception("Professor appendix LaTeX generation failed: %s", exc)

    presentation_plot_paths: list[Path] = []
    if settings.make_presentation_plots:
        logger.info("Creating presentation plots.")
        try:
            presentation_plot_paths = create_presentation_plots(
                subject_results=subject_results,
                cluster_metrics_df=cluster_metrics_df,
                centroids_df=centroids_df,
                target_cluster_df=target_cluster_df,
                target_professor_roster_df=target_professor_roster_df,
                target_professor_students_df=target_professor_students_df,
                global_ranking_df=global_ranking_df,
                settings=settings,
            )
        except Exception as exc:  # pragma: no cover - visualization fallback
            logger.exception("Presentation plots failed: %s", exc)

    logger.info("Pipeline finished.")
    return {
        "merged_dataset_path": merged_dataset_path,
        "data_quality_path": data_quality_path,
        "missingness_path": missingness_path,
        "cluster_metrics_path": cluster_metrics_path,
        "centroids_path": centroids_path,
        "target_cluster_path": target_cluster_path,
        "target_students_path": target_students_path,
        "professor_report_path": professor_report_path,
        "target_professor_roster_path": target_professor_roster_path,
        "target_professor_students_path": target_professor_students_path,
        "global_ranking_path": global_ranking_path,
        "presentation_plots_dir": settings.output_presentation_plots_dir,
        "presentation_plot_paths": presentation_plot_paths,
        "paradoxical_analysis_dir": settings.output_paradoxical_root_dir,
        "paradoxical_plot_paths": paradoxical_plot_paths,
        "manual_50_50_8_dir": settings.output_manual_root_dir,
        **paradoxical_artifacts,
        "processed_data_dir": settings.processed_data_dir,
        **processed_artifacts,
    }
