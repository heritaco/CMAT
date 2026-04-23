from __future__ import annotations


def test_core_imports() -> None:
    from config.settings import get_settings
    from visitas_analysis.pipeline.main_pipeline import run_visitas_pipeline
    from visitas_analysis.reporting.report_assets import generate_report_assets

    assert get_settings().project_root.name == "proyecto visitas"
    assert callable(run_visitas_pipeline)
    assert callable(generate_report_assets)
