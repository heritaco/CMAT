# Dataset Info for the CMAT Report

## Data Sources Used

- `Materias estudiantes-profesores 2019-2025 P y O.xlsx`
- `Asesorias2024.xlsx`

These are the only primary raw inputs used by this descriptive module. The analytical data reuse the repository's existing cleaning and imputation logic built from those two workbooks.

## Construction Notes

- `student` means unique `CLAVEALUMNO`.
- `student_classroom_observation` means one cleaned row in materias after merging the report's `VISITAS` variable.
- `classroom_unit` means `CLAVEPROFESOR, CLAVEVARIANTEMATERIA, anio, CLAVESESION`.
- `VISITAS` follows the report pipeline definition: total advisory visits per student from the full asesorias workbook, merged back onto every cleaned materias observation for that student.
- Raw advisory events by year are counted separately from `fecha` in asesorias and should not be confused with the merged `VISITAS` variable.

## Column Roles

| role                      | column               |
| ------------------------- | -------------------- |
| student_id                | CLAVEALUMNO          |
| professor_id              | CLAVEPROFESOR        |
| subject_variant           | CLAVEVARIANTEMATERIA |
| subject_name              | DESCRIBEMATERIA      |
| year                      | anio                 |
| session                   | CLAVESESION          |
| career                    | CLAVECARRERA         |
| visit_count               | VISITAS              |
| raw_grade                 | CALIFICACION         |
| raw_grade_numeric         | CALIFICACION_NUM     |
| mean_imputed_grade        | IMPMEAN              |
| mean_imputed_grade_z      | IMPMEAN_Z            |
| kde_imputed_grade         | IMPKDE               |
| kde_imputed_grade_z       | IMPKDE_Z             |
| student_collapsed_outcome | MEAN_IMPKDE_Z        |
| classroom_unit_id         | CLASSROOM_UNIT_ID    |

## Dataset Overview

| metric                          | value                                                | unit                           | definition                                                                      |
| ------------------------------- | ---------------------------------------------------- | ------------------------------ | ------------------------------------------------------------------------------- |
| materias_source_file            | Materias estudiantes-profesores 2019-2025 P y O.xlsx | file_name                      | Primary materias workbook used by the report pipeline.                          |
| asesorias_source_file           | Asesorias2024.xlsx                                   | file_name                      | Primary asesorias workbook used by the report pipeline.                         |
| raw_rows_materias               | 27788                                                | raw_rows                       | Rows read directly from the materias workbook.                                  |
| raw_rows_asesorias              | 13500                                                | raw_rows                       | Rows read directly from the asesorias workbook.                                 |
| cleaned_student_classroom_rows  | 26140                                                | student_classroom_observations | Rows after applying the existing materias cleaning rules.                       |
| removed_rows_total              | 1648                                                 | rows_removed                   | Total materias rows removed by the tracked cleaning steps.                      |
| removed_rows_share_total        | 0.059                                                | share                          | Share of materias rows removed by cleaning.                                     |
| unique_students_cleaned         | 10413                                                | students                       | Unique students in the cleaned analytical materias dataset.                     |
| unique_professors_cleaned       | 77                                                   | professors                     | Unique professors in the cleaned analytical materias dataset.                   |
| unique_subject_variants_cleaned | 7                                                    | subject_variants               | Unique subject or course variants in the cleaned analytical materias dataset.   |
| unique_sessions_cleaned         | 2                                                    | sessions                       | Unique session labels in the cleaned analytical materias dataset.               |
| unique_years_cleaned            | 7                                                    | years                          | Unique academic years covered by the cleaned materias dataset.                  |
| years_covered                   | 2019, 2020, 2021, 2022, 2023, 2024, 2025             | year_list                      | Sorted list of academic years covered by the cleaned materias dataset.          |
| unique_classroom_units          | 769                                                  | classroom_units                | Unique classroom units defined as professor x subject variant x year x session. |
| student_classroom_observations  | 26140                                                | student_classroom_observations | Rows in the cleaned and imputed analytical dataset used by the report.          |

## Cleaning Summary

| step                                      | rows_before | rows_after | rows_removed | pct_removed_from_previous | pct_removed_from_raw | note                                                                                   |
| ----------------------------------------- | ----------- | ---------- | ------------ | ------------------------- | -------------------- | -------------------------------------------------------------------------------------- |
| raw_input                                 | 27788       | 27788      | 0            | 0.000                     | 0.000                | Raw materias workbook rows before any cleaning.                                        |
| drop_duplicate_student_subject_grade_rows | 27788       | 26645      | 1143         | 0.041                     | 0.041                | Matches clean_materias_df duplicate rule on student, subject variant, and grade token. |
| drop_NUMORDEN_column                      | 26645       | 26645      | 0            | 0.000                     | 0.041                | Column removal only; row count unchanged.                                              |
| drop_missing_professor_id                 | 26645       | 26140      | 505          | 0.019                     | 0.059                | Matches clean_materias_df dropna on CLAVEPROFESOR.                                     |
| cast_professor_id_to_int                  | 26140       | 26140      | 0            | 0.000                     | 0.059                | Type normalization only; row count unchanged.                                          |

## Student Coverage and Visits

- Students in analytical sample: 10413
- Mean visits: 1.234
- Median visits: 0.000
- Zero visits: 71.9%
- At least one visit: 28.1%
- More than three visits: 9.9%
- Maximum observed visits: 116
- Visit Gini coefficient: 0.871
- Top 10% visit share: 76.2%

Selected visit thresholds:

| visits_k | student_count_exact_k | student_prop_exact_k | student_tail_prop_ge_k | student_continuation_prob_ge_k_plus_1_given_ge_k |
| -------- | --------------------- | -------------------- | ---------------------- | ------------------------------------------------ |
| 0        | 7488                  | 71.9%                | 100.0%                 | 28.1%                                            |
| 1        | 1045                  | 10.0%                | 28.1%                  | 64.3%                                            |
| 2        | 494                   | 4.7%                 | 18.1%                  | 73.7%                                            |
| 3        | 350                   | 3.4%                 | 13.3%                  | 74.7%                                            |
| 4        | 208                   | 2.0%                 | 9.9%                   | 79.9%                                            |
| 5        | 145                   | 1.4%                 | 8.0%                   | 82.5%                                            |
| 10       | 54                    | 0.5%                 | 3.1%                   | 83.4%                                            |
| 20       | 6                     | 0.1%                 | 0.8%                   | 92.7%                                            |

## Year-by-Year Description

| anio | n_unique_students | n_unique_professors | n_unique_subject_variants | n_unique_classroom_units | n_student_classroom_observations | student_mean_visits_report_variable | student_median_visits_report_variable | student_prop_zero_visits_report_variable | student_prop_ge_1_visits_report_variable | student_prop_gt_3_visits_report_variable | student_sum_visits_report_variable | raw_asesoria_event_count | rank_students_desc | rank_raw_asesoria_events_desc | rank_prop_gt_3_desc |
| ---- | ----------------- | ------------------- | ------------------------- | ------------------------ | -------------------------------- | ----------------------------------- | ------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------------- | ---------------------------------- | ------------------------ | ------------------ | ----------------------------- | ------------------- |
| 2019 | 2827              | 51                  | 7                         | 108                      | 4260                             | 1.487                               | 0.000                                 | 0.647                                    | 0.353                                    | 0.125                                    | 4203                               | 2864                     | 3                  | 2                             | 4                   |
| 2020 | 3175              | 53                  | 7                         | 131                      | 4724                             | 1.621                               | 0.000                                 | 0.626                                    | 0.374                                    | 0.137                                    | 5147                               | 2220                     | 2                  | 4                             | 3                   |
| 2021 | 3265              | 56                  | 7                         | 147                      | 4856                             | 1.216                               | 0.000                                 | 0.696                                    | 0.304                                    | 0.100                                    | 3970                               | 1722                     | 1                  | 5                             | 5                   |
| 2022 | 2256              | 46                  | 7                         | 111                      | 3389                             | 2.043                               | 0.000                                 | 0.608                                    | 0.392                                    | 0.157                                    | 4609                               | 2474                     | 6                  | 3                             | 2                   |
| 2023 | 2287              | 42                  | 6                         | 106                      | 3269                             | 2.224                               | 0.000                                 | 0.588                                    | 0.412                                    | 0.178                                    | 5086                               | 3093                     | 5                  | 1                             | 1                   |
| 2024 | 2518              | 47                  | 6                         | 114                      | 3842                             | 1.104                               | 0.000                                 | 0.760                                    | 0.240                                    | 0.097                                    | 2779                               | 1127                     | 4                  | 6                             | 6                   |
| 2025 | 1438              | 36                  | 6                         | 52                       | 1800                             | 0.219                               | 0.000                                 | 0.937                                    | 0.063                                    | 0.019                                    | 315                                | 0                        | 7                  | 7                             | 7                   |

## Classroom and Professor Structure

- Classroom units: 769
- Students appearing in multiple classroom units: 65.2%
- Largest classroom size observed: 107
- Median classroom size: 30.000

Largest classroom units:

| classroom_size | year | session | professor_id | subject_name                          |
| -------------- | ---- | ------- | ------------ | ------------------------------------- |
| 107            | 2019 | OTOÑO   | 22114        | CÁLCULO II                            |
| 106            | 2021 | OTOÑO   | 23162        | CÁLCULO II                            |
| 95             | 2020 | OTOÑO   | 23133        | MATEMÁTICAS UNIVERSITARIAS            |
| 93             | 2020 | OTOÑO   | 16164        | ESTADÍSTICA PARA CIENCIAS DE LA SALUD |
| 90             | 2019 | OTOÑO   | 23837        | MATEMÁTICAS UNIVERSITARIAS            |

Professors with the most students:

| CLAVEPROFESOR | n_unique_students | n_classroom_units | mean_classroom_size |
| ------------- | ----------------- | ----------------- | ------------------- |
| 23452         | 860               | 25                | 38.760              |
| 23411         | 689               | 26                | 29.692              |
| 23852         | 661               | 19                | 37.842              |
| 24097         | 660               | 21                | 34.286              |
| 23152         | 657               | 23                | 31.565              |
| 21852         | 646               | 23                | 28.304              |
| 23148         | 635               | 21                | 33.714              |
| 22446         | 630               | 23                | 29.652              |
| 23978         | 594               | 21                | 30.381              |
| 23149         | 581               | 17                | 36.353              |

## Grades and Missingness

- Raw numeric grade missing count: 3217
- Raw numeric grade missing proportion: 12.3%
- Imputed observations in the report pipeline: 3217

Grade variable definitions:

| variable         | definition                                                                                                |
| ---------------- | --------------------------------------------------------------------------------------------------------- |
| CALIFICACION     | Raw grade token from the materias file. May contain non-numeric codes.                                    |
| CALIFICACION_NUM | Numeric projection of CALIFICACION. Non-numeric tokens are coerced to NaN.                                |
| IMPMEAN          | Classroom-level mean imputation using observed numeric grades at or below 7.5.                            |
| IMPMEAN_Z        | Within-classroom Z-score of IMPMEAN.                                                                      |
| IMPKDE           | Classroom-level KDE imputation from observed grades at or below 7.4/7.5, following the existing pipeline. |
| IMPKDE_Z         | Within-classroom Z-score of IMPKDE.                                                                       |
| MEAN_IMPKDE_Z    | Student-level mean of IMPKDE_Z across all student-classroom observations.                                 |

## Key Descriptive Findings

- The cleaned analytical sample spans 7 years: 2019, 2020, 2021, 2022, 2023, 2024, 2025.
- The year with the most cleaned students was 2021, while the year with the most raw advisory events was 2023.
- Visit concentration is substantial: the top 10% of students account for 76.2% of all merged visits.
- Missing raw numeric grades account for 12.3% of student-classroom observations.

## Generated Tables

- `report_assets/tables/source_data_overview.csv`
- `report_assets/tables/cleaning_summary.csv`
- `report_assets/tables/student_visit_distribution_exact.csv`
- `report_assets/tables/student_visit_distribution_tail.csv`
- `report_assets/tables/student_visit_thresholds.csv`
- `report_assets/tables/visit_summary.csv`
- `report_assets/tables/visits_by_year.csv`
- `report_assets/tables/asesorias_raw_year_summary.csv`
- `report_assets/tables/classroom_unit_summary.csv`
- `report_assets/tables/classroom_size_distribution.csv`
- `report_assets/tables/professor_summary.csv`
- `report_assets/tables/subject_summary.csv`
- `report_assets/tables/student_summary.csv`
- `report_assets/tables/grade_missingness_summary.csv`
- `report_assets/tables/raw_grade_non_numeric_tokens.csv`
- `report_assets/tables/threshold_summaries.csv`
- `report_assets/tables/concentration_summary.csv`
- `report_assets/tables/lorenz_visits.csv`
- `report_assets/tables/student_classroom_units_per_student_distribution.csv`
- `report_assets/tables/top_students_by_visits.csv`

## Generated Figures

- `report_assets/figures/visits_histogram.pdf`
- `report_assets/figures/visits_histogram.png`
- `report_assets/figures/visits_histogram_low_counts.pdf`
- `report_assets/figures/visits_histogram_low_counts.png`
- `report_assets/figures/visits_ecdf.pdf`
- `report_assets/figures/visits_ecdf.png`
- `report_assets/figures/visits_tail_curve.pdf`
- `report_assets/figures/visits_tail_curve.png`
- `report_assets/figures/visits_continuation_curve.pdf`
- `report_assets/figures/visits_continuation_curve.png`
- `report_assets/figures/visits_by_year.pdf`
- `report_assets/figures/visits_by_year.png`
- `report_assets/figures/classroom_size_distribution.pdf`
- `report_assets/figures/classroom_size_distribution.png`
- `report_assets/figures/visits_lorenz_curve.pdf`
- `report_assets/figures/visits_lorenz_curve.png`

## Generated LaTeX Snippets

- `report_assets/tex/descriptive_summary.tex`
- `report_assets/tex/core_counts_table.tex`
- `report_assets/tex/visits_distribution_table.tex`
- `report_assets/tex/year_summary_table.tex`
- `report_assets/tex/cleaning_summary_table.tex`
- `report_assets/tex/concentration_table.tex`
- `report_assets/tex/grade_overview_table.tex`
- `report_assets/tex/grade_tokens_table.tex`

