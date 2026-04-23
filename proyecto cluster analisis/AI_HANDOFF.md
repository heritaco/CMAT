# AI Handoff - Proyecto Cluster Analisis

Este archivo esta pensado para que otra IA, analista o desarrollador pueda continuar el proyecto sin reconstruir el contexto desde cero.

## Contexto Corto

El proyecto vive en:

```text
proyecto cluster analisis/
```

No asumir que se ejecuta desde el root global del repositorio. El entrypoint esperado es:

```bash
python run_analysis.py
```

El objetivo es identificar, por materia, alumnos con:

```text
Porcentaje_DMU bajo
Porcentaje_GA_GB bajo
CALIFICACION alta
```

Luego se reportan los profesores (`CLAVEPROFESOR`) que concentran alumnos en ese grupo.

Desde la ultima extension existen dos analisis:

- Analisis original de clustering por materia, con filtro `CALIFICACION >= 7.5`.
- Analisis binario/paradojico paralelo, usando todos los casos completos sin prefiltro de calificacion y comparando GMM binario, score de discrepancia y benchmark 40/40/8.

## Estado Actual Del Metodo

La version actual del pipeline:

- usa `GaussianMixture` por default
- evalua `k=2..12`
- usa `random_state=42`
- filtra materias a `MAT1012`, `MAT1022`, `MAT1032`, `MAT1052`, `MAT2012`, `MAT2022`
- usa solo casos completos en `Porcentaje_DMU`, `Porcentaje_GA_GB`, `CALIFICACION`
- excluye del clustering alumnos con `CALIFICACION < 7.5`
- selecciona clusters con estrategia `target_oriented`
- define el cluster objetivo con `S_c = z(CALIFICACION)-z(Porcentaje_DMU)-z(Porcentaje_GA_GB)`
- valida que el cluster objetivo tenga calificacion arriba de la media y ambos porcentajes abajo de la media

El analisis binario/paradojico:

- usa todos los registros con `data_complete_r3 == True`
- estandariza por materia `Porcentaje_DMU`, `Porcentaje_GA_GB`, `CALIFICACION`
- ajusta `GaussianMixture(n_components=2, random_state=42)` por materia
- selecciona el componente objetivo con `S_c = z(CAL)-z(DMU)-z(GAGB)`
- calcula `discrepancy_score = z(CAL)-0.5*z(DMU)-0.5*z(GAGB)`
- compara contra `DMU < 40`, `GA-GB < 40`, `CALIFICACION > 8`
- agrega columnas binarias y z-scores a `merged_dataset.csv`

Configuracion principal:

```text
config/settings.py
```

## Archivos De Entrada

Materias:

```text
../data/onedrive/Archivos2024/Materias estudiantes-profesores 2019-2025 P y O.xlsx
```

Examenes:

```text
../onedrive/RicardoMJ/resultados_examenes.xlsx
```

Hojas relevantes:

- `DMU`
- `GA-GB`

Nota: en ejecuciones previas, Python recibio `PermissionError` al leer el Excel de materias por OneDrive. `io/readers.py` tiene un fallback que copia temporalmente el archivo con PowerShell y lee esa copia.

## Flujo Del Pipeline

1. `io/readers.py`: lee materias y examenes.
2. `preprocessing/cleaning.py`: valida columnas, tipa IDs/años/calificaciones y normaliza.
3. `preprocessing/merging.py`: une materias con DMU y GA-GB.
4. `features/subject_filter.py`: filtra materias, casos completos y elegibilidad por calificacion.
5. `clustering/selection.py`: ajusta candidatos y selecciona `k`.
6. `clustering/target_cluster.py`: selecciona cluster objetivo.
7. `analytics/professor_stats.py`: calcula reportes por profesor.
8. `analytics/target_details.py`: genera archivos accionables de alumnos/profesores objetivo.
9. `visualization/`: genera plots tecnicos y plots de presentacion.
10. `pipeline/main_pipeline.py`: orquesta todo.

Modulos nuevos relevantes:

- `analytics/paradoxical_group.py`: calcula z-scores por materia, GMM binario, score y benchmark.
- `analytics/method_comparison.py`: crea resumenes por materia, overlaps, rankings por profesor y estabilidad.
- `visualization/paradoxical_plots.py`: genera figuras del analisis binario.
- `reporting/latex_report.py`: genera `reportes/seccion_analisis_paradojico.tex`.

## Resultados Actuales

Ultima corrida con filtro `CALIFICACION >= 7.5`:

| Materia | Elegibles | Alumnos objetivo | Fraccion objetivo | Score | Valida calificacion alta | Valida ambos porcentajes bajos |
|---|---:|---:|---:|---:|---|---|
| MAT1012 | 2570 | 272 | 10.58% | 1.539 | Si | Si |
| MAT1022 | 2376 | 71 | 2.99% | 2.617 | No | Si |
| MAT1032 | 1215 | 278 | 22.88% | 1.545 | Si | Si |
| MAT1052 | 16 | 5 | 31.25% | 1.182 | No | Si |
| MAT2012 | 1208 | 171 | 14.16% | 1.234 | Si | Si |
| MAT2022 | 625 | 12 | 1.92% | 2.834 | Si | Si |

Archivos clave generados:

```text
output_cluster_analisis/data_clean/merged_dataset.csv
output_cluster_analisis/data_clean/alumnos_cluster_objetivo.csv
output_cluster_analisis/metrics/cluster_metrics_por_materia.csv
output_cluster_analisis/centroids/centroides_por_materia.csv
output_cluster_analisis/summaries/cluster_objetivo_por_materia.csv
output_cluster_analisis/professor_reports/profesores_por_materia.csv
output_cluster_analisis/professor_reports/profesores_cluster_objetivo_detalle.csv
output_cluster_analisis/professor_reports/alumnos_profesores_cluster_objetivo.csv
output_cluster_analisis/professor_reports/ranking_profesores_global.csv
output_cluster_analisis/presentation_plots/
output_cluster_analisis/paradoxical_analysis/tables/binary_group_summary_by_subject.csv
output_cluster_analisis/paradoxical_analysis/tables/method_comparison_by_subject.csv
output_cluster_analisis/paradoxical_analysis/tables/overlap_between_methods.csv
output_cluster_analisis/paradoxical_analysis/tables/professor_paradoxical_summary.csv
output_cluster_analisis/paradoxical_analysis/tables/professor_paradoxical_global_ranking.csv
output_cluster_analisis/paradoxical_analysis/tables/professor_ranking_stability.csv
output_cluster_analisis/paradoxical_analysis/diagnostics/paradoxical_group_diagnostics.csv
output_cluster_analisis/paradoxical_analysis/figures/
reportes/seccion_analisis_paradojico.tex
```

Conteos actuales:

- `alumnos_cluster_objetivo.csv`: 809 alumnos
- `profesores_cluster_objetivo_detalle.csv`: 140 profesores
- `ranking_profesores_global.csv`: 77 profesores

Conteos actuales del analisis binario/paradojico principal GMM:

| Materia | Completo R3 | GMM objetivo | Fraccion GMM | Benchmark 40/40/8 |
|---|---:|---:|---:|---:|
| MAT1012 | 2598 | 1309 | 50.38% | 41 |
| MAT1052 | 16 | 5 | 31.25% | 0 |
| MAT1022 | 2510 | 2362 | 94.10% | 39 |
| MAT1032 | 1262 | 1215 | 96.28% | 13 |
| MAT2012 | 1252 | 1212 | 96.81% | 20 |
| MAT2022 | 642 | 221 | 34.42% | 19 |

Importantisimo: el GMM binario selecciona grupos demasiado grandes en MAT1022, MAT1032 y MAT2012. Esto no es un error tecnico, sino una senal de que una mezcla de dos componentes puede estar separando un grupo pequeno de excepciones y dejando la mayoria como componente con mayor score. Revisar `paradoxical_group_diagnostics.csv` y `overlap_between_methods.csv` antes de interpretar.

## Profesores Destacados Actuales

Top profesores por materia usando `included_in_ranking == True` y ordenando por `share_cluster_objetivo`:

| Materia | Profesores destacados |
|---|---|
| MAT1012 | 23897, 23824, 23835 |
| MAT1022 | 23823, 22114, 23442 |
| MAT1032 | 23824, 23425, 24086 |
| MAT1052 | 21852, con muestra muy pequeña |
| MAT2012 | 22473, 23452, 23148 |
| MAT2022 | 24013, 22115, 23887 |

Importante: no interpretar estos IDs como evidencia causal. Son profesores asociados descriptivamente al cluster objetivo.

## Comandos Utiles

Ejecutar pipeline:

```bash
python run_analysis.py
```

Correr tests:

```bash
python -m pytest
```

Cambiar metodo a KMeans en PowerShell:

```powershell
$env:SCA_CLUSTERING_METHOD = "kmeans"
python run_analysis.py
```

Cambiar umbral de calificacion:

```powershell
$env:SCA_MINIMUM_GRADE_FOR_CLUSTERING = "8.0"
python run_analysis.py
```

Cambiar rango de `k`:

```powershell
$env:SCA_K_VALUES = "3,4,5,6,7,8,9,10,11,12,13,14,15"
python run_analysis.py
```

Desactivar analisis binario/paradojico:

```powershell
$env:SCA_ENABLE_PARADOXICAL_ANALYSIS = "false"
python run_analysis.py
```

Cambiar benchmark manual:

```powershell
$env:SCA_PARADOXICAL_BASELINE_DMU_THRESHOLD = "35"
$env:SCA_PARADOXICAL_BASELINE_GAGB_THRESHOLD = "35"
$env:SCA_PARADOXICAL_BASELINE_GRADE_THRESHOLD = "8.5"
python run_analysis.py
```

## Decisiones Metodologicas Importantes

Merge:

- match principal: `CLAVEALUMNO == ID` y `anio == Año`
- fallback: año mas cercano dentro de tolerancia
- fallback opcional: alumno con unico registro de examen

Clustering:

- se ajusta en variables originales estandarizadas, no en ICA
- ICA solo se usa para plots 2D
- Plotly 3D usa ejes originales

Filtro:

- `CALIFICACION < 7.5` queda fuera del clustering
- esos alumnos siguen existiendo en `merged_dataset.csv`, pero no en `analysis_df`

Target:

- score objetivo premia calificacion alta y penaliza porcentajes altos
- validacion baja exige ambos porcentajes debajo de la media
- si la validacion falla, se reporta pero no se aborta

## Puntos De Cuidado

- `MAT1052` tiene muy pocos casos elegibles. Cualquier conclusion ahi es fragil.
- `MAT1022` y `MAT1052` actualmente no validan calificacion por encima de la media filtrada, aunque si validan ambos porcentajes bajos.
- En el nuevo analisis binario, el metodo GMM puede seleccionar grupos muy grandes. No asumir automaticamente que todo el grupo es "sospechoso"; usarlo como particion estadistica descriptiva y revisar sensibilidad.
- El ranking global mezcla materias, asi que debe usarse solo como vista descriptiva.
- Revisar siempre denominadores como `total_observaciones_clusterizadas_profesor`.
- No modificar carpetas fuera de `proyecto cluster analisis/` salvo que el usuario lo pida.
- No asumir que outputs son versionados; pueden regenerarse con `python run_analysis.py`.

## Siguientes Pasos Recomendados

1. Revisar si conviene exigir `validation_grade_above_mean == True` estrictamente para seleccionar cluster objetivo.
2. Probar umbrales `CALIFICACION >= 8.0` o `>= 8.5` y comparar estabilidad de profesores.
3. Generar un reporte HTML o PDF consolidado para presentacion.
4. Agregar tablas por profesor con distribuciones por `anio` y `CLAVESESION`.
5. Comparar GMM vs KMeans para ver si los profesores destacados son robustos.
6. Agregar bootstrap o sensibilidad por cohorte para evaluar estabilidad.
7. Probar alternativas al GMM binario para evitar grupos objetivo excesivamente grandes, por ejemplo cortes por colas del `discrepancy_score`, modelos semisupervisados o reglas por cuantiles dentro de materia.
8. Comparar estabilidad de profesores entre analisis original y analisis binario/paradojico.

## Si Una IA Continúa

Antes de editar:

1. Leer `README.md`.
2. Leer `config/settings.py`.
3. Leer `src/student_cluster_analysis/pipeline/main_pipeline.py`.
4. Revisar `output_cluster_analisis/summaries/cluster_objetivo_por_materia.csv`.
5. Revisar `output_cluster_analisis/professor_reports/profesores_cluster_objetivo_detalle.csv`.

Despues de editar:

1. Ejecutar `python -m pytest`.
2. Ejecutar `python run_analysis.py` si el cambio afecta pipeline o outputs.
3. Verificar que se regeneren los CSV clave.
4. Actualizar este archivo si cambia una decision metodologica.
