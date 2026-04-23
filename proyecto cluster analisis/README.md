# Proyecto Cluster Analisis

Proyecto modular en Python para detectar, por materia, grupos de alumnos en el espacio:

```text
X_i = (Porcentaje_DMU_i, Porcentaje_GA_GB_i, CALIFICACION_i)
```

El objetivo analitico es encontrar alumnos con el patron:

```text
porcentajes bajos en DMU y GA-GB, pero calificacion alta en la materia
```

Despues de identificar ese grupo, el pipeline reporta los profesores asociados (`CLAVEPROFESOR`) y resume las distribuciones de sus alumnos.

Este proyecto vive dentro de un repositorio mayor. Por eso esta encapsulado en la carpeta `proyecto cluster analisis/`, con configuracion propia, paquete Python propio y salidas propias.

## Estado Actual

La version actual del pipeline ya incorpora dos decisiones importantes:

- El clustering solo se ajusta con alumnos que tienen observaciones completas en `Porcentaje_DMU`, `Porcentaje_GA_GB` y `CALIFICACION`.
- Ademas, se excluyen del clustering alumnos con `CALIFICACION < 7.5`.

El cluster objetivo se selecciona buscando maximizar:

```text
S_c = z(CALIFICACION)_c - z(Porcentaje_DMU)_c - z(Porcentaje_GA_GB)_c
```

La validacion metodologica revisa que el cluster elegido tenga:

- centroide de `CALIFICACION` por encima de la media de la materia filtrada
- centroide de `Porcentaje_DMU` por debajo de la media de la materia filtrada
- centroide de `Porcentaje_GA_GB` por debajo de la media de la materia filtrada

Resultados de la ultima corrida generada:

| Materia | Filas completas R3 | Excluidas por calificacion < 7.5 | Filas elegibles | Alumnos cluster objetivo | Fraccion objetivo | Score objetivo | Valida calificacion alta | Valida ambos porcentajes bajos |
|---|---:|---:|---:|---:|---:|---:|---|---|
| MAT1012 | 2598 | 28 | 2570 | 272 | 10.58% | 1.539 | Si | Si |
| MAT1022 | 2510 | 134 | 2376 | 71 | 2.99% | 2.617 | No | Si |
| MAT1032 | 1262 | 47 | 1215 | 278 | 22.88% | 1.545 | Si | Si |
| MAT1052 | 16 | 0 | 16 | 5 | 31.25% | 1.182 | No | Si |
| MAT2012 | 1252 | 44 | 1208 | 171 | 14.16% | 1.234 | Si | Si |
| MAT2022 | 642 | 17 | 625 | 12 | 1.92% | 2.834 | Si | Si |

Nota importante: `MAT1022` y `MAT1052` tienen clusters objetivo con ambos porcentajes bajos, pero el centroide de calificacion no queda por encima de la media filtrada de su materia. El pipeline no aborta en estos casos; deja la advertencia en los reportes para que se revise metodologicamente.

## Extension Actual: Analisis Binario Paradojico

Ademas del clustering original, el proyecto incluye ahora un segundo analisis paralelo, mas estadistico y sin el prefiltro `CALIFICACION >= 7.5`.

Este analisis usa todos los alumnos con datos completos en:

- `Porcentaje_DMU`
- `Porcentaje_GA_GB`
- `CALIFICACION`

Para cada materia, estandariza las tres variables dentro de la materia y ajusta una mezcla gaussiana binaria:

```text
GaussianMixture(n_components=2, random_state=42)
```

El componente objetivo se selecciona con:

```text
S_c = z(CALIFICACION)_c - z(Porcentaje_DMU)_c - z(Porcentaje_GA_GB)_c
```

Tambien se comparan dos metodos adicionales:

- `score`: score individual interpretable `D_i = z(CAL_i) - 0.5*z(DMU_i) - 0.5*z(GAGB_i)` y mezcla univariada.
- `baseline`: regla manual `DMU < 40`, `GA-GB < 40`, `CALIFICACION > 8`, solo como benchmark secundario.

Este segundo analisis agrega columnas a `merged_dataset.csv`:

- `binary_group_gmm`
- `binary_group_score`
- `binary_group_baseline_40_40_8`
- `discrepancy_score`
- `subject_z_dmu`
- `subject_z_gagb`
- `subject_z_calificacion`
- `target_component_score`
- `is_paradoxical_group_main`

Salidas nuevas:

- `output_cluster_analisis/paradoxical_analysis/tables/binary_group_summary_by_subject.csv`
- `output_cluster_analisis/paradoxical_analysis/tables/method_comparison_by_subject.csv`
- `output_cluster_analisis/paradoxical_analysis/tables/overlap_between_methods.csv`
- `output_cluster_analisis/paradoxical_analysis/tables/professor_paradoxical_summary.csv`
- `output_cluster_analisis/paradoxical_analysis/tables/professor_paradoxical_global_ranking.csv`
- `output_cluster_analisis/paradoxical_analysis/tables/professor_ranking_stability.csv`
- `output_cluster_analisis/paradoxical_analysis/diagnostics/paradoxical_group_diagnostics.csv`
- `output_cluster_analisis/paradoxical_analysis/figures/`
- `reportes/seccion_analisis_paradojico.tex`

Hallazgo importante de la corrida actual: el GMM binario selecciona grupos muy grandes en algunas materias, especialmente `MAT1022`, `MAT1032` y `MAT2012`. Esto queda marcado con advertencias de tamano grande y debe interpretarse como posible inestabilidad de la particion binaria de dos componentes. El benchmark manual selecciona grupos mucho mas pequenos, por lo que el solapamiento entre metodos es bajo.

## Extension Actual: Datos Procesados Y Apendices De Profesores

El pipeline exporta ahora una capa estable de dataframes procesados en:

```text
data/datos_procesados/
```

Cada dataframe principal se guarda en CSV y Excel (`.xlsx`). Los archivos mas importantes son:

- `merged_dataset.csv` / `.xlsx`: dataset integrado base despues de limpieza y merge.
- `analysis_dataset.csv` / `.xlsx`: dataset exacto usado por el clustering principal.
- `paradoxical_group_dataset.csv` / `.xlsx`: dataset enriquecido con columnas del analisis binario/paradojico.
- `subject_summary.csv` / `.xlsx`: resumen por materia.
- `subject_period_summary.csv` / `.xlsx`: resumen por materia y periodo `(anio, CLAVESESION)`.
- `professor_summary_all_years.csv` / `.xlsx`: tabla por materia y profesor agregando todos los anios.
- `professor_summary_by_period.csv` / `.xlsx`: tabla por materia, periodo y profesor.
- `professor_appendix_all_years.csv` / `.xlsx`: tabla canonica que alimenta el Apendice A del reporte LaTeX.
- `professor_appendix_by_period.csv` / `.xlsx`: tabla canonica que alimenta el Apendice B del reporte LaTeX.
- `README.md` y `data_dictionary.md`: documentacion de datos procesados, diccionario de variables y trazabilidad.

El reporte LaTeX incluye ahora `reportes/apendice_tablas_profesores.tex`, generado automaticamente desde `professor_appendix_all_years` y `professor_appendix_by_period`. El apendice se organiza en:

- Apendice A: tablas globales por materia.
- Apendice B: tablas por materia y periodo observado `(anio, CLAVESESION)`.

El desglose temporal no hardcodea sesiones. Si el archivo trae otros valores de `CLAVESESION`, apareceran automaticamente en las subsecciones y en las tablas exportadas.

## Estructura

```text
proyecto cluster analisis/
├── README.md
├── AI_HANDOFF.md
├── requirements.txt
├── pyproject.toml
├── run_analysis.py
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── .gitkeep
│   └── datos_procesados/
├── output_cluster_analisis/
├── reportes/
├── src/
│   └── student_cluster_analysis/
│       ├── analytics/
│       ├── clustering/
│       ├── features/
│       ├── io/
│       ├── pipeline/
│       ├── preprocessing/
│       ├── visualization/
│       └── entities.py
└── tests/
```

Responsabilidades principales:

| Ruta | Responsabilidad |
|---|---|
| `run_analysis.py` | Entry point para ejecutar todo con `python run_analysis.py`. |
| `config/settings.py` | Configuracion centralizada de rutas, materias, parametros de clustering, filtros y nombres de salidas. |
| `src/student_cluster_analysis/io/` | Lectura de Excel y escritura de CSV, HTML y PNG. |
| `src/student_cluster_analysis/preprocessing/` | Limpieza, tipado, validacion de columnas y merge entre materias y examenes. |
| `src/student_cluster_analysis/features/` | Filtros por materia, casos completos y elegibilidad para clustering. |
| `src/student_cluster_analysis/clustering/` | Escalado, modelos GMM/KMeans, seleccion de `k` y cluster objetivo. |
| `src/student_cluster_analysis/analytics/` | Diagnosticos, resumenes, estadisticas por profesor y archivos detalle. |
| `src/student_cluster_analysis/analytics/paradoxical_group.py` | Etiquetado binario GMM/score/benchmark para el nuevo analisis paradojico. |
| `src/student_cluster_analysis/analytics/method_comparison.py` | Comparacion entre metodos, overlaps, rankings y diagnosticos del analisis binario. |
| `src/student_cluster_analysis/analytics/processed_data.py` | Construccion de dataframes procesados, tablas de apendice y resumenes por periodo. |
| `src/student_cluster_analysis/visualization/` | Plots 2D ICA, plots 3D Plotly y plots explicativos para presentacion. |
| `src/student_cluster_analysis/visualization/paradoxical_plots.py` | Figuras especificas del analisis binario/paradojico. |
| `src/student_cluster_analysis/reporting/` | Generacion de secciones LaTeX auxiliares y documentacion de datos procesados. |
| `src/student_cluster_analysis/pipeline/main_pipeline.py` | Orquestacion completa del flujo. |
| `tests/` | Pruebas minimas de columnas, merge, target cluster y smoke test. |

## Instalacion

Desde la carpeta `proyecto cluster analisis`:

```bash
python -m pip install -r requirements.txt
```

Dependencias principales:

- `pandas`
- `openpyxl`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `plotly`
- `pytest`

## Ejecucion

Desde la carpeta `proyecto cluster analisis`:

```bash
python run_analysis.py
```

El comando ejecuta el pipeline completo:

1. lee los archivos de entrada
2. limpia columnas y tipos
3. une materias con resultados DMU y GA-GB
4. filtra materias objetivo
5. filtra casos completos y `CALIFICACION >= 7.5`
6. ajusta clustering por materia
7. selecciona el cluster objetivo por materia
8. identifica profesores asociados
9. genera CSV, Excel, HTML, PNG, diagnosticos, README de datos, diccionario y logs
10. genera secciones LaTeX auxiliares, incluido el apendice completo de profesores

Para correr pruebas:

```bash
python -m pytest
```

## Archivos De Entrada

Por default, `config/settings.py` apunta a:

```text
../data/onedrive/Archivos2024/Materias estudiantes-profesores 2019-2025 P y O.xlsx
../onedrive/RicardoMJ/resultados_examenes.xlsx
```

El primer archivo contiene materias, alumnos, profesores y calificaciones.

Columnas relevantes:

- `CLAVEALUMNO`
- `CLAVECARRERA`
- `anio`
- `CLAVESESION`
- `NUMORDEN`
- `CLAVEVARIANTEMATERIA`
- `DESCRIBEMATERIA`
- `CALIFICACION`
- `CLAVEPROFESOR`

El segundo archivo contiene las hojas:

- `DMU`
- `GA-GB`

Columnas relevantes en `DMU`:

- `ID`
- `Año`
- `Porcentaje_DMU`

Columnas relevantes en `GA-GB`:

- `ID`
- `Año`
- `Porcentaje_GA_GB`

## Materias Analizadas

Las materias se configuran en `config/settings.py`:

```python
subjects=(
    "MAT1012",
    "MAT1022",
    "MAT1032",
    "MAT1052",
    "MAT2012",
    "MAT2022",
)
```

## Configuracion

Toda la configuracion modificable vive en `config/settings.py`.

Parametros clave actuales:

| Parametro | Valor default | Descripcion |
|---|---:|---|
| `feature_columns` | `Porcentaje_DMU`, `Porcentaje_GA_GB`, `CALIFICACION` | Variables del espacio R3. |
| `clustering_method` | `gmm` | Modelo principal. Tambien se soporta `kmeans`. |
| `k_values` | `2..12` | Valores candidatos de `k`. |
| `selection_strategy` | `target_oriented` | Selecciona `k` priorizando el patron buscado. |
| `preferred_min_clusters` | `4` | Intenta evitar soluciones demasiado gruesas. |
| `minimum_grade_for_clustering` | `7.5` | Excluye alumnos con calificacion menor al ajustar clusters. |
| `random_state` | `42` | Reproducibilidad. |
| `merge_year_tolerance` | `1` | Tolerancia para match por año cercano. |
| `min_cluster_size` | `5` | Tamano minimo absoluto por cluster. |
| `min_cluster_fraction` | `0.005` | Fraccion minima por cluster para candidatos validos. |
| `target_cluster_min_fraction` | `0.01` | Fraccion minima deseada del cluster objetivo. |
| `enable_paradoxical_analysis` | `True` | Activa el analisis binario/paradojico. |
| `paradoxical_main_method` | `gmm` | Metodo principal del nuevo analisis binario. |
| `paradoxical_baseline_dmu_threshold` | `40` | Umbral DMU del benchmark manual. |
| `paradoxical_baseline_gagb_threshold` | `40` | Umbral GA-GB del benchmark manual. |
| `paradoxical_baseline_grade_threshold` | `8` | Umbral de calificacion del benchmark manual. |
| `paradoxical_max_group_fraction_warning` | `0.80` | Advertencia cuando el grupo GMM principal es demasiado grande. |
| `min_students_per_professor` | `10` | Minimo para ranking por profesor. |
| `min_students_per_professor_relaxed` | `5` | Umbral relajado cuando hay pocos casos. |
| `high_data_loss_threshold` | `0.30` | Umbral para advertencias de perdida de muestra. |

Variables de entorno soportadas:

- `SCA_MATERIAS_PATH`
- `SCA_EXAMENES_PATH`
- `SCA_OUTPUT_ROOT`
- `SCA_PROCESSED_DATA_DIR`
- `SCA_CLUSTERING_METHOD`
- `SCA_K_VALUES`
- `SCA_SELECTION_STRATEGY`
- `SCA_PREFERRED_MIN_CLUSTERS`
- `SCA_MINIMUM_GRADE_FOR_CLUSTERING`
- `SCA_MERGE_YEAR_TOLERANCE`
- `SCA_MIN_CLUSTER_SIZE`
- `SCA_MIN_CLUSTER_FRACTION`
- `SCA_TARGET_CLUSTER_MIN_FRACTION`
- `SCA_ENABLE_PARADOXICAL_ANALYSIS`
- `SCA_PARADOXICAL_MAIN_METHOD`
- `SCA_PARADOXICAL_BASELINE_DMU_THRESHOLD`
- `SCA_PARADOXICAL_BASELINE_GAGB_THRESHOLD`
- `SCA_PARADOXICAL_BASELINE_GRADE_THRESHOLD`
- `SCA_PARADOXICAL_MAX_GROUP_FRACTION_WARNING`
- `SCA_MIN_STUDENTS_PER_PROFESSOR`
- `SCA_MIN_STUDENTS_PER_PROFESSOR_RELAXED`
- `SCA_MAKE_ICA_PLOTS`
- `SCA_MAKE_PLOTLY_PLOTS`
- `SCA_MAKE_PRESENTATION_PLOTS`
- `SCA_PRESENTATION_TOP_N_PROFESSORS`

Ejemplo para correr con KMeans:

```powershell
$env:SCA_CLUSTERING_METHOD = "kmeans"
python run_analysis.py
```

Ejemplo para cambiar el umbral de calificacion:

```powershell
$env:SCA_MINIMUM_GRADE_FOR_CLUSTERING = "8.0"
python run_analysis.py
```

## Limpieza E Integracion

La lectura empieza en `io/readers.py`.

Comportamiento relevante:

- Se leen materias y las hojas `DMU` y `GA-GB`.
- Si OneDrive bloquea la lectura directa del Excel de materias, el lector intenta crear una copia temporal por PowerShell y leer esa copia.
- No se copian archivos pesados al proyecto.

La limpieza esta en `preprocessing/cleaning.py`.

Reglas importantes:

- Se validan columnas requeridas.
- Se normalizan identificadores y años.
- `CALIFICACION_RAW` conserva el valor original.
- `CALIFICACION` se convierte a numerico.
- Se eliminan duplicados razonables prefiriendo filas con calificacion numerica.

El merge esta en `preprocessing/merging.py`.

Estrategia jerarquica:

1. match exacto por `CLAVEALUMNO` + `anio`
2. fallback por `CLAVEALUMNO` usando el año de examen mas cercano dentro de `merge_year_tolerance`
3. fallback opcional por `CLAVEALUMNO` si el alumno tiene una sola fila de examen en la fuente

El dataset final conserva auditoria del match:

- `match_type_dmu`
- `matched_exam_year_dmu`
- `matched_year_gap_dmu`
- `match_type_gagb`
- `matched_exam_year_gagb`
- `matched_year_gap_gagb`

## Filtro Para Clustering

El clustering no usa todos los registros del dataset unido.

Primero se exige completitud:

```text
Porcentaje_DMU no nulo
Porcentaje_GA_GB no nulo
CALIFICACION no nula
```

Despues se exige:

```text
CALIFICACION >= minimum_grade_for_clustering
```

Con el default actual:

```text
CALIFICACION >= 7.5
```

El dataset `merged_dataset.csv` incluye:

- `data_complete_r3`
- `passes_minimum_grade_for_clustering`
- `eligible_for_clustering`

Solo `eligible_for_clustering == True` entra al clustering.

## Clustering

El clustering se ejecuta por materia.

Flujo:

1. Tomar alumnos elegibles de una materia.
2. Escalar `Porcentaje_DMU`, `Porcentaje_GA_GB` y `CALIFICACION` con `StandardScaler`.
3. Ajustar modelos candidatos para cada `k` configurado.
4. Calcular metricas de calidad.
5. Calcular, para cada candidato, el cluster que maximiza el score objetivo.
6. Seleccionar el mejor candidato segun estrategia configurada.

Metricas calculadas:

- `silhouette_score`
- `calinski_harabasz_score`
- `davies_bouldin_score`
- `AIC`, si el modelo es GMM
- `BIC`, si el modelo es GMM

La estrategia default `target_oriented` prioriza el objetivo sustantivo:

- prefiere candidatos validos con `k >= preferred_min_clusters`
- cuando existe, prefiere candidatos cuyo cluster objetivo valide calificacion alta y ambos porcentajes bajos
- maximiza `S_c`
- usa las metricas de calidad para desempatar

La estrategia alternativa `quality` prioriza metricas clasicas:

- maximiza silhouette
- desempata con Calinski-Harabasz
- minimiza Davies-Bouldin
- minimiza BIC y AIC
- prefiere menor `k` al final

## Cluster Objetivo

Para cada cluster `c`, se calcula:

```text
S_c = z(CALIFICACION)_c - z(Porcentaje_DMU)_c - z(Porcentaje_GA_GB)_c
```

El cluster objetivo es:

```text
c* = argmax_c S_c
```

La interpretacion es:

- mayor `z(CALIFICACION)` aumenta el score
- mayor `z(Porcentaje_DMU)` reduce el score
- mayor `z(Porcentaje_GA_GB)` reduce el score

En otras palabras, el score favorece clusters con calificacion alta y porcentajes bajos.

Validaciones adicionales:

- `validation_grade_above_mean`: el centroide de calificacion del cluster objetivo supera la media filtrada de la materia
- `validation_low_exam_score`: los centroides de DMU y GA-GB estan ambos por debajo de las medias filtradas de la materia

Si alguna validacion falla, el resultado no se descarta automaticamente. Se reporta en `cluster_objetivo_por_materia.csv` y en las advertencias.

## Analisis Por Profesor

Una vez identificado el cluster objetivo por materia, el pipeline calcula estadisticas por:

```text
(CLAVEVARIANTEMATERIA, CLAVEPROFESOR)
```

Metricas principales:

- total de observaciones del profesor en la fuente
- total de observaciones clusterizadas del profesor
- alumnos unicos clusterizados
- alumnos del profesor en el cluster objetivo
- proporcion del profesor en cluster objetivo
- distribucion de clusters de sus alumnos
- años observados
- sesiones observadas
- media, mediana, desviacion estandar y cuantiles de `CALIFICACION`
- media, mediana, desviacion estandar y cuantiles de `Porcentaje_DMU`
- media, mediana, desviacion estandar y cuantiles de `Porcentaje_GA_GB`

Los rankings usan `min_students_per_professor`.

Si una materia tiene pocos casos, el pipeline puede usar `min_students_per_professor_relaxed`.

## Salidas

Todas las salidas se escriben en:

```text
output_cluster_analisis/
```

Archivos principales:

| Archivo | Uso |
|---|---|
| `data_clean/merged_dataset.csv` | Dataset unido, con flags de completitud y elegibilidad. |
| `data_clean/alumnos_cluster_objetivo.csv` | Alumnos exactos que caen en el cluster objetivo. |
| `metrics/cluster_metrics_por_materia.csv` | Metricas de todos los candidatos de clustering por materia. |
| `centroids/centroides_por_materia.csv` | Centroides originales y estandarizados del modelo seleccionado. |
| `summaries/cluster_objetivo_por_materia.csv` | Resumen del cluster objetivo por materia. |
| `professor_reports/profesores_por_materia.csv` | Estadisticas por materia y profesor. |
| `professor_reports/profesores_cluster_objetivo_detalle.csv` | Profesores que aparecen en el cluster objetivo, con lista de alumnos objetivo. |
| `professor_reports/alumnos_profesores_cluster_objetivo.csv` | Todos los alumnos clusterizados de profesores con al menos un alumno objetivo. |
| `professor_reports/ranking_profesores_global.csv` | Ranking global descriptivo de profesores. |
| `diagnostics/data_quality_report.csv` | Auditoria general del pipeline. |
| `diagnostics/missingness_por_materia.csv` | Perdida de muestra y faltantes por materia. |
| `paradoxical_analysis/tables/*.csv` | Tablas del analisis binario: resumen, overlap, profesores y estabilidad. |
| `paradoxical_analysis/figures/` | Figuras globales y por materia del analisis binario. |
| `paradoxical_analysis/diagnostics/paradoxical_group_diagnostics.csv` | Diagnosticos del metodo GMM binario. |
| `logs/pipeline.log` | Log de ejecucion. |

Datos procesados estables:

| Archivo | Uso |
|---|---|
| `data/datos_procesados/merged_dataset.csv` y `.xlsx` | Dataset maestro integrado despues de limpieza y merge. |
| `data/datos_procesados/analysis_dataset.csv` y `.xlsx` | Filas usadas para clustering principal, con etiquetas de cluster. |
| `data/datos_procesados/paradoxical_group_dataset.csv` y `.xlsx` | Dataset enriquecido para analisis binario/paradojico. |
| `data/datos_procesados/subject_summary.csv` y `.xlsx` | Tabla por materia. |
| `data/datos_procesados/subject_period_summary.csv` y `.xlsx` | Tabla por materia y periodo `(anio, CLAVESESION)`. |
| `data/datos_procesados/professor_summary_all_years.csv` y `.xlsx` | Profesores por materia agregando todos los anios. |
| `data/datos_procesados/professor_summary_by_period.csv` y `.xlsx` | Profesores por materia y periodo. |
| `data/datos_procesados/professor_appendix_all_years.csv` y `.xlsx` | Fuente del Apendice A LaTeX. |
| `data/datos_procesados/professor_appendix_by_period.csv` y `.xlsx` | Fuente del Apendice B LaTeX. |
| `data/datos_procesados/README.md` | Trazabilidad, transformaciones y columnas por dataframe. |
| `data/datos_procesados/data_dictionary.md` | Diccionario de variables. |

Reportes LaTeX auxiliares:

| Archivo | Uso |
|---|---|
| `reportes/seccion_analisis_paradojico.tex` | Seccion del analisis binario/paradojico. |
| `reportes/apendice_tablas_profesores.tex` | Apendices completos de profesores por materia y por periodo. |

Visualizaciones:

| Carpeta | Contenido |
|---|---|
| `plots_2d_ica/` | PNG por materia con proyeccion ICA 2D. ICA solo se usa para visualizar. |
| `plots_3d_plotly/` | HTML interactivo por materia con ejes originales. |
| `presentation_plots/` | Graficos explicativos para presentacion. |

Plots de presentacion:

- `01_resumen_cluster_objetivo_por_materia.png`
- `02_contraste_centroide_objetivo_vs_media.png`
- `03_distribuciones_cluster_objetivo_vs_resto.png`
- `04_top_profesores_por_materia.png`
- `05_ranking_global_profesores.png`
- `06_distribuciones_alumnos_profesores_destacados.png`

## Como Leer Los Resultados

Para empezar rapido:

1. Abrir `summaries/cluster_objetivo_por_materia.csv`.
2. Revisar `validation_grade_above_mean` y `validation_low_exam_score`.
3. Abrir `data_clean/alumnos_cluster_objetivo.csv` para ver los alumnos del grupo.
4. Abrir `professor_reports/profesores_cluster_objetivo_detalle.csv` para ver los profesores asociados.
5. Abrir `presentation_plots/04_top_profesores_por_materia.png` para una vista presentable.
6. Abrir `data/datos_procesados/professor_appendix_all_years.xlsx` y `professor_appendix_by_period.xlsx` para auditoria completa de profesores.
7. Revisar `data/datos_procesados/README.md` y `data_dictionary.md` para trazabilidad y significado de variables.

Columnas utiles en `profesores_cluster_objetivo_detalle.csv`:

- `CLAVEVARIANTEMATERIA`
- `CLAVEPROFESOR`
- `share_cluster_objetivo`
- `observaciones_cluster_objetivo_profesor`
- `total_observaciones_clusterizadas_profesor`
- `alumnos_cluster_objetivo_ids`
- `target_CALIFICACION_mean`
- `target_Porcentaje_DMU_mean`
- `target_Porcentaje_GA_GB_mean`

## Pruebas

Ejecutar:

```bash
python -m pytest
```

Las pruebas cubren:

- validacion de columnas requeridas
- merge basico con fallback por año cercano
- seleccion de cluster objetivo en un caso sintetico
- smoke test del pipeline con inputs sinteticos

La advertencia `ConvergenceWarning` en el smoke test sintetico puede aparecer porque el conjunto de prueba tiene puntos duplicados. No implica falla del pipeline real.

## Advertencias Metodologicas

- El clustering describe patrones; no prueba causalidad.
- Una alta proporcion de alumnos de un profesor en el cluster objetivo no demuestra inflacion de calificaciones.
- El ranking global mezcla materias distintas y debe leerse solo como resumen descriptivo.
- El tamano muestral por profesor importa; revisar denominadores antes de interpretar.
- Los resultados pueden variar por cohorte (`anio`) y por sesion (`CLAVESESION`).
- Las tablas por periodo usan siempre el par observado `(anio, CLAVESESION)`; no asumen nombres fijos de periodo.
- El filtro `CALIFICACION >= 7.5` cambia la poblacion de analisis; las conclusiones aplican a alumnos con calificacion aprobatoria/alta segun ese umbral.
- `MAT1052` tiene muy pocos casos completos y elegibles; interpretar con especial cautela.
- `MAT1022` y `MAT1052` actualmente no validan calificacion del centroide por encima de la media filtrada, aunque si validan ambos porcentajes bajos.
- ICA se usa solo para visualizacion, no para construir clusters.
- Los clusters se ajustan en el espacio R3 original estandarizado, no en la proyeccion ICA.

## Para Continuar El Trabajo

Leer tambien:

```text
AI_HANDOFF.md
```

Ese archivo resume decisiones, comandos, archivos clave, resultados actuales y posibles siguientes pasos para que otra persona o una IA pueda continuar sin reconstruir todo el contexto.
