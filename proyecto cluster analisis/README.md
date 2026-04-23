# Proyecto Cluster Analisis

Proyecto modular en Python para analizar, por materia, grupos de alumnos en el espacio:

- `Porcentaje_DMU`
- `Porcentaje_GA_GB`
- `CALIFICACION`

y después identificar a los profesores cuyos alumnos aparecen con mayor frecuencia en el cluster de:

`calificación alta en la materia, pero porcentajes bajos en DMU y GA-GB`.

## Estructura

- `run_analysis.py`: entrypoint principal.
- `config/settings.py`: configuración centralizada.
- `src/student_cluster_analysis/`: paquete principal.
- `tests/`: pruebas mínimas.
- `output_cluster_analisis/`: resultados del pipeline.

Submódulos del paquete:

- `io/`: lectura y escritura.
- `preprocessing/`: limpieza y merge.
- `features/`: filtrado por materias y selección de columnas.
- `clustering/`: escalado, ajuste, selección de `k`, cluster objetivo.
- `analytics/`: métricas por profesor, resúmenes y diagnósticos.
- `visualization/`: gráficos ICA 2D y Plotly 3D.
- `pipeline/`: orquestación completa.

## Instalación

Desde la raíz de este proyecto:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Desde la carpeta `proyecto cluster analisis`:

```bash
python run_analysis.py
```

## Archivos de entrada esperados

Por default, `config/settings.py` apunta a:

- `../data/onedrive/Archivos2024/Materias estudiantes-profesores 2019-2025 P y O.xlsx`
- `../onedrive/RicardoMJ/resultados_examenes.xlsx`

Hojas esperadas en el segundo archivo:

- `DMU`
- `GA-GB`

Las rutas y parámetros se pueden ajustar desde `config/settings.py` o vía variables de entorno:

- `SCA_MATERIAS_PATH`
- `SCA_EXAMENES_PATH`
- `SCA_OUTPUT_ROOT`
- `SCA_CLUSTERING_METHOD`
- `SCA_K_VALUES`, por ejemplo `2,3,4,5,6,7,8,9,10,11,12`
- `SCA_SELECTION_STRATEGY`, con valores `target_oriented` o `quality`
- `SCA_MERGE_YEAR_TOLERANCE`
- `SCA_MAKE_PRESENTATION_PLOTS`
- `SCA_PRESENTATION_TOP_N_PROFESSORS`

## Configuración principal

Todo lo ajustable vive en `config/settings.py`, incluyendo:

- materias a analizar
- método de clustering principal (`gmm` o `kmeans`)
- rango de `k`
- estrategia de selección (`target_oriented` o `quality`)
- mínimo preferido de clusters para buscar grupos más finos
- `random_state`
- tolerancia de año para merge
- tamaño mínimo por cluster
- mínimo de alumnos por profesor para rankings
- umbral de advertencia por pérdida de datos
- activación de gráficas

## Estrategia de merge

La integración entre materias y exámenes sigue una regla jerárquica y explícita:

1. match exacto por `CLAVEALUMNO` + `anio`
2. si no existe match exacto, fallback por `CLAVEALUMNO` usando el año de examen más cercano dentro de la tolerancia configurada
3. opcionalmente, si un alumno tiene una única fila de examen en toda la fuente, se usa como fallback adicional por `CLAVEALUMNO`

El tipo de match queda guardado en el dataset final:

- `match_type_dmu`
- `match_type_gagb`

También se guardan:

- `matched_exam_year_dmu`
- `matched_exam_year_gagb`
- `matched_year_gap_dmu`
- `matched_year_gap_gagb`

## Salidas

El pipeline genera:

- `output_cluster_analisis/data_clean/merged_dataset.csv`
- `output_cluster_analisis/metrics/cluster_metrics_por_materia.csv`
- `output_cluster_analisis/centroids/centroides_por_materia.csv`
- `output_cluster_analisis/summaries/cluster_objetivo_por_materia.csv`
- `output_cluster_analisis/data_clean/alumnos_cluster_objetivo.csv`
- `output_cluster_analisis/professor_reports/profesores_por_materia.csv`
- `output_cluster_analisis/professor_reports/profesores_cluster_objetivo_detalle.csv`
- `output_cluster_analisis/professor_reports/alumnos_profesores_cluster_objetivo.csv`
- `output_cluster_analisis/professor_reports/ranking_profesores_global.csv`
- `output_cluster_analisis/diagnostics/data_quality_report.csv`
- `output_cluster_analisis/diagnostics/missingness_por_materia.csv`
- `output_cluster_analisis/plots_2d_ica/*.png`
- `output_cluster_analisis/plots_3d_plotly/*.html`
- `output_cluster_analisis/presentation_plots/*.png`
- `output_cluster_analisis/logs/pipeline.log`

Plots explicativos para presentacion:

- `01_resumen_cluster_objetivo_por_materia.png`: tamano, proporcion, score y `k` seleccionado por materia.
- `02_contraste_centroide_objetivo_vs_media.png`: diferencia entre el centroide objetivo y la media de su materia para DMU, GA-GB y calificacion.
- `03_distribuciones_cluster_objetivo_vs_resto.png`: distribuciones del cluster objetivo contra el resto de alumnos.
- `04_top_profesores_por_materia.png`: profesores con mayor concentracion de alumnos en cluster objetivo por materia.
- `05_ranking_global_profesores.png`: ranking global descriptivo.
- `06_distribuciones_alumnos_profesores_destacados.png`: distribuciones de alumnos de profesores destacados; puntos rojos indican alumnos del cluster objetivo.

## Selección de clusters

Para cada materia:

1. se filtran observaciones completas en las tres variables del espacio `R^3`
2. se estandarizan con `StandardScaler`
3. se evalúan candidatos para `k` en el rango configurado; por default se exploran `k=2,...,12`
4. la selección de `k` sigue una regla reproducible configurable

Estrategia default `target_oriented`:

- preferir candidatos válidos con al menos `preferred_min_clusters`
- exigir, cuando exista, que el cluster objetivo valide calificación alta y al menos un porcentaje bajo
- maximizar el score objetivo `z(CALIFICACION)-z(Porcentaje_DMU)-z(Porcentaje_GA_GB)`
- usar `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `BIC` y `AIC` para desempatar
- evitar clusters demasiado pequeños con los umbrales configurados; por default se permite granularidad fina con tamaño mínimo absoluto y fracción mínima de cluster de `0.5%`
- exigir, cuando sea posible, que el cluster objetivo represente al menos `1%` de la materia clusterizada

Estrategia alternativa `quality`:

- maximizar `silhouette_score`
- romper empates con `calinski_harabasz_score`
- luego minimizar `davies_bouldin_score`
- si aplica, minimizar `BIC` y `AIC`
- preferir clusters no demasiado pequeños

Cluster objetivo por materia:

```text
S_c = z(CALIFICACION)_c - z(Porcentaje_DMU)_c - z(Porcentaje_GA_GB)_c
```

Se selecciona:

```text
c* = argmax_c S_c
```

Además se valida que:

- el centroide de `CALIFICACION` esté por encima de la media global de la materia
- al menos uno entre `Porcentaje_DMU` y `Porcentaje_GA_GB` esté por debajo de la media global

## Archivos para la revisión de profesores

Para la etapa de inspección de profesores se generan tres archivos prácticos:

- `data_clean/alumnos_cluster_objetivo.csv`: alumnos exactos que caen en el cluster de alta calificación y porcentajes bajos, con `CLAVEALUMNO`, `CLAVEPROFESOR`, materia, año, sesión, variables y datos del match.
- `professor_reports/profesores_cluster_objetivo_detalle.csv`: profesores que aparecen en ese grupo, con proporción de alumnos en cluster objetivo, distribución de sus alumnos clusterizados y lista de IDs de alumnos del cluster objetivo.
- `professor_reports/alumnos_profesores_cluster_objetivo.csv`: todos los alumnos clusterizados de los profesores que tuvieron al menos un alumno en el cluster objetivo. Este archivo sirve para revisar después las distribuciones de `Porcentaje_DMU`, `Porcentaje_GA_GB` y `CALIFICACION` por profesor.

## Advertencias metodológicas

- El clustering describe patrones; no implica causalidad.
- Que un profesor tenga alta proporción de alumnos en el cluster objetivo no demuestra inflación de calificaciones.
- El ranking global mezcla materias distintas y debe leerse solo como resumen descriptivo.
- El tamaño muestral por profesor importa; el pipeline lo reporta explícitamente.
- Los resultados pueden variar por cohorte (`anio`) y por sesión (`CLAVESESION`).
- `FastICA` se usa solo para visualización 2D, no para construir los clusters.
- Los clusters se ajustan en el espacio 3D estandarizado de las variables originales.

## Pruebas

Para correr tests mínimos:

```bash
python -m pytest
```

Las pruebas incluidas cubren:

- validación de columnas requeridas
- merge básico con fallback por año cercano
- selección del cluster objetivo en un caso sintético
- smoke test del pipeline con inputs sintéticos
