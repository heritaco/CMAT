# Proyecto Visitas

Analisis de visitas al CMAT organizado con una estructura tipo `src`, similar a `proyecto cluster analisis`.

## Como correrlo

```powershell
py -3.11 run_analysis.py
```

El comando regenera los activos descriptivos del reporte y las figuras principales. En esta maquina, `py -3.11` apunta al entorno que ya tiene las dependencias de analisis instaladas; si otro entorno tiene `requirements.txt`, tambien puedes usar `python run_analysis.py`.

Si solo quieres actualizar tablas, figuras descriptivas y fragmentos `.tex`:

```powershell
py -3.11 run_analysis.py --skip-raw-figures
```

## Estructura

- `run_analysis.py`: entrada unica para actualizar resultados.
- `config/`: rutas y configuracion del proyecto.
- `src/visitas_analysis/`: codigo fuente del analisis.
- `output_visitas/`: salidas generadas, tablas, figuras y logs.
- `reportes/`: documentos LaTeX y PDF del reporte.
- `notebooks/`: exploracion historica y desglose por materia.
- `referencias/`: notas y material de apoyo.
- `data/`: marcador local; los datos crudos compartidos siguen viviendo en `../data/`.

## Salidas Principales

- `output_visitas/report_assets/tables/`: tablas CSV descriptivas.
- `output_visitas/report_assets/figures/`: figuras descriptivas usadas por LaTeX.
- `output_visitas/report_assets/tex/`: fragmentos `.tex` incluidos en el reporte.
- `output_visitas/raw_report_figures/`: figuras del flujo compatible con la notebook original.
- `output_visitas/professor_distributions/`: figuras auxiliares por profesor e imputacion.
- `reportes/global/reporte.tex`: reporte principal.
