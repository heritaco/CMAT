# Proyecto analisis calificaciones profesores

Dashboard interactivo en Streamlit y Plotly para explorar, de 2019 a 2025, si ciertos profesores tienen estudiantes con calificacion alta en una materia, bajo desempeno en examenes institucionales GA/GB y DMU, y pocas asistencias a asesorias.

El analisis prioriza trazabilidad: los rangos de "bajo", "alto" y "pocas asistencias" son controles interactivos, no supuestos fijos.

## Archivos usados

El proyecto es autocontenido: los datos viven en `proyecto analisis calificaciones profesores/data/`.

Archivos esperados:

- `GA_GB`: contiene `ID`, `Año`, `Total GA-120`, `Total GB-160`.
- `DMU`: contiene `ID`, `Año`, `Total DMU-150`.
- `Calificaciones`: contiene hojas por año, de 2019 a 2025, con `ID`, `Clave materia`, `ID_Profesor`, `Calificación de materia`.
- `Asesorias_2019-2024`: contiene registros de asesorias con `fecha`, `id`, `periodo` y otros campos.
- `Asesorias_2025`: contiene registros de asesorias de 2025.
- `ID_profesores`: contiene `ID_Profesor`, `Nombre de profesor`.

El cargador acepta formatos tabulares comunes como `.csv`, `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.tsv` y `.parquet`. Para `Calificaciones.xlsx`, lee todas las hojas y usa el nombre de la hoja como `Año`.

## Variables usadas

- `ID`: identificador normalizado del alumno, siempre como texto.
- `Año`: año del registro, derivado de la hoja de calificaciones, de la columna `Año` o de `fecha`.
- `Clave materia`: materia filtrable, con foco inicial en `MAT1012` y `MAT1022`.
- `ID_Profesor`: identificador del profesor, siempre como texto.
- `Nombre de profesor`: nombre legible del profesor.
- `Calificación de materia`: calificacion obtenida en la materia.
- `Total GA-120`, `Total GB-160`, `Total DMU-150`: puntajes institucionales.
- `asesorias_count`: numero de registros de asesorias del alumno en el mismo año. Si el alumno no aparece en asesorias ese año, el valor es `0`.
- `Calificación de materia original`: valor original antes de convertir a numero, util para casos como `BA`.
- `calificacion_no_numerica`: indica si la calificacion original no se pudo convertir a numero.

## Crear entorno Conda

```bash
conda env create -f environment.yml
conda activate cmat
```

Si el entorno `cmat` ya existe:

```bash
conda activate cmat
```

## Alternativa con pip

```bash
pip install -r requirements.txt
```

## Correr el dashboard

Desde esta carpeta:

```bash
streamlit run app.py
```

Flujo completo esperado:

```bash
cd "proyecto analisis calificaciones profesores"
conda activate cmat
streamlit run app.py
```

## Filtros

- `Selector de Clave materia`: permite elegir `MAT1012`, `MAT1022`, ambas, todas o una seleccion personalizada de materias disponibles.
- `Tipo de examen`: `GA`, `GB` o `ambos/autodetectar`. En autodeteccion, si un alumno tiene GA valido se evalua GA, si tiene GB valido se evalua GB, y si tiene ambos puede cumplir por cualquiera de los dos.
- `Rango Total GA-120`: define el intervalo aceptado para GA.
- `Rango Total GB-160`: define el intervalo aceptado para GB.
- `Rango Total DMU-150`: define el intervalo aceptado para DMU.
- `Rango Calificación de materia`: define el intervalo aceptado para la calificacion de materia.
- `Rango numero de asistencias a asesorias`: define el intervalo aceptado de asistencias; incluye `0` para alumnos sin registros de asesorias en ese año.
- `Permitir que alumnos con GA/GB/DMU faltante cumplan la condicion`: si esta apagado, los faltantes excluyen al registro del cumplimiento; si esta encendido, un faltante no impide que cumpla.
- `Permitir calificaciones no numericas en la condicion`: si esta encendido, valores como `BA` pueden cumplir la parte de calificacion aunque no tengan numero.
- `Mostrar filas con -1 en tablas`: controla si las tablas muestran registros con GA/GB/DMU faltante.
- `Mostrar calificaciones no numericas en tablas`: controla si las tablas muestran registros cuyo valor original no se pudo convertir a numero.
- `Filtro por profesor`: limita el analisis a profesores seleccionados.
- `Filtro por año`: limita el analisis a uno o varios años.

## Tablas

- `Resumen por profesor`: una fila por profesor con total de alumnos, alumnos que cumplen, porcentaje que cumple, promedios, medianas e IDs de alumnos que cumplen.
- `Alumnos que cumplen las condiciones`: registros individuales que satisfacen todos los rangos activos. Incluye el `Año` en que el estudiante curso con ese profesor.
- `Alumnos que NO cumplen las condiciones`: registros filtrados que no satisfacen simultaneamente las condiciones activas.

## Valores -1

Para visualizacion, los faltantes en `Total GA-120`, `Total GB-160` y `Total DMU-150` se muestran como `-1`. Esto permite distinguir claramente registros sin puntaje. El toggle de faltantes controla si esos registros pueden cumplir la condicion, y otro toggle controla si aparecen en las tablas.

## Validaciones incluidas

El dashboard reporta advertencias sobre:

- columnas faltantes;
- IDs vacios o duplicados por columnas repetidas;
- valores no numericos convertidos a faltantes;
- materias `MAT1012` o `MAT1022` no encontradas;
- profesores sin nombre;
- alumnos en calificaciones que no aparecen en asesorias;
- alumnos en asesorias que no aparecen en calificaciones;
- examenes GA/GB o DMU faltantes.

## Limitaciones

- Correlacion no implica causalidad.
- Los rangos son arbitrarios y ajustables.
- Los faltantes pueden sesgar resultados, especialmente si no son aleatorios.
- Alumnos con mas de un profesor o mas de una materia deben interpretarse con cuidado, porque el analisis trabaja a nivel registro alumno-materia-profesor-año.
