# CPIA-recommender

Recomendador de películas **Two-Tower** entrenado con **MovieLens 32M**, metadatos de **TMDB** (Kaggle + API) y perfiles de **Letterboxd**.
Una torre codifica al usuario a partir de su historial ponderado y la otra a la película (género, director, reparto, año, duración y
embedding MiniLM de la sinopsis). Se entrena de forma contrastiva con corrección **logQ** del sesgo de muestreo, y la inferencia
usa búsqueda exacta más re-ranking **MMR** (diversidad).

Proyecto de CPIA (UPC, Ingeniería Física) de Arnau Montull y Uriel Martínez.
## Instalación
```
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Datos necesarios

Los datos crudos **no están en el repo** (`data/` está en `.gitignore`):

- `data/ml-32m/` : MovieLens 32M (`ratings.csv`, `movies.csv`, `links.csv`, `tags.csv`).
- `data/the_movies_dataset_kaggle/` : *The Movies Dataset* de Kaggle (`movies_metadata.csv`, `credits.csv`, `keywords.csv`, ...).

## Reproducir la pipeline

Desde la raíz del repo. Cada etapa guarda su salida en `data/processed/` y se salta si ya existe (salvo `--forzar`).
| # | Comando |
|---|---|
| 1 | `python -m src.preprocesamiento.validar` |
| 2 | `python -m src.preprocesamiento.descargar_tmdb` (opcional; necesita token de TMDB, ver su docstring) |
| 3 | `python -m src.preprocesamiento.submuestra` (solo `dev`: crea el 10 % por usuarios) |
| 4 | `python -m src.preprocesamiento.split --muestra dev\|completo` |
| 5 | `python -m src.preprocesamiento.senal --muestra dev\|completo` |
| 6 | `python -m src.preprocesamiento.historial --muestra dev\|completo` |
| 7 | `python -m src.preprocesamiento.features_peliculas --muestra dev\|completo` |
| 8 | `python -m src.preprocesamiento.frio --muestra dev\|completo` |
| 9 | `python -m src.letterboxd --export <zip\|carpeta> --nombre <alias> [--sin-api]` |
- Si haces el paso 2, repite el 1 para incorporar lo descargado.
- El paso 9 usa la API de TMDB para emparejar títulos que no coinciden de forma exacta; con `--sin-api` solo empareja por título y año.

## Entrenar
```
python -m src.entrenar --muestra dev --device cpu --salida modelos/
python -m src.entrenar --muestra completo --device cuda --salida <Drive>/modelos   # p. ej. Colab con GPU
```
Parámetros en `config.yaml` (`modelo:`, `entrenamiento:`); `--tau`, `--batch`, `--epocas`, `--semilla` los sobrescriben.
Reanuda solo si existe `ultimo.pt` en la carpeta de la ejecución.

## Tests
```
python -m pytest tests
```
Usan datos sintéticos calculados a mano. Los tests de invariantes (`test_invariantes_*.py`) comprueban las salidas reales y
**se saltan si no existe `data/processed/dev/`** (los de `completo`, si no existen sus salidas).

## Privacidad

Los exports de Letterboxd son datos personales: viven en `data/external/letterboxd/` (fuera de git). Nunca se suben tokens.
