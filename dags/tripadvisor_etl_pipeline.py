from __future__ import annotations

import ast
import gc
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from airflow.decorators import dag, task
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype


# Ubicación del archivo de configuración
CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.toml"

# Cargar configuración desde toml (Requiere Python 3.11+)
with open(CONFIG_FILE, "rb") as f:
    CONFIG = tomllib.load(f)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=CONFIG["logging"].get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
)
logger = logging.getLogger(__name__)

# Directorio base
BASE_DIR = Path(__file__).resolve().parents[1]

# Funciones auxiliares

def _get_source_csv() -> Path:
    """Obtiene ruta del CSV de entrada desde config o variable de entorno."""
    csv_name = os.getenv("TRIPADVISOR_SOURCE_CSV", CONFIG["data"]["source_csv"])
    return BASE_DIR / csv_name

def _get_output_dir() -> Path:
    """Obtiene directorio de salida desde config o variable de entorno."""
    output_path = os.getenv("TRIPADVISOR_OUTPUT_DIR", CONFIG["data"]["output_dir"])
    return BASE_DIR / output_path

def parse_string_or_list_column(var_data: pd.Series) -> tuple[pd.Series, bool]:
    """Evalúa si una columna contiene listas en formato string y las explota."""
    if var_data.empty:
        return var_data, False
        
    sample_val = var_data.iloc[0]
    
    if isinstance(sample_val, str) and sample_val.strip().startswith('[') and sample_val.strip().endswith(']'):
        try:
            # Usar un fillna seguro antes de aplicar literal_eval
            var_data_clean = var_data.fillna("[]")
            parsed_lists = var_data_clean.apply(ast.literal_eval)
            return parsed_lists.explode(), True
        except (ValueError, SyntaxError):
            return var_data, False
    return var_data, False

# Definicion del dag
@dag(
    dag_id=CONFIG["pipeline"]["name"],
    description=CONFIG["pipeline"]["description"],
    schedule=None,
    start_date=datetime(2026, 3, 1),
    catchup=False,
    tags=["etl", "data-pipeline", "tripadvisor", "kafka"],
)
def tripadvisor_complete_etl_pipeline():
    """"
    Pipeline:
    1. EXTRACT -> Lectura y validación inicial del CSV
    2. TRANSFORM -> Limpieza y transformaciones avanzadas
    3. EDA -> Análisis y exportación de gráficas
    4. LOAD TO KAFKA -> Envío de datos a Kafka
    """

    @task(
        task_id="extract_validate_data",
        doc="EXTRACCIÓN: Lee CSV y valida estructura inicial.",
    )
    def extract_and_validate() -> dict:
        source = _get_source_csv()
        chunk_size = int(os.getenv("TRIPADVISOR_CHUNK_SIZE", CONFIG["extraction"]["chunk_size"]))
        
        if not source.exists():
            raise FileNotFoundError(f"CSV no encontrado en: {source}")
        
        logger.info(f"Leyendo CSV por lotes desde: {source} (chunksize={chunk_size})")

        rows_input = 0
        columns_input = 0
        column_names: list[str] = []
        missing_counts: pd.Series | None = None
        all_empty_flags: pd.Series | None = None

        for chunk in pd.read_csv(
            source,
            low_memory=CONFIG["extraction"]["csv_low_memory"],
            chunksize=chunk_size,
        ):
            if not column_names:
                column_names = chunk.columns.tolist()
                columns_input = len(column_names)
                missing_counts = pd.Series(0, index=chunk.columns, dtype="int64")
                all_empty_flags = pd.Series(True, index=chunk.columns)

            rows_input += len(chunk)
            missing_counts = missing_counts.add(chunk.isna().sum(), fill_value=0).astype("int64")
            all_empty_flags = all_empty_flags & chunk.isna().all()

        if rows_input == 0:
            raise ValueError("El CSV no contiene filas")
        
        missing_cols = all_empty_flags[all_empty_flags].index.tolist()
        missing_rates = ((missing_counts / rows_input) * 100).sort_values(ascending=False)
        cols_over_threshold = missing_rates[
            missing_rates > CONFIG["transformation"]["cleaning"]["null_threshold"]
        ].index.tolist()
        
        extraction_report = {
            "source_path": str(source),
            "rows_input": int(rows_input),
            "columns_input": int(columns_input),
            "column_names": column_names,
            "columns_completely_empty": missing_cols,
            "columns_with_high_nulls": cols_over_threshold,
            "top_missing_by_column": missing_rates.head(10).to_dict(),
            "chunk_size": chunk_size,
            "timestamp_extract": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"EXTRACT completado: {extraction_report['rows_input']:,} filas validadas")
        return extraction_report

    

    @task(
        task_id="transform_clean_data",
        doc="TRANSFORMACIÓN: Limpieza, normalización e ingeniería de características",
    )
    def transform_and_clean_data(extraction_report: dict) -> dict:
        source_path = extraction_report["source_path"]
        chunk_size = int(extraction_report.get("chunk_size", CONFIG["extraction"]["chunk_size"]))
        columns_to_drop = extraction_report.get("columns_with_high_nulls", [])
        fill_val = CONFIG["transformation"]["cleaning"]["categorical_fill_value"]

        rows_before = 0
        rows_after = 0
        cols_before = int(extraction_report.get("columns_input", 0))

        # Config OHE y Normalización
        ohe_features = {
            "country": CONFIG["transformation"]["encoding"]["country_top_n"],
            "price_level": CONFIG["transformation"]["encoding"]["price_level_top_n"],
            "meals": CONFIG["transformation"]["encoding"]["meals_top_n"],
        }
        drop_first = CONFIG["transformation"]["encoding"]["drop_first"]

        category_counts = {col: pd.Series(dtype="int64") for col in ohe_features}
        scaler_track = {
            col: {"min": None, "max": None}
            for col in CONFIG["transformation"]["normalization"]["numeric_features"]
        }

        # Calcular métricas globales para OHE y Escalado
        for chunk in pd.read_csv(source_path, low_memory=CONFIG["extraction"]["csv_low_memory"], chunksize=chunk_size):
            rows_before += len(chunk)

            object_columns = chunk.select_dtypes(include=["object", "string"]).columns
            for col in object_columns:
                chunk[col] = chunk[col].astype("string").str.strip()
                chunk[col] = chunk[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

            required_cols = [c for c in CONFIG["transformation"]["cleaning"]["required_columns"] if c in chunk.columns]
            if required_cols:
                chunk = chunk.dropna(subset=required_cols)

            rows_after += len(chunk)

            for col in scaler_track:
                if col in chunk.columns:
                    series = chunk[col].dropna()
                    if not series.empty:
                        col_min, col_max = float(series.min()), float(series.max())
                        if scaler_track[col]["min"] is None or col_min < scaler_track[col]["min"]:
                            scaler_track[col]["min"] = col_min
                        if scaler_track[col]["max"] is None or col_max > scaler_track[col]["max"]:
                            scaler_track[col]["max"] = col_max

            for col in ohe_features:
                if col in chunk.columns:
                    values = chunk[col].fillna(fill_val).astype("string").value_counts()
                    category_counts[col] = category_counts[col].add(values, fill_value=0).astype("int64")

        if rows_before == 0:
            raise ValueError("No hay filas para transformar")

        top_categories = {}
        ohe_expected_cols = {}
        ohe_info = {}
        for col, top_n in ohe_features.items():
            if not category_counts[col].empty:
                tops = category_counts[col].sort_values(ascending=False).head(top_n).index.tolist()
                top_categories[col] = tops
                all_groups = tops + ["other"]
                encoded_groups = all_groups[1:] if (drop_first and len(all_groups) > 1) else all_groups
                ohe_expected_cols[col] = [f"{col}_{group}" for group in encoded_groups]
                ohe_info[col] = {"top_categories": tops, "ohe_columns": ohe_expected_cols[col]}

        scaler_info = {}
        for col, track in scaler_track.items():
            if track["min"] is not None and track["max"] is not None:
                scaler_info[col] = {
                    "min": float(track["min"]), "max": float(track["max"]), "range": float(track["max"] - track["min"])
                }

        # Preparar archivo de salida temporal
        output_dir = _get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = output_dir / "_temp_transformed.csv"
        if temp_csv.exists():
            temp_csv.unlink()

        wrote_any_chunk = False
        columns_after = 0

        # Aplicar transformaciones y guardar
        for chunk in pd.read_csv(source_path, low_memory=CONFIG["extraction"]["csv_low_memory"], chunksize=chunk_size):
            object_columns = chunk.select_dtypes(include=["object", "string"]).columns
            for col in object_columns:
                chunk[col] = chunk[col].astype("string").str.strip()
                chunk[col] = chunk[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

            required_cols = [c for c in CONFIG["transformation"]["cleaning"]["required_columns"] if c in chunk.columns]
            if required_cols:
                chunk = chunk.dropna(subset=required_cols)

            if columns_to_drop:
                existing_drop = [col for col in columns_to_drop if col in chunk.columns]
                if existing_drop:
                    chunk = chunk.drop(columns=existing_drop)

            numeric_cols = chunk.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                median_val = chunk[col].median()
                if pd.notna(median_val):
                    chunk[col] = chunk[col].fillna(median_val)

            object_cols_remaining = chunk.select_dtypes(include=["object", "string"]).columns
            for col in object_cols_remaining:
                chunk[col] = chunk[col].fillna(fill_val)

            if "total_reviews_count" in chunk.columns:
                # Prevenir problemas con logaritmos de números negativos
                chunk["log_reviews_count"] = np.log1p(chunk["total_reviews_count"].clip(lower=0))

            for col in CONFIG["transformation"]["normalization"]["numeric_features"]:
                if col in chunk.columns and col in scaler_info:
                    col_min, col_range = scaler_info[col]["min"], scaler_info[col]["range"]
                    chunk[f"{col}_normalized"] = (chunk[col] - col_min) / col_range if col_range > 0 else 0.0

            for col, tops in top_categories.items():
                if col in chunk.columns:
                    grouped_col = f"{col}_grouped"
                    chunk[grouped_col] = chunk[col].astype("string").apply(lambda x: x if x in tops else "other")
                    dummies = pd.get_dummies(chunk[grouped_col], prefix=col, dtype=int)
                    dummies = dummies.reindex(columns=ohe_expected_cols[col], fill_value=0)
                    chunk = pd.concat([chunk, dummies], axis=1)

            chunk.to_csv(temp_csv, mode="a", header=not wrote_any_chunk, index=False)
            wrote_any_chunk = True
            columns_after = len(chunk.columns)

        if not wrote_any_chunk:
            raise ValueError("No se generaron filas transformadas")

        transform_report = {
            "source_path": str(temp_csv),
            "temp_csv": str(temp_csv),
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "columns_before": cols_before,
            "columns_after": columns_after,
            "columns_dropped": columns_to_drop,
            "scaler_info": scaler_info,
            "ohe_info": ohe_info,
            "chunk_size": chunk_size,
            "timestamp_transform": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"TRANSFORM completado: {rows_after:,} filas, {columns_after} columnas")
        return transform_report

    @task(
        task_id="eda",
        doc="EDA: Análisis Exploratorio de Datos con Muestreo de RAM amigable",
    )
    def eda(transform_report: dict) -> dict:
        source_path = transform_report.get("temp_csv", transform_report["source_path"])
        chunk_size = int(transform_report.get("chunk_size", 100000))
        total_rows = transform_report.get("rows_after", 100000)
        
        #En lugar de cargar todo, leemos una muestra 
        max_rows_for_eda = 50000
        sample_fraction = min(1.0, max_rows_for_eda / total_rows) if total_rows > 0 else 1.0
        
        logger.info(f"Leyendo muestra para EDA (Aprox {sample_fraction:.1%}) desde: {source_path}")
        
        sampled_chunks = []
        for chunk in pd.read_csv(source_path, chunksize=chunk_size):
            # Muestrear aleatoriamente una fracción del chunk actual
            sampled_chunks.append(chunk.sample(frac=sample_fraction, random_state=42))
            
        dataset = pd.concat(sampled_chunks, ignore_index=True)
        del sampled_chunks # Liberar memoria intermitente
        gc.collect() 
        
        output_dir = str(_get_output_dir() / "eda_output_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        float_columns, integer_columns, string_columns, generated_images = [], [], [], []

        for var_name in dataset.columns:
            var_data = dataset[var_name].dropna()
            if var_data.empty: continue
            
            if is_float_dtype(var_data):
                float_columns.append(var_name)
                fig, ax = plt.subplots(figsize=(8, 6))
                var_data.plot.hist(density=True, bins=30, alpha=0.6, color='skyblue', edgecolor='black', ax=ax)
                var_data.plot.kde(color='red', linewidth=2, ax=ax)
                
                ax.set_title(f'Distribution and Density: {var_name}')
                ax.set_xlabel(var_name)
                ax.set_ylabel('Density')
                ax.grid(axis='y', alpha=0.75)
                
                file_path = os.path.join(output_dir, f"hist_{var_name}.png")
                fig.savefig(file_path, bbox_inches='tight')
                plt.close(fig) # Liberar memoria de la figura de forma segura
                generated_images.append(file_path)
                
            elif is_integer_dtype(var_data):
                integer_columns.append(var_name)
                fig, ax = plt.subplots(figsize=(8, 6))
                
                value_counts = var_data.value_counts().sort_index()
                value_counts.plot.bar(color='mediumseagreen', edgecolor='black', ax=ax)
                
                ax.set_title(f'Category Frequencies: {var_name}')
                ax.set_xlabel('Categories')
                ax.set_ylabel('Count')
                
                file_path = os.path.join(output_dir, f"bar_{var_name}.png")
                fig.savefig(file_path, bbox_inches='tight')
                plt.close(fig)
                generated_images.append(file_path)
                
            elif is_string_dtype(var_data):
                string_columns.append(var_name)
                words_series, is_list_col = parse_string_or_list_column(var_data)
                
                words_series = words_series.astype(str).str.strip()
                top_words = words_series.value_counts().head(10)
                
                if top_words.empty:
                    continue

                fig, ax = plt.subplots(figsize=(10, 8))
                top_words.sort_values().plot.barh(color='coral', edgecolor='black', ax=ax)
                
                title_type = "List Items" if is_list_col else "Words/Categories"
                ax.set_title(f'Top Frequencies (Max 10) {title_type}: {var_name}')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Item')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                
                file_path = os.path.join(output_dir, f"text_bar_{var_name}.png")
                fig.savefig(file_path, bbox_inches='tight')
                plt.close(fig)
                generated_images.append(file_path)

        if float_columns:
            logger.info(f"Generando gráfico de violín para: {float_columns}")
            fig, ax = plt.subplots(figsize=(10, 6))
            data_to_plot = [dataset[col].values for col in float_columns]
            
            ax.violinplot(data_to_plot, showmeans=True, showmedians=True)
            ax.set_xticks(range(1, len(float_columns) + 1))
            ax.set_xticklabels(float_columns, rotation=45)
            ax.set_title('Combined Violin Plot for Continuous Variables')
            ax.set_ylabel('Value')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            file_path = os.path.join(output_dir, "violin_continuous_vars.png")
            fig.savefig(file_path, bbox_inches='tight')
            plt.close(fig)
            generated_images.append(file_path)
            
        plt.close('all') 
        del dataset
        gc.collect()

        eda_report = {
            "source_path": source_path,
            "plots_dir": output_dir,
            "continuous_vars_processed": float_columns,
            "categorical_vars_processed": integer_columns,
            "string_vars_processed": string_columns,
            "total_plots_generated": len(generated_images),
            "timestamp_eda": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.info(f"EDA completado ")
        return eda_report
    
    # Orquestación de dependencias
    extraction = extract_and_validate()
    transformation = transform_and_clean_data(extraction)
    eda_result = eda(transformation)

tripadvisor_etl_dag = tripadvisor_complete_etl_pipeline()

logger.info("DAG Tripadvisor ETL Pipeline cargado exitosamente")