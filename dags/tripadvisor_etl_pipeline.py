
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tomllib
from sklearn.preprocessing import OneHotEncoder
from airflow.decorators import dag, task
import ast
import matplotlib.pyplot as plt
from pandas.api.types import is_float_dtype, is_integer_dtype, is_string_dtype

# CONFIGURACIÓN GLOBAL Y CARGAS

# Ubicación del archivo de configuración
CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.toml"

# Cargar configuración desde toml
with open(CONFIG_FILE, "rb") as f:
    CONFIG = tomllib.load(f)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format=CONFIG["logging"]["format"],
)
logger = logging.getLogger(__name__)

# Directorio base
BASE_DIR = Path(__file__).resolve().parents[1]

# Funciones auxiliares para rutas
def _get_source_csv() -> Path:
    """Obtiene ruta del CSV de entrada desde config o variable de entorno."""
    csv_name = os.getenv("TRIPADVISOR_SOURCE_CSV", CONFIG["data"]["source_csv"])
    return BASE_DIR / csv_name

def _get_output_dir() -> Path:
    """Obtiene directorio de salida desde config o variable de entorno."""
    output_path = os.getenv("TRIPADVISOR_OUTPUT_DIR", CONFIG["data"]["output_dir"])
    return BASE_DIR / output_path

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
    1. EXTRACT-> Lectura y validación inicial del CSV
    2. TRANSFORM-> Limpieza y transformaciones avanzadas
    3. EDA-> EDA y exportación
    4. LOAD TO KAFKA -> Envío de datos a Kafka
    
    Habia pensado esto: 
    CSV → Validación → Limpieza → Transformación → EDA → CSV+Kafka
    """

    @task(
        task_id="extract_validate_data",
        doc="EXTRACCIÓN: Lee CSV y valida estructura inicial.",
    )
    def extract_and_validate() -> dict:
        """
        EXTRACT -> leer el dataset y validar que el archivo existe, no está vacío y detectar problemas iniciales
        - leer el csv y certificar que existe y se cargo bien
        - validar que no esta vacio
        - ver los nulos por columna y eliminar aquellas que esten casi completamente vacias (>70%)
        - generar un report con metricas basicas para ver como son los datos antes de limpiarlos
        - recive el csv y devuelve un diccionario con estas metricas ademas del csv para que el siguiente task lo pueda usar
        """
        source = _get_source_csv()
        
        # Validar existencia del archivo
        if not source.exists():
            raise FileNotFoundError(f"CSV no encontrado en: {source}")
        
        logger.info(f"Leyendo CSV desde: {source}")
        
        dataset = pd.read_csv(source, low_memory=CONFIG["extraction"]["csv_low_memory"])
        
        if dataset.empty:
            raise ValueError("El CSV no contiene filas")
        
        logger.info(f" Dimensiones: {dataset.shape[0]:,} filas × {dataset.shape[1]} columnas")
        
        # Análisis de calidad
        missing_cols = dataset.columns[dataset.isna().all()].tolist()
        missing_rates = (dataset.isna().mean() * 100).sort_values(ascending=False)
        cols_over_threshold = missing_rates[
            missing_rates > CONFIG["transformation"]["cleaning"]["null_threshold"]
        ].index.tolist()
        
        logger.info(f" Columnas completamente vacías: {len(missing_cols)}")
        logger.info(f" Columnas con >{CONFIG['transformation']['cleaning']['null_threshold']}% nulos: {len(cols_over_threshold)}")
        
        extraction_report = {
            "source_path": str(source),
            "rows_input": int(dataset.shape[0]),
            "columns_input": int(dataset.shape[1]),
            "column_names": dataset.columns.tolist(),
            "columns_completely_empty": missing_cols,
            "columns_with_high_nulls": cols_over_threshold,
            "top_missing_by_column": missing_rates.head(10).to_dict(),
            "timestamp_extract": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"EXTRACT completado: {extraction_report['rows_input']:,} filas validadas")
        return extraction_report

    @task(
        task_id="transform_clean_data",
        doc="TRANSFORMACIÓN: Limpieza, normalización e ingenieria de caracteristicas",
    )
    def transform_and_clean_data(extraction_report: dict) -> dict:
        """
        TRANSFORM -> limpieza de datos, transformaciones numericas, feature engineering
        - normalizar valores falttantes
        - eliminar filas sin campos críticos
        - imputar nulos 
        - eliminar columnas con muchos nulos
        - transformar logarítmicamente el conteo de reseñas
        - normalizar numéricas con min-max
        - one-hot encoding para categóricas
        - procesar campos de listas (cuisines, tags, features) para crear variables binarias
          
        Entrada:
          - extraction_report 
          - CSV de entrada
          
        Salida:
          - dict con métricas de transformación (rows antes/después, columnas, etc)
        """

        source_path = extraction_report["source_path"]
        logger.info(f" Leyendo datos desde: {source_path}")
        
        dataset = pd.read_csv(source_path)
        rows_before = len(dataset)
        cols_before = len(dataset.columns)
        
        logger.info(f" Cargado: {rows_before:,} filas")
        
        # Limpieza de espacios en blanco
        logger.info(" Limpiando espacios en blanco")
        object_columns = dataset.select_dtypes(include=["object", "string"]).columns
        for col in object_columns:
            dataset[col] = dataset[col].astype("string").str.strip()
        
        # Normalizar variantes de vacío a pd.NA
        logger.info("   → Normalizando valores vacíos a NA")
        for col in object_columns:
            dataset[col] = dataset[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        
        # Eliminar filas sin campos críticos
        required_cols = [
            c for c in CONFIG["transformation"]["cleaning"]["required_columns"]
            if c in dataset.columns
        ]
        if required_cols:
            logger.info(f" Eliminando filas incompletas (campos: {required_cols})")
            dataset = dataset.dropna(subset=required_cols)
        
        # Imputación de nulos
        logger.info(" Imputando valores faltantes")
        numeric_cols = dataset.select_dtypes(include=["number"]).columns
        for col in numeric_cols:
            median_val = dataset[col].median()
            if pd.notna(median_val):
                dataset[col] = dataset[col].fillna(median_val)
        
        object_cols_remaining = dataset.select_dtypes(include=["object", "string"]).columns
        for col in object_cols_remaining:
            fill_val = CONFIG["transformation"]["cleaning"]["categorical_fill_value"]
            dataset[col] = dataset[col].fillna(fill_val)
        
        # Eliminar columnas con >70% nulos
        logger.info(" Eliminando columnas con muchos nulos...")
        missing_rates = (dataset.isna().mean() * 100)
        threshold = CONFIG["transformation"]["cleaning"]["null_threshold"]
        columns_to_drop = missing_rates[missing_rates > threshold].index.tolist()
        if columns_to_drop:
            logger.info(f" Eliminadas {len(columns_to_drop)} columnas")
            dataset = dataset.drop(columns=columns_to_drop)
        
        # Transformación logarítmica
        logger.info(" Aplicando transformación logarítmica al conteo de las reseñas")
        if "total_reviews_count" in dataset.columns:
            dataset["log_reviews_count"] = np.log1p(dataset["total_reviews_count"])
        
        # Normalización min-max
        logger.info(" Normalizando variables numéricas")
        scaler_info = {}
        for col in CONFIG["transformation"]["normalization"]["numeric_features"]:
            if col in dataset.columns:
                col_min = dataset[col].min()
                col_max = dataset[col].max()
                col_range = col_max - col_min
                
                if col_range > 0:
                    dataset[f"{col}_normalized"] = (dataset[col] - col_min) / col_range
                else:
                    dataset[f"{col}_normalized"] = 0.0
                
                scaler_info[col] = {
                    "min": float(col_min),
                    "max": float(col_max),
                    "range": float(col_range),
                }
        
        # One-Hot Encoding
        logger.info("   → Aplicando One-Hot Encoding")
        ohe_features = {
            "country": CONFIG["transformation"]["encoding"]["country_top_n"],
            "price_level": CONFIG["transformation"]["encoding"]["price_level_top_n"],
            "meals": CONFIG["transformation"]["encoding"]["meals_top_n"],
        }
        
        ohe_info = {}
        for col, top_n in ohe_features.items():
            if col in dataset.columns:
                # Seleccionar top N y agrupar el resto
                top_cats = dataset[col].value_counts().head(top_n).index.tolist()
                dataset[f"{col}_grouped"] = dataset[col].apply(
                    lambda x: x if x in top_cats else "other"
                )
                
                encoder = OneHotEncoder(
                    categories="auto",
                    drop="first" if CONFIG["transformation"]["encoding"]["drop_first"] else None,
                    sparse_output=False,
                    dtype=int,
                )
                
                col_reshaped = dataset[f"{col}_grouped"].values.reshape(-1, 1)
                encoded_array = encoder.fit_transform(col_reshaped)
                feature_names = encoder.get_feature_names_out([col])
                
                ohe_cols = pd.DataFrame(encoded_array, columns=feature_names, index=dataset.index)
                dataset = pd.concat([dataset, ohe_cols], axis=1)
                
                ohe_info[col] = {
                    "top_categories": top_cats,
                    "ohe_columns": ohe_cols.columns.tolist(),
                }
        
        ### TENGO QUE REVISAR ESTO
        # Procesamiento de listas 
        logger.info(" Procesando listas")
        listas = {}
        for field in ["cuisines", "top_tags", "features"]:
            if field in dataset.columns:
                try:
                    all_values = []
                    for val in dataset[field].dropna():
                        if isinstance(val, str):
                            if val.startswith("["):
                                try:
                                    parsed = eval(val)
                                    if isinstance(parsed, list):
                                        all_values.extend(parsed)
                                except:
                                    all_values.append(val)
                            else:
                                all_values.append(val)
                    
                    top_values = pd.Series(all_values).value_counts().head(5).index.tolist()
                    for top_val in top_values:
                        dataset[f"{field}_has_{top_val}"] = (
                            dataset[field].fillna("").apply(lambda x: 1 if top_val in str(x) else 0)
                        )
                    
                    listas[field] = {
                        "top_values": top_values,
                    }
                except Exception as e:
                    logger.warning(f"Error procesando {field}: {e}")
        
        # Guardar dataset transformado temporalmente
        output_dir = _get_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_csv = output_dir / "_temp_transformed.csv"
        dataset.to_csv(temp_csv, index=False)
        logger.info(f"Guardado en: {temp_csv}")
        
        rows_after = len(dataset)
        
        transform_report = {
            "source_path": str(output_dir),
            "temp_csv": str(temp_csv),
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "columns_before": cols_before,
            "columns_after": len(dataset.columns),
            "columns_dropped": columns_to_drop,
            "scaler_info": scaler_info,
            "ohe_info": ohe_info,
            "listas": listas,
            "timestamp_transform": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"TRANSFORM completado: {rows_after:,} filas, {len(dataset.columns)} columnas")
        return transform_report

        #--------------------------

        # guardo el dataset transformado en temp_csv para luego pasarlo al siguiente task despues se hace el eda y exportamos a un csv final, no se si es lo mejor
        # ahora queda generar el eda (graficas y tal) y detectar outliers -> un task aparte
        # despues ya es cargar a kafka con otro task -> quiza transformar a json y enviarlo por batches, habra que monitorizar de alguna manera
        # quedaria algo asi:
        # E-T-L: extract_validate_data() -> transform_clean_data() -> eda() -> load_to_kafka()
        # no se si faltaria hacer algo mas

        # --------------------------


    logger = logging.getLogger(__name__)

    def parse_string_or_list_column(var_data: pd.Series) -> tuple[pd.Series, bool]:
        if var_data.empty:
            return var_data, False
            
        sample_val = var_data.iloc[0]
        
        if isinstance(sample_val, str) and sample_val.strip().startswith('[') and sample_val.strip().endswith(']'):
            try:
                parsed_lists = var_data.apply(ast.literal_eval)
                return parsed_lists.explode(), True
            except (ValueError, SyntaxError):
                return var_data, False
        return var_data, False

    @task(
        task_id="eda",
        doc="EDA: Análisis Exploratorio de Datos con generación de gráficos",
    )
    def eda(transform_report: dict) -> dict:
        source_path = transform_report["source_path"]
        logger.info(f"Leyendo datos desde: {source_path}")
        
        dataset = pd.read_csv(source_path)
        
        output_dir = "eda_output_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        float_columns = []
        integer_columns = []
        string_columns = []
        generated_images = []

        for var_name in dataset.columns:
            var_data = dataset[var_name]
            
            if is_float_dtype(var_data):
                float_columns.append(var_name)
                
                plt.figure(figsize=(8, 6))
                var_data.plot.hist(density=True, bins=30, alpha=0.6, color='skyblue', edgecolor='black')
                var_data.plot.kde(color='red', linewidth=2)
                
                plt.title(f'Distribution and Density: {var_name}')
                plt.xlabel(var_name)
                plt.ylabel('Density')
                plt.grid(axis='y', alpha=0.75)
                
                file_path = os.path.join(output_dir, f"hist_{var_name}.png")
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                generated_images.append(file_path)
                
            elif is_integer_dtype(var_data):
                integer_columns.append(var_name)
                
                plt.figure(figsize=(8, 6))
                
                value_counts = var_data.value_counts().sort_index()
                value_counts.plot.bar(color='mediumseagreen', edgecolor='black')
                
                plt.title(f'Category Frequencies: {var_name}')
                plt.xlabel('Categories')
                plt.ylabel('Count')
                
                file_path = os.path.join(output_dir, f"bar_{var_name}.png")
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                generated_images.append(file_path)
                
            elif is_string_dtype(var_data):
                string_columns.append(var_name)
                
                words_series, is_list_col = parse_string_or_list_column(var_data)
                
                words_series = words_series.astype(str).str.strip()
                top_words = words_series.value_counts().head(10)
                
                if top_words.empty:
                    continue

                plt.figure(figsize=(10, 8))
                
                top_words.sort_values().plot.barh(color='coral', edgecolor='black')
                
                title_type = "List Items" if is_list_col else "Words/Categories"
                plt.title(f'Top Frequencies (Max 10) {title_type}: {var_name}')
                plt.xlabel('Frequency')
                plt.ylabel('Item')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                
                file_path = os.path.join(output_dir, f"text_bar_{var_name}.png")
                plt.savefig(file_path, bbox_inches='tight')
                plt.close()
                generated_images.append(file_path)

        if float_columns:
            logger.info(f"Generando gráfico de violín para: {float_columns}")
            plt.figure(figsize=(10, 6))
            
            data_to_plot = [dataset[col].values for col in float_columns]
            
            plt.violinplot(data_to_plot, showmeans=True, showmedians=True)
            
            plt.xticks(ticks=range(1, len(float_columns) + 1), labels=float_columns, rotation=45)
            plt.title('Combined Violin Plot for Normalized Continuous Variables')
            plt.ylabel('Value')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            file_path = os.path.join(output_dir, "violin_continuous_vars.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            generated_images.append(file_path)

        eda_report = {
            "source_path": source_path,
            "plots_dir": output_dir,
            "continuous_vars_processed": float_columns,
            "categorical_vars_processed": integer_columns,
            "string_vars_processed": string_columns,
            "total_plots_generated": len(generated_images),
            "generated_images_list": generated_images,
            "timestamp_eda": datetime.utcnow().isoformat(),
        }
        
        logger.info(f"EDA completado: {len(generated_images)} gráficos generados. Continuas: {len(float_columns)}, Categóricas: {len(integer_columns)}, Strings: {len(string_columns)}")
        
        return eda_report
    
    extraction = extract_and_validate()
    transformation = transform_and_clean_data(extraction)
    eda = eda(transformation)


tripadvisor_etl_dag = tripadvisor_complete_etl_pipeline()

logger.info("DAG Tripadvisor ETL Pipeline cargado exitosamente")
logger.info(f"Configuración desde: {CONFIG_FILE}")
