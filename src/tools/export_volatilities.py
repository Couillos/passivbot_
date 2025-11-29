"""
Script para exportar cálculos de volatilidad (STD y ROC) a CSV.

Este script utiliza el motor Rust para calcular las volatilidades basadas en STD y ROC
para múltiples activos, permitiendo entender cómo funcionan los límites de volatilidad.
"""

import os
import sys
import argparse
import logging
import numpy as np

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import passivbot_rust as pbr
from tools.event_loop_policy import set_windows_event_loop_policy
from config_utils import (
    load_config,
    require_config_value,
    add_arguments_recursively,
    update_config_with_args,
    parse_overrides,
)
from utils import date_to_ts
from backtest import (
    prepare_hlcvs_mss,
    prep_backtest_args,
    create_shared_memory_file,
)

# Configurar event loop policy para Windows
set_windows_event_loop_policy()
import asyncio

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


async def export_volatilities(
    config: dict,
    exchange: str,
    output_csv_path: str,
):
    """
    Exporta las volatilidades calculadas (STD y ROC) a un archivo CSV.

    Args:
        config: Diccionario de configuración
        exchange: Nombre del exchange
        output_csv_path: Ruta del archivo CSV de salida
    """
    logging.info(f"Iniciando exportación de volatilidades para {exchange}...")

    # Preparar datos HLCV
    logging.info("Cargando datos HLCV...")
    coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = (
        await prepare_hlcvs_mss(config, exchange)
    )

    logging.info(
        f"Datos HLCV cargados: {len(coins)} monedas, {hlcvs.shape[0]} timesteps"
    )

    # Preparar parámetros del backtest
    bot_params_list, exchange_params, backtest_params = prep_backtest_args(
        config, mss, exchange
    )

    # Calcular first_timestamp_ms
    try:
        first_ts_ms = (
            int(timestamps[0]) if (timestamps is not None and len(timestamps) > 0) else 0
        )
    except Exception:
        first_ts_ms = 0

    # Si no hay timestamps, intentar obtener desde metadata
    if first_ts_ms == 0:
        meta = mss.get("__meta__", {}) if isinstance(mss, dict) else {}
        candidate_ts = (
            meta.get("requested_start_ts")
            or meta.get("effective_start_ts")
            or require_config_value(config, "backtest.start_date")
        )
        if isinstance(candidate_ts, (int, float)):
            first_ts_ms = int(candidate_ts)
        elif isinstance(candidate_ts, str):
            try:
                first_ts_ms = int(date_to_ts(candidate_ts))
            except Exception:
                first_ts_ms = 0

    backtest_params = dict(backtest_params)
    backtest_params["first_timestamp_ms"] = first_ts_ms

    # Asegurar que el directorio de salida existe
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Directorio creado: {output_dir}")

    # Crear shared memory file para HLCV
    logging.info("Creando archivo de memoria compartida...")
    with create_shared_memory_file(hlcvs) as shared_memory_file:
        # Llamar a la función Rust para exportar volatilidades
        logging.info("Calculando volatilidades y exportando a CSV...")
        pbr.export_volatilities_to_csv(
            shared_memory_file=shared_memory_file,
            hlcvs_shape=hlcvs.shape,
            hlcvs_dtype=hlcvs.dtype.str,
            bot_params=bot_params_list,
            backtest_params_dict=backtest_params,
            output_csv_path=output_csv_path,
        )

    logging.info(f"Volatilidades exportadas exitosamente a: {output_csv_path}")


async def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Exporta cálculos de volatilidad (STD y ROC) a CSV"
    )
    parser.add_argument(
        "config_path",
        type=str,
        help="Ruta al archivo de configuración JSON",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        required=True,
        help="Nombre del exchange (ej: binanceusdm, bybit, etc.)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="volatilities_export.csv",
        help="Ruta del archivo CSV de salida (default: volatilities_export.csv)",
    )
    parser.add_argument(
        "--overrides",
        type=str,
        default="",
        help="Overrides de configuración en formato key=value,key2=value2",
    )

    # Agregar argumentos recursivos desde el template de configuración
    config_template = load_config("configs/template.json")
    add_arguments_recursively(parser, config_template, prefix="")

    args = parser.parse_args()

    # Cargar configuración
    config = load_config(args.config_path)

    # Aplicar overrides desde argumentos
    if args.overrides:
        overrides_dict = parse_overrides(args.overrides)
        update_config_with_args(config, overrides_dict)

    # Aplicar overrides desde argumentos de línea de comandos
    update_config_with_args(config, vars(args))

    # Ejecutar exportación
    await export_volatilities(
        config=config,
        exchange=args.exchange,
        output_csv_path=args.output,
    )

    logging.info("Proceso completado exitosamente.")


if __name__ == "__main__":
    asyncio.run(main())

