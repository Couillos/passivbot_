"""
Script para dividir un CSV grande de volatilidades en archivos separados por asset.

Este script procesa el CSV generado por export_volatilities.py y crea un archivo CSV
separado para cada asset (coin), facilitando el análisis individual de cada activo.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def split_csv_by_coin(
    input_csv_path: str,
    output_dir: str,
    chunk_size: int = 100000,
):
    """
    Divide un CSV grande de volatilidades en archivos separados por coin.

    Args:
        input_csv_path: Ruta al CSV grande de entrada
        output_dir: Directorio donde se guardarán los CSVs por coin
        chunk_size: Tamaño del chunk para procesar el archivo (default: 100000 filas)
    """
    logging.info(f"Iniciando división del CSV: {input_csv_path}")
    logging.info(f"Tamaño del archivo: {os.path.getsize(input_csv_path) / (1024**3):.2f} GB")

    # Crear directorio de salida si no existe
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directorio de salida: {output_path}")

    # Diccionario para almacenar los archivos CSV abiertos por coin
    csv_files = {}
    header_written = {}

    # Contador de filas procesadas
    total_rows = 0
    rows_per_coin = defaultdict(int)

    try:
        # Leer el CSV en chunks para manejar archivos grandes
        logging.info(f"Procesando CSV en chunks de {chunk_size} filas...")
        
        for chunk_idx, chunk_df in enumerate(pd.read_csv(input_csv_path, chunksize=chunk_size)):
            # Verificar que la columna 'coin' existe
            if 'coin' not in chunk_df.columns:
                raise ValueError(
                    "La columna 'coin' no se encuentra en el CSV. "
                    f"Columnas disponibles: {list(chunk_df.columns)}"
                )

            # Procesar cada chunk
            for coin in chunk_df['coin'].unique():
                coin_data = chunk_df[chunk_df['coin'] == coin]

                # Crear archivo CSV para este coin si no existe
                if coin not in csv_files:
                    coin_filename = f"volatilities_{coin}.csv"
                    coin_filepath = output_path / coin_filename
                    csv_file = open(coin_filepath, 'w', newline='')
                    # Escribir header y primeros datos
                    coin_data.to_csv(
                        csv_file,
                        index=False,
                        header=True,
                        lineterminator='\n'
                    )
                    csv_files[coin] = csv_file
                    header_written[coin] = True
                    logging.info(f"Archivo creado para {coin}: {coin_filename}")
                else:
                    # Escribir datos al archivo existente (sin header)
                    coin_data.to_csv(
                        csv_files[coin],
                        index=False,
                        header=False,
                        lineterminator='\n'
                    )

                rows_per_coin[coin] += len(coin_data)

            total_rows += len(chunk_df)
            
            if (chunk_idx + 1) % 10 == 0:
                logging.info(
                    f"Procesados {chunk_idx + 1} chunks, "
                    f"{total_rows:,} filas totales, "
                    f"{len(csv_files)} assets únicos"
                )

        # Cerrar todos los archivos
        for coin, csv_file in csv_files.items():
            csv_file.close()
            file_size_mb = os.path.getsize(output_path / f"volatilities_{coin}.csv") / (1024**2)
            logging.info(
                f"Archivo completado para {coin}: "
                f"{rows_per_coin[coin]:,} filas, "
                f"{file_size_mb:.2f} MB"
            )

        logging.info(f"\n{'='*60}")
        logging.info(f"Proceso completado exitosamente:")
        logging.info(f"  - Total de filas procesadas: {total_rows:,}")
        logging.info(f"  - Total de assets: {len(csv_files)}")
        logging.info(f"  - Archivos generados en: {output_path}")
        logging.info(f"{'='*60}")

    except Exception as e:
        # Cerrar archivos en caso de error
        for csv_file in csv_files.values():
            try:
                csv_file.close()
            except:
                pass
        raise


def main():
    """Función principal del script."""
    parser = argparse.ArgumentParser(
        description="Divide un CSV grande de volatilidades en archivos separados por asset"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Ruta al archivo CSV grande de volatilidades",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="volatilities_by_coin",
        help="Directorio donde se guardarán los CSVs por coin (default: volatilities_by_coin)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Tamaño del chunk para procesar el archivo (default: 100000 filas)",
    )

    args = parser.parse_args()

    # Validar que el archivo de entrada existe
    if not os.path.exists(args.input_csv):
        logging.error(f"El archivo de entrada no existe: {args.input_csv}")
        sys.exit(1)

    # Ejecutar división
    split_csv_by_coin(
        input_csv_path=args.input_csv,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

