"""
Script para ejecutar backtest con configuraciones de Pareto.

Este script permite:
- Encontrar automáticamente el mejor resultado de Pareto (más cercano al ideal)
- Ejecutar backtest con un archivo de Pareto específico por índice
- Ejecutar backtest con un archivo de Pareto específico por ruta
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Agregar el directorio raíz al path para importar módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pareto_store import compute_ideal, comma_separated_values_float
import glob
import json
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def find_best_pareto(pareto_dir: str, mode: str = "weighted", weights: tuple = None):
    """
    Encuentra el mejor archivo de Pareto (más cercano al ideal).

    Args:
        pareto_dir: Directorio que contiene los archivos de Pareto
        mode: Modo para calcular el punto ideal (min, weighted, geomedian, etc.)
        weights: Pesos para el modo weighted

    Returns:
        Ruta al archivo de Pareto más cercano al ideal
    """
    pareto_path = Path(pareto_dir)
    if not pareto_path.exists():
        if not pareto_dir.endswith("pareto"):
            pareto_path = Path(pareto_dir) / "pareto"
        if not pareto_path.exists():
            raise ValueError(f"Directorio no encontrado: {pareto_dir}")

    entries = sorted(glob.glob(str(pareto_path / "*.json")))
    if not entries:
        raise ValueError(f"No se encontraron archivos de Pareto en: {pareto_path}")

    points = []
    filenames = {}
    w_keys = []
    metric_names = None
    metric_name_map = None

    for entry_path in entries:
        try:
            with open(entry_path) as f:
                entry = json.load(f)
            
            h = os.path.splitext(os.path.basename(entry_path))[0].split("_")[-1]
            
            if metric_names is None:
                metric_names = entry.get("optimize", {}).get("scoring", [])
                metric_name_map = {f"w_{i}": name for i, name in enumerate(metric_names)}
            
            if not w_keys:
                w_keys = sorted(
                    k for k in entry.get("analyses_combined", {}) if k.startswith("w_")
                )
            
            values = [entry.get("analyses_combined", {}).get(k) for k in w_keys]
            if all(v is not None for v in values):
                points.append((*values, h))
                filenames[h] = os.path.split(entry_path)[-1]
        except Exception as e:
            logging.warning(f"Error cargando {entry_path}: {e}")

    if not points:
        raise ValueError("No se encontraron puntos de Pareto válidos")

    values_matrix = np.array([p[:-1] for p in points])
    hashes = [p[-1] for p in points]

    if weights is None:
        weights = tuple([0.0] * values_matrix.shape[1])
    elif len(weights) == 1:
        weights = tuple([weights[0]] * values_matrix.shape[1])

    ideal = compute_ideal(values_matrix, mode=mode, weights=weights)
    mins = np.min(values_matrix, axis=0)
    maxs = np.max(values_matrix, axis=0)

    norm_matrix = np.array(
        [
            [
                (v - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else v
                for i, v in enumerate(row)
            ]
            for row in values_matrix
        ]
    )
    ideal_norm = [
        (ideal[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] > mins[i] else ideal[i]
        for i in range(len(ideal))
    ]

    dists = np.linalg.norm(norm_matrix - ideal_norm, axis=1)
    closest_idx = int(np.argmin(dists))

    best_file = os.path.join(pareto_path, filenames[hashes[closest_idx]])
    
    logging.info(f"Mejor Pareto encontrado: {filenames[hashes[closest_idx]]}")
    logging.info(f"Distancia normalizada al ideal: {dists[closest_idx]:.6f}")
    
    return best_file


def run_backtest(
    pareto_file: str,
    disable_plotting: bool = True,
    extra_args: list = None,
):
    """
    Ejecuta un backtest con el archivo de Pareto especificado.

    Args:
        pareto_file: Ruta al archivo de configuración de Pareto
        disable_plotting: Si True, deshabilita el plotting
        extra_args: Lista de argumentos adicionales para pasar al backtest
    """
    if not os.path.exists(pareto_file):
        raise ValueError(f"Archivo de Pareto no encontrado: {pareto_file}")

    logging.info(f"Ejecutando backtest con: {pareto_file}")

    # Construir comando
    cmd = [sys.executable, "src/backtest.py", pareto_file]
    
    if disable_plotting:
        cmd.append("-dp")
    
    if extra_args:
        cmd.extend(extra_args)

    logging.info(f"Comando: {' '.join(cmd)}")
    
    # Ejecutar backtest
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    
    if result.returncode != 0:
        logging.error(f"Backtest falló con código de salida: {result.returncode}")
        sys.exit(result.returncode)
    
    logging.info("Backtest completado exitosamente")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta backtest con configuraciones de Pareto"
    )
    parser.add_argument(
        "pareto_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directorio que contiene los archivos de Pareto (o ruta al directorio pareto/)"
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="Índice del archivo de Pareto a usar (0-based, ordenado por distancia). Si no se especifica, usa el mejor."
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        default=None,
        help="Ruta directa a un archivo de Pareto específico"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="weighted",
        help="Modo para calcular el punto ideal (min, weighted, geomedian, etc.). Default: weighted"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=comma_separated_values_float,
        default=None,
        help="Pesos para el modo weighted (separados por comas)"
    )
    parser.add_argument(
        "--enable-plotting",
        action="store_true",
        help="Habilitar plotting (por defecto está deshabilitado)"
    )
    parser.add_argument(
        "--extra-args",
        type=str,
        default="",
        help="Argumentos adicionales para pasar al backtest (ej: '--backtest.end_date 2025-11-20')"
    )

    args = parser.parse_args()

    # Determinar qué archivo de Pareto usar
    if args.file:
        pareto_file = args.file
        logging.info(f"Usando archivo de Pareto especificado: {pareto_file}")
    elif args.index is not None:
        if not args.pareto_dir:
            logging.error("Se requiere --pareto-dir cuando se usa --index")
            sys.exit(1)
        # Importar función de paretos.py
        sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
        from paretos import get_pareto_by_index
        pareto_file = get_pareto_by_index(args.pareto_dir, args.index)
        logging.info(f"Usando Pareto en índice {args.index}: {pareto_file}")
    elif args.pareto_dir:
        pareto_file = find_best_pareto(args.pareto_dir, mode=args.mode, weights=args.weights)
        logging.info(f"Usando mejor Pareto encontrado: {pareto_file}")
    else:
        # Buscar el último directorio de optimize_results
        optimize_results_dir = "optimize_results"
        if not os.path.exists(optimize_results_dir):
            logging.error(f"Directorio {optimize_results_dir} no encontrado")
            sys.exit(1)
        
        # Encontrar el último directorio
        dirs = [d for d in os.listdir(optimize_results_dir) 
                if os.path.isdir(os.path.join(optimize_results_dir, d))]
        if not dirs:
            logging.error(f"No se encontraron directorios en {optimize_results_dir}")
            sys.exit(1)
        
        latest_dir = max(dirs, key=lambda d: os.path.getmtime(os.path.join(optimize_results_dir, d)))
        pareto_dir = os.path.join(optimize_results_dir, latest_dir, "pareto")
        
        pareto_file = find_best_pareto(pareto_dir, mode=args.mode, weights=args.weights)
        logging.info(f"Usando mejor Pareto del último directorio: {pareto_file}")
    
    # Parsear argumentos extra
    extra_args_list = []
    if args.extra_args:
        extra_args_list = args.extra_args.split()

    # Ejecutar backtest
    run_backtest(
        pareto_file=pareto_file,
        disable_plotting=not args.enable_plotting,
        extra_args=extra_args_list,
    )


if __name__ == "__main__":
    main()

