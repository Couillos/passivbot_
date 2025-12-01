"""
Script para listar y obtener configuraciones de Pareto por índice.

Este script permite:
- Listar todos los archivos de Pareto ordenados por distancia al ideal
- Obtener la ruta de un archivo de Pareto específico por índice
- Útil para scripts bash que necesitan seleccionar configuraciones de Pareto
"""

import os
import sys
import argparse
import glob
import json
from pathlib import Path


def list_pareto_files(pareto_dir: str, show_details: bool = False):
    """
    Lista todos los archivos de Pareto ordenados por distancia.

    Args:
        pareto_dir: Directorio que contiene los archivos de Pareto
        show_details: Si True, muestra detalles adicionales de cada archivo
    """
    pareto_path = Path(pareto_dir)
    if not pareto_path.exists():
        if not pareto_dir.endswith("pareto"):
            pareto_path = Path(pareto_dir) / "pareto"
        if not pareto_path.exists():
            print(f"Error: Directorio no encontrado: {pareto_dir}", file=sys.stderr)
            sys.exit(1)

    # Buscar todos los archivos JSON
    json_files = sorted(glob.glob(str(pareto_path / "*.json")))
    
    if not json_files:
        print(f"No se encontraron archivos de Pareto en: {pareto_path}", file=sys.stderr)
        sys.exit(1)

    # Extraer distancia y ordenar
    files_with_dist = []
    for filepath in json_files:
        filename = os.path.basename(filepath)
        # Formato: {distance}_{hash}.json
        try:
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 2:
                distance = float(parts[0])
                hash_part = "_".join(parts[1:])
                files_with_dist.append((distance, hash_part, filepath))
        except ValueError:
            # Si no se puede parsear, usar 0 como distancia
            files_with_dist.append((0.0, filename.replace(".json", ""), filepath))

    # Ordenar por distancia (menor distancia = mejor)
    files_with_dist.sort(key=lambda x: x[0])

    if show_details:
        print(f"{'Index':<6} {'Distance':<12} {'Hash':<20} {'File'}")
        print("-" * 80)
        for idx, (dist, hash_part, filepath) in enumerate(files_with_dist):
            print(f"{idx:<6} {dist:<12.6f} {hash_part:<20} {os.path.basename(filepath)}")
    else:
        # Solo mostrar rutas
        for _, _, filepath in files_with_dist:
            print(filepath)

    return files_with_dist


def get_pareto_by_index(pareto_dir: str, index: int) -> str:
    """
    Obtiene la ruta del archivo de Pareto en el índice especificado.

    Args:
        pareto_dir: Directorio que contiene los archivos de Pareto
        index: Índice del archivo (0-based, ordenado por distancia)

    Returns:
        Ruta completa al archivo de Pareto
    """
    files_with_dist = list_pareto_files(pareto_dir, show_details=False)
    
    if index < 0 or index >= len(files_with_dist):
        print(
            f"Error: Índice {index} fuera de rango. Hay {len(files_with_dist)} archivos de Pareto.",
            file=sys.stderr
        )
        sys.exit(1)

    return files_with_dist[index][2]


def show_pareto_info(pareto_file: str):
    """
    Muestra información sobre un archivo de Pareto.

    Args:
        pareto_file: Ruta al archivo de Pareto
    """
    try:
        with open(pareto_file, 'r') as f:
            config = json.load(f)
        
        # Mostrar métricas principales
        analyses_combined = config.get("analyses_combined", {})
        scoring = config.get("optimize", {}).get("scoring", [])
        
        print(f"\nArchivo: {os.path.basename(pareto_file)}")
        print(f"\nMétricas de scoring:")
        for i, metric in enumerate(scoring):
            w_key = f"w_{i}"
            value = analyses_combined.get(w_key, "N/A")
            print(f"  {metric}: {value}")
        
        # Mostrar otras métricas importantes si existen
        important_metrics = ["adg", "adg_mean", "sharpe", "sharpe_mean", 
                           "drawdown_worst", "drawdown_worst_mean"]
        print(f"\nOtras métricas:")
        for metric in important_metrics:
            if metric in analyses_combined:
                print(f"  {metric}: {analyses_combined[metric]}")
                
    except Exception as e:
        print(f"Error leyendo archivo: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Lista y obtiene configuraciones de Pareto por índice"
    )
    parser.add_argument(
        "pareto_dir",
        type=str,
        help="Directorio que contiene los archivos de Pareto (o ruta al directorio pareto/)"
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="Índice del archivo de Pareto a obtener (0-based, ordenado por distancia)"
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="Listar todos los archivos de Pareto con detalles"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Mostrar información detallada del archivo de Pareto"
    )

    args = parser.parse_args()

    if args.list:
        # Modo lista: mostrar todos los archivos con detalles
        list_pareto_files(args.pareto_dir, show_details=True)
    elif args.index is not None:
        # Modo índice: obtener archivo específico
        pareto_file = get_pareto_by_index(args.pareto_dir, args.index)
        if args.info:
            show_pareto_info(pareto_file)
        else:
            # Solo imprimir la ruta (útil para scripts bash)
            print(pareto_file)
    else:
        # Por defecto: listar solo rutas
        list_pareto_files(args.pareto_dir, show_details=False)


if __name__ == "__main__":
    main()

