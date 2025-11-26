"""
Custom Exchange Module for Passivbot
Handles loading of custom market data from CSV files (e.g., traditional stock market data)
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils import date_to_ts, ts_to_date, make_get_filepath
from downloader import dump_ohlcv_data, load_ohlcv_data, ensure_millis


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


CUSTOM_DATA_DIR = "custom_exchange_data"


def scan_custom_data_directory() -> Dict[str, str]:
    """
    Scan custom_exchange_data directory for available CSV files

    Returns:
        Dict mapping symbol names to CSV file paths
        Example: {"TSLA": "custom_exchange_data/TSLA_1min.csv"}
    """
    if not os.path.exists(CUSTOM_DATA_DIR):
        logging.warning(f"Custom data directory not found: {CUSTOM_DATA_DIR}")
        return {}

    available_files = {}
    for filename in os.listdir(CUSTOM_DATA_DIR):
        if filename.endswith(".csv"):
            # Extract symbol from filename (e.g., "TSLA_1min.csv" -> "TSLA")
            symbol = filename.replace("_1min.csv", "").replace(".csv", "").upper()
            filepath = os.path.join(CUSTOM_DATA_DIR, filename)
            available_files[symbol] = filepath
            logging.info(f"Found custom data: {symbol} -> {filepath}")

    return available_files


def get_csv_date_range(csv_filepath: str) -> tuple:
    """
    Extract start and end dates from CSV file

    Returns:
        Tuple of (start_date_str, end_date_str)
    """
    try:
        # Read only first and last few rows to get date range
        df_first = pd.read_csv(csv_filepath, nrows=1)
        df_last = pd.read_csv(csv_filepath).tail(1)

        if "datetime" in df_first.columns:
            start_dt = pd.to_datetime(df_first["datetime"].iloc[0])
            end_dt = pd.to_datetime(df_last["datetime"].iloc[0])

            start_date = start_dt.strftime("%Y-%m-%d")
            end_date = end_dt.strftime("%Y-%m-%d")

            return start_date, end_date
        else:
            logging.warning(f"No datetime column found in {csv_filepath}")
            return None, None
    except Exception as e:
        logging.error(f"Error reading date range from {csv_filepath}: {e}")
        return None, None


class CustomOHLCVManager:
    """
    Manager for custom market data (stocks, ETFs, etc.)
    Loads data from CSV files and converts to passivbot format
    """

    def __init__(
        self,
        data_dir: str = CUSTOM_DATA_DIR,
        cache_dir: str = "historical_data/ohlcvs_custom",
        verbose: bool = True,
    ):
        """
        Initialize Custom OHLCV Manager

        Args:
            data_dir: Directory containing CSV files
            cache_dir: Directory to cache processed data
            verbose: Enable verbose logging
        """
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.exchange = "custom"

        # Scan available data files
        self.available_symbols = scan_custom_data_directory()

        if not self.available_symbols:
            logging.warning(f"No custom data files found in {data_dir}")

        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)

        # Market metadata for all available symbols
        self.market_metadata = self._create_market_metadata()

    def _create_market_metadata(self) -> Dict[str, Any]:
        """
        Create fictional but realistic market metadata for all available symbols
        """
        metadata = {}
        for symbol in self.available_symbols.keys():
            metadata[symbol] = {
                "symbol": f"{symbol}/USD:USD",
                "maker": 0.001,  # 0.1% maker fee (typical for stocks)
                "taker": 0.001,  # 0.1% taker fee
                "maker_fee": 0.001,
                "taker_fee": 0.001,
                "contractSize": 1.0,  # 1 share per contract
                "c_mult": 1.0,
                "min_cost": 1.0,  # Minimum $1 order
                "min_qty": 1.0,  # Minimum 1 share
                "qty_step": 1.0,  # Trade in whole shares
                "price_step": 0.01,  # Penny increments
                "precision": {
                    "amount": 1.0,
                    "price": 0.01,
                },
                "limits": {
                    "amount": {"min": 1.0, "max": 1000000.0},
                    "cost": {"min": 1.0, "max": None},
                    "price": {"min": 0.01, "max": None},
                },
                "hedge_mode": False,  # Stocks don't have hedge mode
                "exchange": "custom",
            }
        return metadata

    def load_csv_data(self, symbol: str) -> pd.DataFrame:
        """
        Load and parse CSV data for a specific symbol

        Args:
            symbol: Trading symbol (e.g., "TSLA")

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        if symbol not in self.available_symbols:
            raise ValueError(f"Symbol {symbol} not found in {self.data_dir}")

        csv_filepath = self.available_symbols[symbol]

        if self.verbose:
            logging.info(f"Loading CSV data for {symbol} from {csv_filepath}")

        # Read CSV
        df = pd.read_csv(csv_filepath)

        # Parse datetime column
        if "datetime" in df.columns:
            df["timestamp"] = pd.to_datetime(df["datetime"]).astype(int) // 10**6  # Convert to milliseconds
        else:
            raise ValueError(f"CSV for {symbol} must contain 'datetime' column")

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV for {symbol} must contain '{col}' column")

        # Add volume if missing (use synthetic volume based on price movement)
        if "volume" not in df.columns:
            # Synthetic volume: base volume + volatility component
            price_range = (df["high"] - df["low"]) / df["close"]
            base_volume = 1000000  # 1M base volume
            df["volume"] = base_volume * (1 + price_range * 10)

        # Select and reorder columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

        if self.verbose:
            logging.info(f"Loaded {len(df)} rows for {symbol} from {ts_to_date(df['timestamp'].iloc[0])} to {ts_to_date(df['timestamp'].iloc[-1])}")

        return df

    def prepare_and_cache_data(self, symbol: str, df: pd.DataFrame) -> str:
        """
        Prepare data and cache it in daily files (passivbot format)

        Args:
            symbol: Trading symbol
            df: DataFrame with columns [timestamp, open, high, low, close, volume]

        Returns:
            Path to cache directory
        """
        cache_path = os.path.join(self.cache_dir, symbol)
        os.makedirs(cache_path, exist_ok=True)

        # Group by day
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        grouped = df.groupby("date")

        cached_files = 0
        for date, day_df in grouped:
            date_str = str(date)
            cache_file = os.path.join(cache_path, f"{date_str}.npy")

            # Skip if already cached
            if os.path.exists(cache_file):
                continue

            # Save daily data
            day_data = day_df[["timestamp", "open", "high", "low", "close", "volume"]].values
            dump_ohlcv_data(day_data, cache_file)
            cached_files += 1

        if self.verbose and cached_files > 0:
            logging.info(f"Cached {cached_files} daily files for {symbol} to {cache_path}")

        return cache_path

    def get_market_specific_settings(self, symbol: str) -> Dict[str, Any]:
        """
        Get market metadata for a symbol
        """
        if symbol not in self.market_metadata:
            raise ValueError(f"Unknown symbol: {symbol}")
        return self.market_metadata[symbol].copy()

    def has_coin(self, coin: str) -> bool:
        """
        Check if coin/symbol is available
        """
        return coin.upper() in self.available_symbols

    def get_symbol(self, coin: str) -> str:
        """
        Get exchange symbol for coin
        """
        if self.has_coin(coin):
            return f"{coin.upper()}/USD:USD"
        return None

    def get_available_coins(self) -> List[str]:
        """
        Get list of all available coins/symbols
        """
        return list(self.available_symbols.keys())


async def prepare_custom_hlcvs(config: dict) -> tuple:
    """
    Prepare HLCV data from custom CSV files
    Supports multiple symbols as specified in config["live"]["approved_coins"]

    Args:
        config: Backtest configuration

    Returns:
        Tuple of (mss, timestamps, hlcvs, btc_usd_prices)
    """
    from downloader import compute_per_coin_warmup_minutes, compute_backtest_warmup_minutes
    from procedures import get_first_timestamps_unified

    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]

    # Initialize custom manager
    manager = CustomOHLCVManager()

    # Get requested coins from config
    approved_coins = config.get("live", {}).get("approved_coins", [])

    # Handle both list and dict formats for approved_coins
    if isinstance(approved_coins, dict):
        # Dict format: {'long': ['TSLA'], 'short': ['TSLA']}
        # Combine both long and short coins into a single list
        coin_list = []
        for side in ['long', 'short']:
            if side in approved_coins and isinstance(approved_coins[side], list):
                coin_list.extend(approved_coins[side])
        # Remove duplicates while preserving order
        approved_coins = list(dict.fromkeys(coin_list))
        logging.info(f"Extracted coins from dict format: {approved_coins}")

    if not approved_coins:
        logging.warning("No approved_coins specified, using all available symbols")
        approved_coins = manager.get_available_coins()

    # Filter to only available coins (convert to uppercase for matching)
    coins = []
    logging.info(f"Checking approved_coins: {approved_coins}")
    logging.info(f"Available symbols in manager: {manager.available_symbols}")
    for coin in approved_coins:
        if isinstance(coin, str):
            logging.info(f"Checking coin: {coin}, has_coin: {manager.has_coin(coin)}")
            if manager.has_coin(coin):
                coins.append(coin.upper())
            else:
                logging.warning(f"Coin {coin} not available in manager")
        else:
            logging.warning(f"Skipping non-string coin: {coin}")

    if not coins:
        raise ValueError(f"No valid coins found. Available: {manager.get_available_coins()}")

    logging.info(f"Preparing data for coins: {coins}")

    # Compute warmup
    warmup_minutes = 0
    try:
        warmup_minutes = compute_backtest_warmup_minutes(config)
    except Exception as e:
        logging.warning(f"Could not compute warmup: {e}")

    warmup_ms = warmup_minutes * 60 * 1000
    start_ts = date_to_ts(start_date)
    end_ts = date_to_ts(end_date)
    effective_start_ts = max(0, start_ts - warmup_ms)

    # Load data for all coins
    coin_dataframes = {}
    for coin in coins:
        try:
            df = manager.load_csv_data(coin)
            cache_path = manager.prepare_and_cache_data(coin, df)

            # Filter by date range
            df_filtered = df[
                (df["timestamp"] >= effective_start_ts) & (df["timestamp"] <= end_ts)
            ].copy()

            if len(df_filtered) > 0:
                coin_dataframes[coin] = df_filtered
                logging.info(f"{coin}: {len(df_filtered)} rows from {ts_to_date(df_filtered['timestamp'].iloc[0])} to {ts_to_date(df_filtered['timestamp'].iloc[-1])}")
            else:
                logging.warning(f"No data for {coin} in date range {start_date} to {end_date}")
        except Exception as e:
            logging.error(f"Error loading data for {coin}: {e}")
            import traceback
            traceback.print_exc()

    if not coin_dataframes:
        raise ValueError(f"No data found for any coin between {start_date} and {end_date}")

    # Create unified timestamp array covering all coins
    # Only include timestamps where at least one coin has data (no trading on weekends/nights)
    all_timestamps = []
    for df in coin_dataframes.values():
        all_timestamps.extend(df["timestamp"].values)

    all_timestamps = np.unique(sorted(all_timestamps))
    n_timesteps = len(all_timestamps)
    n_coins = len(coin_dataframes)

    logging.info(f"Creating unified array with {n_timesteps} timesteps and {n_coins} coins")

    # Create HLCVS array: shape (n_timesteps, n_coins, 4) where last dim is [high, low, close, volume]
    hlcvs = np.full((n_timesteps, n_coins, 4), np.nan, dtype=np.float64)

    # Create market specific settings
    mss = {}
    per_coin_warmups = compute_per_coin_warmup_minutes(config)
    default_warmup = int(per_coin_warmups.get("__default__", warmup_minutes))

    # Fill data for each coin
    for coin_idx, coin in enumerate(sorted(coin_dataframes.keys())):
        df = coin_dataframes[coin]

        # Reindex to match unified timestamps and forward fill to extend last known values
        df_indexed = df.set_index("timestamp").reindex(all_timestamps).ffill()

        # Fill HLCVS data [high, low, close, volume]
        hlcvs[:, coin_idx, 0] = df_indexed["high"].values
        hlcvs[:, coin_idx, 1] = df_indexed["low"].values
        hlcvs[:, coin_idx, 2] = df_indexed["close"].values
        hlcvs[:, coin_idx, 3] = df_indexed["volume"].values

        # Find valid data range for this coin
        valid_mask = np.isfinite(hlcvs[:, coin_idx, 2])  # Use close price to determine validity
        if valid_mask.any():
            first_valid_idx = int(np.where(valid_mask)[0][0])
            last_valid_idx = int(np.where(valid_mask)[0][-1])
        else:
            first_valid_idx = 0
            last_valid_idx = 0

        # Create market settings for this coin
        mss[coin] = manager.get_market_specific_settings(coin)
        mss[coin]["first_valid_index"] = first_valid_idx
        mss[coin]["last_valid_index"] = last_valid_idx

        coin_warmup = int(per_coin_warmups.get(coin, default_warmup))
        mss[coin]["warmup_minutes"] = coin_warmup
        mss[coin]["trade_start_index"] = min(first_valid_idx + coin_warmup, last_valid_idx)

    # BTC/USD prices: use 1.0 for stocks (no BTC collateral needed)
    btc_usd_prices = np.ones(n_timesteps, dtype=np.float64)

    # Add metadata
    mss["__meta__"] = {
        "requested_start_ts": int(start_ts),
        "requested_start_date": ts_to_date(start_ts),
        "effective_start_ts": int(all_timestamps[0]),
        "effective_start_date": ts_to_date(all_timestamps[0]),
        "warmup_minutes_requested": warmup_minutes,
        "warmup_minutes_provided": int((start_ts - all_timestamps[0]) / 60000) if start_ts > all_timestamps[0] else 0,
    }

    logging.info(f"Prepared HLCVS with shape {hlcvs.shape}")

    return mss, all_timestamps, hlcvs, btc_usd_prices
