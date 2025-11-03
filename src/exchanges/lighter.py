from passivbot import Passivbot, logging
from uuid import uuid4
import asyncio
import traceback
import json
import numpy as np
import passivbot_rust as pbr
from utils import ts_to_date, utc_ms
from config_utils import require_live_value
from pure_funcs import (
    floatify,
    calc_hash,
    shorten_custom_id,
)
from procedures import print_async_exception

calc_diff = pbr.calc_diff
round_ = pbr.round_
round_up = pbr.round_up
round_dn = pbr.round_dn

try:
    import lighter
    from lighter import ApiClient, AccountApi, OrderApi, CandlestickApi, TransactionApi, FundingApi
    from lighter.lighter_client import Client as LighterSignerClient
    from lighter.modules.blockchain import OrderSide
except ImportError:
    lighter = None
    LighterSignerClient = None
    OrderSide = None
    ApiClient = None
    AccountApi = None
    OrderApi = None
    CandlestickApi = None
    TransactionApi = None
    FundingApi = None
    logging.warning(
        "lighter-v1-python SDK not found. Install with: pip install lighter-v1-python"
    )


class LighterBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36
        self.quote = "USDC"
        self.hedge_mode = False  # Lighter doesn't support hedge mode

        # Verify lighter SDK is available
        if lighter is None or LighterSignerClient is None:
            raise Exception(
                "lighter-v1-python SDK is required. Install with: pip install lighter-v1-python"
            )

        # Validate required user info fields
        if "private_key" not in self.user_info or not self.user_info["private_key"]:
            raise Exception(
                "private_key is required for Lighter exchange in api-keys.json"
            )

        # Account index and API key index (default to 0 if not specified)
        self.account_index = int(self.user_info.get("account_index", 0))
        self.api_key_index = int(self.user_info.get("api_key_index", 2))

        # L1 address (wallet address) - optional but useful for some queries
        self.l1_address = self.user_info.get("wallet_address", "")

    def create_ccxt_sessions(self):
        """
        Lighter doesn't use CCXT, so we create a custom Lighter SDK client instead.
        We disable websocket support for now.
        """
        self.ws_enabled = False
        self.ccp = None

        # Create Lighter SDK clients
        try:
            # Create API client for read-only operations
            self.api_client = ApiClient()

            # Create Signer client for trading operations
            self.signer_client = LighterSignerClient(
                private_key=self.user_info["private_key"],
                account_index=self.account_index,
                api_key_index=self.api_key_index
            )

            # Create API instances
            self.account_api = AccountApi(self.api_client)
            self.order_api = OrderApi(self.api_client)
            self.candlestick_api = CandlestickApi(self.api_client)
            self.transaction_api = TransactionApi(self.api_client)
            self.funding_api = FundingApi(self.api_client)

            logging.info("Lighter SDK clients initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Lighter clients: {e}")
            raise

        # Create a minimal async wrapper to mimic ccxt interface
        self.cca = self._create_ccxt_wrapper()

    def _create_ccxt_wrapper(self):
        """
        Create a wrapper object that mimics CCXT interface but uses Lighter SDK
        """
        class LighterCCXTWrapper:
            def __init__(self, parent):
                self.parent = parent

            async def close(self):
                """Close API client connections"""
                try:
                    await self.parent.api_client.close()
                    await self.parent.signer_client.close()
                except Exception as e:
                    logging.error(f"Error closing Lighter clients: {e}")

        return LighterCCXTWrapper(self)

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm.get("id", symbol)
            self.min_costs[symbol] = (
                0.1 if elm.get("limits", {}).get("cost", {}).get("min") is None
                else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm.get("limits", {}).get("amount", {}).get("min", 0.001)
            self.qty_steps[symbol] = elm.get("precision", {}).get("amount", 0.001)
            self.price_steps[symbol] = elm.get("precision", {}).get("price", 0.01)
            self.c_mults[symbol] = elm.get("contractSize", 1)

            # Set max leverage if available
            if "limits" in elm and "leverage" in elm["limits"]:
                self.max_leverage[symbol] = int(elm["limits"]["leverage"].get("max", 20))
            else:
                self.max_leverage[symbol] = 20  # default

    async def watch_orders(self):
        """
        Lighter websocket support - to be implemented later
        For now, we'll use REST API polling
        """
        logging.warning("Lighter websocket support not yet implemented. Using REST polling.")
        while True:
            try:
                if self.stop_websocket:
                    break
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"exception watch_orders {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_pos_side(self, order_or_position):
        """
        Determine position side for non-hedge mode exchange
        """
        # Check if we have an existing position for this symbol
        symbol = order_or_position.get("symbol", "")

        if symbol in self.positions:
            if self.positions[symbol]["long"]["size"] != 0.0:
                return "long"
            elif self.positions[symbol]["short"]["size"] != 0.0:
                return "short"

        # Determine from order side
        if "side" in order_or_position:
            side = order_or_position["side"]
            reduce_only = order_or_position.get("reduceOnly", False)

            if side == "buy":
                return "short" if reduce_only else "long"
            elif side == "sell":
                return "long" if reduce_only else "short"

        # Default to long
        return "long"

    async def fetch_open_orders(self, symbol: str = None):
        """
        Fetch open orders from Lighter API
        """
        fetched = None
        try:
            # Fetch account information which includes active orders
            account_info = await self._get_account_info()

            if not account_info or "positions" not in account_info:
                return []

            # Parse and format orders from account positions
            open_orders = []
            for position_data in account_info.get("positions", []):
                # Extract orders from position data
                # This will need to be adapted based on actual Lighter API response format
                pass

            return sorted(open_orders, key=lambda x: x.get("timestamp", 0))

        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def _get_account_info(self):
        """
        Helper to get account information from Lighter
        """
        try:
            # Fetch account by index
            account_data = await self.account_api.account(
                by="index",
                value=str(self.account_index)
            )
            return account_data
        except Exception as e:
            logging.error(f"error fetching account info: {e}")
            traceback.print_exc()
            return {}

    async def fetch_positions(self):
        """
        Fetch positions and balance from Lighter
        """
        fetched_positions, account_info = None, None
        try:
            # Fetch account information
            account_info = await self._get_account_info()

            if not account_info:
                return [], 0.0

            # Extract balance from account info
            balance = 0.0
            if "balance" in account_info:
                balance = float(account_info["balance"])
            elif "total_value" in account_info:
                balance = float(account_info["total_value"])

            # Extract positions
            positions = []
            if "positions" in account_info:
                for pos_data in account_info["positions"]:
                    # Parse position data based on actual Lighter API format
                    size = float(pos_data.get("size", 0))
                    if size != 0:
                        position = {
                            "symbol": self._format_symbol(pos_data.get("market_id")),
                            "position_side": "long" if size > 0 else "short",
                            "size": abs(size),
                            "price": float(pos_data.get("entry_price", 0)),
                        }
                        positions.append(position)

            return positions, balance

        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(fetched_positions)
            print_async_exception(account_info)
            traceback.print_exc()
            return False

    def _format_symbol(self, market_id):
        """
        Convert Lighter market_id to symbol format
        """
        # This will need to be implemented based on actual market mapping
        # For now, return a placeholder
        return f"UNKNOWN/{self.quote}:{self.quote}"

    async def fetch_tickers(self):
        """
        Fetch market tickers from Lighter
        """
        fetched = None
        try:
            # Fetch order books to get current prices
            order_books = await self.order_api.order_books()

            if not order_books:
                return {}

            tickers = {}
            for ob in order_books:
                market_id = ob.get("market_id")
                symbol = self._format_symbol(market_id)

                # Get best bid/ask from order book
                bid = float(ob.get("best_bid", 0))
                ask = float(ob.get("best_ask", 0))
                last = (bid + ask) / 2 if bid and ask else 0

                tickers[symbol] = {
                    "symbol": symbol,
                    "bid": bid,
                    "ask": ask,
                    "last": last,
                    "timestamp": utc_ms(),
                }

            return tickers

        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        """
        Fetch OHLCV candlestick data from Lighter
        """
        fetched = None
        try:
            # Get market_id from symbol
            market_id = self._symbol_to_market_id(symbol)

            # Map timeframe to Lighter resolution format
            resolution_map = {
                "1m": "1m",
                "5m": "5m",
                "15m": "15m",
                "1h": "1h",
                "4h": "4h",
                "1d": "1d",
            }
            resolution = resolution_map.get(timeframe, "1m")

            # Fetch candlestick data
            end_time = int(utc_ms() / 1000)  # Convert to seconds
            start_time = end_time - (1000 * 60)  # Last 1000 minutes

            candles = await self.candlestick_api.candlesticks(
                market_id=market_id,
                resolution=resolution,
                start_time=start_time,
                end_time=end_time
            )

            # Format to CCXT-style OHLCV
            ohlcv = []
            if candles and "data" in candles:
                for candle in candles["data"]:
                    ohlcv.append([
                        int(candle.get("timestamp", 0)) * 1000,  # timestamp in ms
                        float(candle.get("open", 0)),
                        float(candle.get("high", 0)),
                        float(candle.get("low", 0)),
                        float(candle.get("close", 0)),
                        float(candle.get("volume", 0)),
                    ])

            return ohlcv

        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    def _symbol_to_market_id(self, symbol: str):
        """
        Convert symbol to Lighter market_id
        """
        # This needs to be implemented based on actual market mapping
        # For now, return 0 as placeholder
        return 0

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        """
        Fetch 1-minute OHLCV candles
        """
        n_candles_limit = 1000 if limit is None else limit
        result = await self.fetch_ohlcv(symbol, timeframe="1m")
        return result[:n_candles_limit] if result else []

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        """
        Fetch PnL history from Lighter
        """
        try:
            # Fetch account PnL data
            pnl_data = await self.account_api.pnl(
                account_index=self.account_index,
                start_time=int(start_time / 1000) if start_time else None,
                end_time=int(end_time / 1000) if end_time else None,
                limit=limit or 100
            )

            pnls = []
            if pnl_data and "data" in pnl_data:
                for pnl_entry in pnl_data["data"]:
                    pnls.append({
                        "timestamp": int(pnl_entry.get("timestamp", 0)) * 1000,
                        "pnl": float(pnl_entry.get("pnl", 0)),
                        "symbol": self._format_symbol(pnl_entry.get("market_id")),
                        "position_side": pnl_entry.get("side", "long"),
                        "id": pnl_entry.get("id", uuid4().hex),
                    })

            return sorted(pnls, key=lambda x: x["timestamp"])

        except Exception as e:
            logging.error(f"error fetching pnls {e}")
            traceback.print_exc()
            return []

    async def execute_order(self, order: dict) -> dict:
        """
        Execute an order on Lighter exchange
        """
        executed = None
        try:
            # Get market_id from symbol
            market_id = self._symbol_to_market_id(order["symbol"])

            # Determine order side
            order_side = "buy" if order["side"] == "buy" else "sell"

            # Get auth token
            auth_token = await self.signer_client.create_auth_token_with_expiry(
                expiry=60  # 60 seconds expiry
            )

            # Create order using Lighter SDK
            executed = await self.signer_client.create_order(
                market_index=market_id,
                base_amount=order["qty"],
                price=order["price"],
                order_type="limit",  # or "market" based on order params
                side=order_side,
                reduce_only=order.get("reduce_only", False),
                client_order_id=order.get("custom_id", ""),
                auth_token=auth_token
            )

            return executed

        except Exception as e:
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellation(self, order: dict) -> dict:
        """
        Cancel an order on Lighter exchange
        """
        executed = None
        try:
            # Get market_id from symbol
            market_id = self._symbol_to_market_id(order["symbol"])

            # Get auth token
            auth_token = await self.signer_client.create_auth_token_with_expiry(
                expiry=60  # 60 seconds expiry
            )

            # Cancel order using Lighter SDK
            executed = await self.signer_client.cancel_order(
                market_index=market_id,
                order_index=order.get("id", 0),
                auth_token=auth_token
            )

            return executed

        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    def did_cancel_order(self, executed, order=None) -> bool:
        """
        Check if order cancellation was successful
        """
        try:
            return "status" in executed and executed["status"] == "success"
        except:
            return False

    def get_order_execution_params(self, order: dict) -> dict:
        """
        Get Lighter-specific order execution parameters
        """
        return {
            "timeInForce": (
                "PO" if require_live_value(self.config, "time_in_force") == "post_only" else "GTC"
            ),
            "reduceOnly": order["reduce_only"],
            "clientOrderId": order["custom_id"],
        }

    async def determine_utc_offset(self, verbose=True):
        """
        Determine UTC offset for Lighter exchange
        """
        # Lighter is blockchain-based, should use UTC
        self.utc_offset = 0
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def update_exchange_config_by_symbols(self, symbols):
        """
        Update exchange configuration for specific symbols
        """
        # TODO: Implement leverage and margin mode settings if supported
        for symbol in symbols:
            try:
                leverage = int(
                    min(
                        self.max_leverage.get(symbol, 20),
                        self.config_get(["live", "leverage"], symbol=symbol),
                    )
                )
                logging.info(f"{symbol}: setting leverage to {leverage}")
                # TODO: Implement actual leverage setting via Lighter SDK

            except Exception as e:
                logging.error(f"{symbol}: error setting leverage {e}")

    async def update_exchange_config(self):
        """
        Update global exchange configuration
        Lighter doesn't support hedge mode
        """
        logging.info("Lighter exchange config: hedge mode not supported")
        pass

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return formatted[: self.custom_id_max_length]

    def symbol_is_eligible(self, symbol):
        """
        Check if a symbol is eligible for trading on Lighter
        """
        try:
            if symbol not in self.markets_dict:
                return False
            # Add Lighter-specific eligibility checks here
            return True
        except Exception as e:
            logging.error(f"error with symbol_is_eligible {e} {symbol}")
            return False
