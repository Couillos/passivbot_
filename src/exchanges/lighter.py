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
    lighter_available = True
except ImportError as e:
    lighter = None
    lighter_available = False
    logging.warning(
        f"lighter-v1-python SDK not found. Install with: pip install lighter-v1-python. Import error: {e}"
    )


class LighterBot(Passivbot):
    def __init__(self, config: dict):
        # Verify lighter SDK is available first
        if not lighter_available or lighter is None:
            raise Exception(
                "lighter-v1-python SDK is required. Install with: pip install lighter-v1-python"
            )

        # Pre-initialize attributes needed by create_ccxt_sessions
        from procedures import load_user_info
        user_str = require_live_value(config, "user")
        temp_user_info = load_user_info(user_str)

        # Validate required user info fields
        if "private_key" not in temp_user_info or not temp_user_info["private_key"]:
            raise Exception(
                "private_key is required for Lighter exchange in api-keys.json"
            )
        
        # Set defaults
        self.l1_address = temp_user_info.get("wallet_address", "")
        self.is_testnet = bool(temp_user_info.get("testnet", False))
        self.account_index = int(temp_user_info.get("account_index", 0))
        self.api_key_index = int(temp_user_info.get("api_key_index", 0))
        
        # Store user_info temporarily for create_ccxt_sessions
        self._temp_user_info = temp_user_info

        # Log the mode
        if self.is_testnet:
            logging.info("Lighter: Using TESTNET mode")
        else:
            logging.info("Lighter: Using MAINNET mode")

        # Now call parent __init__ which will call create_ccxt_sessions()
        super().__init__(config)

        # Set Lighter-specific settings
        self.custom_id_max_length = 36
        self.quote = "USDC"
        self.hedge_mode = False  # Lighter doesn't support hedge mode

    def create_ccxt_sessions(self):
        """
        Lighter doesn't use CCXT, so we create a custom Lighter SDK client instead.
        """
        self.ws_enabled = False
        self.ccp = None

        # Get credentials
        if hasattr(self, 'user_info') and self.user_info:
            private_key = self.user_info.get("private_key")
        elif hasattr(self, '_temp_user_info'):
            private_key = self._temp_user_info.get("private_key")
        else:
            raise Exception("private_key not found")
        
        if not private_key:
            raise Exception("private_key is required")
        
        # Determine API URL based on testnet flag
        if self.is_testnet:
            url = "https://testnet.zklighter.elliot.ai"
        else:
            url = "https://mainnet.zklighter.elliot.ai"
        
        # Store connection parameters for lazy initialization
        self._lighter_params = {
            'url': url,
            'private_key': private_key,
            'account_index': self.account_index,
            'api_key_index': self.api_key_index
        }
        
        # Initialize clients as None - will be created on first use
        self.signer_client = None
        self.api_client = None
        self.account_api = None
        self.order_api = None
        self.transaction_api = None
        
        network_type = "TESTNET" if self.is_testnet else "MAINNET"
        logging.info(f"Lighter SDK configured for {network_type} at {url}")
    
    async def _init_lighter_clients(self):
        """Lazy initialization of Lighter SDK clients"""
        if self.signer_client is not None:
            return  # Already initialized
        
        try:
            # Create SignerClient for transactions (orders, cancellations)
            self.signer_client = lighter.SignerClient(**self._lighter_params)
            
            # Create ApiClient and API instances for queries
            configuration = lighter.Configuration(host=self._lighter_params['url'])
            self.api_client = lighter.ApiClient(configuration)
            self.account_api = lighter.AccountApi(self.api_client)
            self.order_api = lighter.OrderApi(self.api_client)
            self.transaction_api = lighter.TransactionApi(self.api_client)
            
            # Verify connection
            err = self.signer_client.check_client()
            if err:
                raise Exception(f"SignerClient check failed: {err}")
            
            logging.info("Lighter SDK clients initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize Lighter clients: {e}")
            raise

    async def determine_utc_offset(self, verbose=True):
        """Override to avoid CCXT - Lighter uses direct API"""
        # Lighter API doesn't require time offset calculation
        self.utc_offset = 0
        if verbose:
            logging.info("Lighter: Using UTC time (no offset)")

    async def init_markets(self, verbose=True):
        """Initialize markets from Lighter API"""
        from utils import utc_ms
        
        self.init_markets_last_update_ms = utc_ms()
        await self.update_exchange_config()
        
        # Initialize Lighter clients
        await self._init_lighter_clients()
        
        # Fetch markets from Lighter API
        # For now, use hardcoded markets - TODO: fetch from API
        self.markets_dict = {
            "BTC/USDC:USDC": {
                "id": "WBTC-USDC",
                "symbol": "BTC/USDC:USDC",
                "base": "BTC",
                "quote": "USDC",
                "settle": "USDC",
                "type": "swap",
                "spot": False,
                "swap": True,
                "linear": True,
                "active": True,
                "contract": True,
                "contractSize": 1.0,
                "precision": {"amount": 0.001, "price": 0.01},
                "limits": {
                    "amount": {"min": 0.001, "max": 1000},
                    "price": {"min": 0.01, "max": 1000000},
                    "cost": {"min": 10.0, "max": None},
                    "leverage": {"min": 1, "max": 50}
                },
                "info": {"maxLeverage": 50, "market_index": 0}
            },
            "ETH/USDC:USDC": {
                "id": "WETH-USDC",
                "symbol": "ETH/USDC:USDC",
                "base": "ETH",
                "quote": "USDC",
                "settle": "USDC",
                "type": "swap",
                "spot": False,
                "swap": True,
                "linear": True,
                "active": True,
                "contract": True,
                "contractSize": 1.0,
                "precision": {"amount": 0.01, "price": 0.01},
                "limits": {
                    "amount": {"min": 0.01, "max": 10000},
                    "price": {"min": 0.01, "max": 100000},
                    "cost": {"min": 10.0, "max": None},
                    "leverage": {"min": 1, "max": 50}
                },
                "info": {"maxLeverage": 50, "market_index": 1}
            }
        }
        
        if verbose:
            logging.info(f"Lighter: Initialized with {len(self.markets_dict)} markets")
        
        await self.determine_utc_offset(verbose)
        
        # Set eligible symbols
        self.eligible_symbols = set(self.markets_dict.keys())
        self.ineligible_symbols = {}
        
        self.set_market_specific_settings()
        
        if self.markets_dict:
            self.max_len_symbol = max([len(s) for s in self.markets_dict])
            self.sym_padding = max(self.sym_padding, self.max_len_symbol + 1)
        else:
            self.max_len_symbol = 20
            self.sym_padding = 21
        
        self.init_coin_overrides()
        self.refresh_approved_ignored_coins_lists()
        self.set_wallet_exposure_limits()
        
        await self.update_positions()
        await self.update_open_orders()
        await self.update_effective_min_cost()
        
        if self.is_forager_mode():
            await self.update_first_timestamps()

    def set_market_specific_settings(self):
        """Override to set Lighter-specific market settings"""
        # Call parent to set symbol_ids
        super().set_market_specific_settings()
        
        # Set min_qtys from market limits
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.min_qtys[symbol] = elm.get("limits", {}).get("amount", {}).get("min", 0.001)
            self.qty_steps[symbol] = elm.get("precision", {}).get("amount", 0.001)
            self.price_steps[symbol] = elm.get("precision", {}).get("price", 0.01)
            self.min_costs[symbol] = elm.get("limits", {}).get("cost", {}).get("min", 10.0)
            self.c_mults[symbol] = elm.get("contractSize", 1.0)
            # Lighter uses 10x leverage for all markets
            self.max_leverage[symbol] = 10

    def _get_market_index(self, symbol: str) -> int:
        """Get market_index for a symbol"""
        market = self.markets_dict.get(symbol)
        if not market:
            raise Exception(f"Unknown symbol: {symbol}")
        return market["info"]["market_index"]

    def _symbol_to_orderbook_id(self, symbol: str) -> str:
        """Convert symbol to Lighter orderbook ID (e.g., BTC/USDC:USDC -> WBTC-USDC)"""
        market = self.markets_dict.get(symbol)
        if not market:
            raise Exception(f"Unknown symbol: {symbol}")
        return market["id"]

    async def fetch_tickers(self):
        """Fetch ticker data from Lighter using the CandlestickApi"""
        await self._init_lighter_clients()
        
        try:
            # Initialize CandlestickApi if not already done
            if not hasattr(self, 'candlestick_api'):
                self.candlestick_api = lighter.CandlestickApi(self.api_client)
            
            tickers = {}
            for symbol in self.markets_dict:
                try:
                    orderbook_symbol = self._symbol_to_orderbook_id(symbol)
                    
                    # Fetch latest 1-minute candle
                    candles = await self.candlestick_api.candlestick(
                        orderbook_symbol=orderbook_symbol,
                        interval='1m',
                        limit=1
                    )
                    
                    if candles and hasattr(candles, 'candles') and len(candles.candles) > 0:
                        latest = candles.candles[0]
                        close_price = float(latest.close)
                        high_price = float(latest.high)
                        low_price = float(latest.low)
                        
                        tickers[symbol] = {
                            "symbol": symbol,
                            "last": close_price,
                            "bid": (close_price + low_price) / 2,
                            "ask": (close_price + high_price) / 2,
                            "timestamp": utc_ms()
                        }
                        logging.info(f"Lighter ticker {symbol}: price={close_price}")
                    else:
                        logging.warning(f"No candle data for {symbol}")
                except Exception as e:
                    logging.error(f"Error fetching ticker for {symbol}: {e}")
            
            return tickers
        except Exception as e:
            logging.error(f"Error fetching tickers: {e}")
            traceback.print_exc()
            return {}

    async def fetch_open_orders(self, symbol: str = None):
        """Fetch open orders from Lighter"""
        await self._init_lighter_clients()
        
        try:
            # Create auth token for API calls
            auth, err = self.signer_client.create_auth_token_with_expiry()
            if err:
                raise Exception(f"Failed to create auth token: {err}")
            
            open_orders = []
            symbols_to_fetch = [symbol] if symbol else list(self.markets_dict.keys())
            
            for sym in symbols_to_fetch:
                market_index = self._get_market_index(sym)
                
                # Fetch active orders for this market
                orders = await self.order_api.account_active_orders(
                    account_index=self.account_index,
                    market_id=market_index,
                    auth=auth
                )
                
                if orders and hasattr(orders, 'orders'):
                    for order in orders.orders:
                        open_orders.append({
                            "symbol": sym,
                            "id": str(order.order_index),
                            "price": float(order.price.replace(".", "")),
                            "amount": float(order.base_amount.replace(".", "")),
                            "side": "buy" if not order.is_ask else "sell",
                            "type": "limit",  # Lighter uses limit orders by default
                            "timestamp": None  # Order timestamp not in API response
                        })
            
            return open_orders
        except Exception as e:
            logging.error(f"Error fetching open orders: {e}")
            traceback.print_exc()
            return []

    async def fetch_positions(self) -> tuple:
        """Fetch positions and balance from Lighter"""
        await self._init_lighter_clients()
        
        try:
            # Create auth token
            auth, err = self.signer_client.create_auth_token_with_expiry()
            if err:
                raise Exception(f"Failed to create auth token: {err}")
            
            # Fetch account info - pass auth via headers
            account_info = await self.account_api.account(
                by="index",
                value=str(self.account_index),
                _headers={'Authorization': auth}
            )
            
            # Extract balance from account
            balance = 0.0
            if account_info and hasattr(account_info, 'accounts') and len(account_info.accounts) > 0:
                acc = account_info.accounts[0]
                # Use collateral as balance (USDC) - bot handles leverage internally
                if hasattr(acc, 'collateral'):
                    balance = float(acc.collateral)
                    logging.info(f"Lighter balance: {balance} USDC (collateral)")
                
                # Extract positions
                positions = []
                if hasattr(acc, 'positions'):
                    for pos in acc.positions:
                        # Map market_id to symbol
                        symbol = None
                        for s, m in self.markets_dict.items():
                            if m["info"]["market_index"] == pos.market_id:
                                symbol = s
                                break
                        
                        if symbol and hasattr(pos, 'position') and float(pos.position) != 0:
                            size = float(pos.position)
                            positions.append({
                                "symbol": symbol,
                                "side": "long" if size > 0 else "short",
                                "size": abs(size),
                                "entryPrice": float(pos.avg_entry_price) if hasattr(pos, 'avg_entry_price') else 0.0,
                                "timestamp": utc_ms()
                            })
                
                logging.info(f"DEBUG LIGHTER: Returning positions={len(positions)}, balance={balance}")
                return positions, balance
            
            # No account found
            logging.error("DEBUG LIGHTER: No account found in API response!")
            return [], 0.0
            
        except Exception as e:
            logging.error(f"Error fetching positions: {e}")
            traceback.print_exc()
            return []

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        """Fetch OHLCV candles from Lighter"""
        await self._init_lighter_clients()
        
        try:
            if not hasattr(self, 'candlestick_api'):
                self.candlestick_api = lighter.CandlestickApi(self.api_client)
            
            orderbook_symbol = self._symbol_to_orderbook_id(symbol)
            
            # Map timeframe to Lighter interval
            interval_map = {
                "1m": "1m",
                "3m": "3m",
                "5m": "5m",
                "15m": "15m",
                "30m": "30m",
                "1h": "1h",
                "2h": "2h",
                "4h": "4h",
                "6h": "6h",
                "12h": "12h",
                "1d": "1d",
                "1w": "1w"
            }
            interval = interval_map.get(timeframe, "1m")
            
            candles = await self.candlestick_api.candlestick(
                orderbook_symbol=orderbook_symbol,
                interval=interval,
                limit=1000
            )
            
            if not candles or not hasattr(candles, 'candles'):
                return []
            
            # Convert to OHLCV format [timestamp, open, high, low, close, volume]
            result = []
            for candle in candles.candles:
                result.append([
                    int(candle.timestamp) * 1000,  # Convert to ms
                    float(candle.open),
                    float(candle.high),
                    float(candle.low),
                    float(candle.close),
                    float(candle.volume) if hasattr(candle, 'volume') else 0.0
                ])
            
            return result
        except Exception as e:
            logging.error(f"Error fetching OHLCV for {symbol}: {e}")
            traceback.print_exc()
            return []

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        """Fetch 1-minute candles"""
        n_candles_limit = 1000 if limit is None else limit
        result = await self.fetch_ohlcv(symbol, timeframe="1m")
        return result[:n_candles_limit] if result else []

    async def fetch_pnls(self, start_time=None, limit=None):
        """Fetch PnL history from Lighter"""
        # Lighter doesn't have a direct PnL history endpoint
        # Return empty list for now - PnL will be calculated from positions
        return []

    async def execute_order(self, order: dict) -> dict:
        """Execute an order on Lighter"""
        await self._init_lighter_clients()
        
        try:
            symbol = order["symbol"]
            market_index = self._get_market_index(symbol)
            
            # Convert amount and price to Lighter format (integer with scaling)
            base_amount = int(order["amount"] * 10000)  # Scale to 4 decimals
            price = int(order["price"] * 100)  # Scale to 2 decimals
            
            is_ask = 1 if order["side"] == "sell" else 0
            
            # Generate unique client_order_index
            client_order_index = int(utc_ms()) % 1000000000
            
            # Create order using SignerClient
            tx, tx_hash, err = await self.signer_client.create_order(
                market_index=market_index,
                client_order_index=client_order_index,
                base_amount=base_amount,
                price=price,
                is_ask=is_ask,
                order_type=lighter.SignerClient.ORDER_TYPE_LIMIT,
                time_in_force=lighter.SignerClient.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                reduce_only=order.get("reduce_only", False),
                trigger_price=0
            )
            
            if err:
                raise Exception(f"Order creation failed: {err}")
            
            return {
                "symbol": symbol,
                "id": str(client_order_index),
                "price": order["price"],
                "amount": order["amount"],
                "side": order["side"],
                "type": "limit",
                "timestamp": utc_ms()
            }
        except Exception as e:
            logging.error(f"Error executing order: {e}")
            traceback.print_exc()
            raise

    async def execute_cancellation(self, order: dict) -> dict:
        """Cancel an order on Lighter"""
        await self._init_lighter_clients()
        
        try:
            symbol = order["symbol"]
            market_index = self._get_market_index(symbol)
            order_index = int(order["id"])
            
            # Cancel order using SignerClient
            tx, tx_hash, err = await self.signer_client.cancel_order(
                market_index=market_index,
                order_index=order_index
            )
            
            if err:
                raise Exception(f"Order cancellation failed: {err}")
            
            return {
                "symbol": symbol,
                "id": order["id"],
                "timestamp": utc_ms()
            }
        except Exception as e:
            logging.error(f"Error cancelling order: {e}")
            traceback.print_exc()
            raise

    async def execute_multiple_orders(self, orders: [dict]) -> [dict]:
        """Execute multiple orders"""
        results = []
        for order in orders:
            try:
                result = await self.execute_order(order)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to execute order {order}: {e}")
        return results

    async def execute_multiple_cancellations(self, orders: [dict]) -> [dict]:
        """Cancel multiple orders"""
        results = []
        for order in orders:
            try:
                result = await self.execute_cancellation(order)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to cancel order {order}: {e}")
        return results

    async def update_exchange_config(self):
        """Update exchange-specific config"""
        # Lighter doesn't support hedge mode
        self.hedge_mode = False

    async def set_leverage(self, symbol: str, leverage: float):
        """Set leverage for a symbol"""
        # Lighter leverage is managed per-position, not per-symbol
        # This is a no-op for now
        logging.info(f"Lighter: Leverage setting not implemented (symbol={symbol}, leverage={leverage})")

    async def close_position(self, position: dict):
        """Close a position by creating a reverse order"""
        try:
            symbol = position["symbol"]
            size = position["size"]
            side = "sell" if position["side"] == "long" else "buy"
            
            # Get current market price
            tickers = await self.fetch_tickers()
            ticker = tickers.get(symbol)
            if not ticker:
                raise Exception(f"No ticker data for {symbol}")
            
            price = ticker["bid"] if side == "sell" else ticker["ask"]
            
            order = {
                "symbol": symbol,
                "amount": size,
                "price": price,
                "side": side,
                "reduce_only": True
            }
            
            return await self.execute_order(order)
        except Exception as e:
            logging.error(f"Error closing position: {e}")
            traceback.print_exc()
            raise
