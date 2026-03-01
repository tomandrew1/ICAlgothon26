import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from threading import Thread

# Import from the provided framework
from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade

# --- CONFIGURATION ---
EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
USERNAME = "RATT"
PASSWORD = "ratt67"

POS_LIMIT = 100    # Strict position cap
MAX_TRADE_VOL = 5  # Max volume per MM quote or Arb leg
MM_EDGE = 15       # Distance from FV to place MM quotes
SKEW_FACTOR = 10   # Max price skew applied when position is at +/- POS_LIMIT

# --- CONSTANT FAIR VALUES (Markets 1 & 2) ---
FV_TIDE_SPOT = 2750   # Market 1
FV_TIDE_SWING = 506   # Market 2

LONDON_LAT, LONDON_LON = 51.5074, -0.1278

# --- HELPER FUNCTIONS ---

def get_weather_theos():
    """Fetches London weather and calculates theos for Mkt 3 (WX_SPOT) and Mkt 4 (WX_SUM)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LONDON_LAT,
        "longitude": LONDON_LON,
        "minutely_15": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/London",
        "past_days": 2, 
        "forecast_days": 1 
    }
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Weather API Error: {e}")
        return None, None

    df = pd.DataFrame({
        "time": pd.to_datetime(data["minutely_15"]["time"]).tz_localize("Europe/London"),
        "temperature": data["minutely_15"]["temperature_2m"],
        "humidity": data["minutely_15"]["relative_humidity_2m"]
    })
    
    df['temp_F'] = (df['temperature'] * 9/5) + 32
    df['wx_metric'] = df['temp_F'].round() * df['humidity']

    now = pd.Timestamp.now(tz="Europe/London")
    target_sunday = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now.dayofweek != 6:
        target_sunday += pd.Timedelta(days=(6 - now.dayofweek))
    target_saturday = target_sunday - pd.Timedelta(days=1)
    
    session_df = df[(df['time'] > target_saturday) & (df['time'] <= target_sunday)].copy()
    if session_df.empty:
        return None, None

    spot_row = session_df[session_df['time'] == target_sunday]
    wx_spot_theo = spot_row['wx_metric'].iloc[0] if not spot_row.empty else None
    wx_sum_theo = session_df['wx_metric'].sum() / 100

    return wx_spot_theo, wx_sum_theo

def call_payoff(S, K):
    return max(0.0, S - K)

def put_payoff(S, K):
    return max(0.0, K - S)

def lon_fly_payoff(S):
    """Calculates Mkt 8 (LON_FLY) payoff based on the ETF Settlement."""
    return (
        2 * put_payoff(S, 6200)
        + call_payoff(S, 6200)
        - 2 * call_payoff(S, 6600)
        + 3 * call_payoff(S, 7000)
    )

# --- BOT CLASS ---

class UnifiedBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orderbooks = {}
        # LHR_INDEX (Market 6) omitted
        self.products = ["TIDE_SPOT", "TIDE_SWING", "WX_SPOT", "WX_SUM", "LHR_COUNT", "LON_ETF", "LON_FLY"]
        self.etf_fv = None  # Cache for the ETF mid-price

    def on_orderbook(self, ob: OrderBook):
        self.orderbooks[ob.product] = ob

    def on_trades(self, trade: Trade):
        direction = "BOUGHT" if trade.buyer == self.username else "SOLD"
        print(f"  >>> FILL: {direction} {trade.volume}x {trade.product} @ {trade.price}")

    def _send_ioc(self, order: OrderRequest):
        """Simulate Immediate-Or-Cancel for Arbitrage legs."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            try:
                self.cancel_order(resp.id)
            except:
                pass
        return resp

    def check_etf_arbitrage(self, local_positions):
        """Zero-edge ETF Arbitrage for Mkts 1, 3, 5, 7."""
        if not all(p in self.orderbooks for p in ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]):
            return local_positions

        E = self.orderbooks["LON_ETF"]
        A = self.orderbooks["TIDE_SPOT"]
        B = self.orderbooks["WX_SPOT"]
        C = self.orderbooks["LHR_COUNT"]

        if not (E.buy_orders and E.sell_orders and A.buy_orders and A.sell_orders and 
                B.buy_orders and B.sell_orders and C.buy_orders and C.sell_orders):
            return local_positions

        E_bid, E_ask = E.buy_orders[0].price, E.sell_orders[0].price
        basket_bid = A.buy_orders[0].price + B.buy_orders[0].price + C.buy_orders[0].price
        basket_ask = A.sell_orders[0].price + B.sell_orders[0].price + C.sell_orders[0].price

        pe = local_positions.get("LON_ETF", 0)
        pa = local_positions.get("TIDE_SPOT", 0)
        pb = local_positions.get("WX_SPOT", 0)
        pc = local_positions.get("LHR_COUNT", 0)

        orders_to_send = []
        vol = 0

        # Zero edge thresholds
        if E_bid >= basket_ask:
            vol = min(
                E.buy_orders[0].volume, A.sell_orders[0].volume, B.sell_orders[0].volume, C.sell_orders[0].volume,
                MAX_TRADE_VOL,
                POS_LIMIT + pe, POS_LIMIT - pa, POS_LIMIT - pb, POS_LIMIT - pc
            )
            if vol >= 1:
                print(f"[ARB] ETF Rich! Edge: {E_bid - basket_ask}. Selling ETF, Buying Basket.")
                orders_to_send = [
                    OrderRequest("LON_ETF", E_bid, Side.SELL, vol),
                    OrderRequest("TIDE_SPOT", A.sell_orders[0].price, Side.BUY, vol),
                    OrderRequest("WX_SPOT", B.sell_orders[0].price, Side.BUY, vol),
                    OrderRequest("LHR_COUNT", C.sell_orders[0].price, Side.BUY, vol)
                ]
                local_positions["LON_ETF"] -= vol
                local_positions["TIDE_SPOT"] += vol
                local_positions["WX_SPOT"] += vol
                local_positions["LHR_COUNT"] += vol

        elif basket_bid >= E_ask:
            vol = min(
                E.sell_orders[0].volume, A.buy_orders[0].volume, B.buy_orders[0].volume, C.buy_orders[0].volume,
                MAX_TRADE_VOL,
                POS_LIMIT - pe, POS_LIMIT + pa, POS_LIMIT + pb, POS_LIMIT + pc
            )
            if vol >= 1:
                print(f"[ARB] ETF Cheap! Edge: {basket_bid - E_ask}. Buying ETF, Selling Basket.")
                orders_to_send = [
                    OrderRequest("LON_ETF", E_ask, Side.BUY, vol),
                    OrderRequest("TIDE_SPOT", A.buy_orders[0].price, Side.SELL, vol),
                    OrderRequest("WX_SPOT", B.buy_orders[0].price, Side.SELL, vol),
                    OrderRequest("LHR_COUNT", C.buy_orders[0].price, Side.SELL, vol)
                ]
                local_positions["LON_ETF"] += vol
                local_positions["TIDE_SPOT"] -= vol
                local_positions["WX_SPOT"] -= vol
                local_positions["LHR_COUNT"] -= vol

        if orders_to_send:
            threads = [Thread(target=self._send_ioc, args=(o,)) for o in orders_to_send]
            for t in threads: t.start()
            for t in threads: t.join()

        return local_positions

    def place_mm_quotes(self, theos, local_positions):
        """Places GTC Market Making quotes skewed by current position inventory."""
        orders_to_send = []
        
        for product, theo in theos.items():
            if theo is None: continue
            
            pos = local_positions.get(product, 0)
            
            # Dynamic Skew Logic
            # If pos is positive (long), skew is positive -> lower prices
            # If pos is negative (short), skew is negative -> higher prices
            skew = (pos / POS_LIMIT) * SKEW_FACTOR
            
            bid_px = round(theo - MM_EDGE - skew)
            ask_px = round(theo + MM_EDGE - skew)
            
            if pos + MAX_TRADE_VOL <= POS_LIMIT:
                orders_to_send.append(OrderRequest(product, bid_px, Side.BUY, MAX_TRADE_VOL))
            if pos - MAX_TRADE_VOL >= -POS_LIMIT:
                orders_to_send.append(OrderRequest(product, ask_px, Side.SELL, MAX_TRADE_VOL))
                
        if orders_to_send:
            self.send_orders(orders_to_send)

    def run_loop(self):
        self.start()
        print("Starting Unified ETF Arb & Market Maker Bot with Position Skew...")
        
        while True:
            try:
                fv_wx_spot, fv_wx_sum = get_weather_theos()
                
                # --- UPDATE ETF FV BASED ON 10% MARKET WIDTH ---
                etf_ob = self.orderbooks.get("LON_ETF")
                if etf_ob and etf_ob.buy_orders and etf_ob.sell_orders:
                    best_bid = etf_ob.buy_orders[0].price
                    best_ask = etf_ob.sell_orders[0].price
                    mid = (best_bid + best_ask) / 2
                    spread = best_ask - best_bid
                    
                    if spread / mid <= 0.10:
                        self.etf_fv = mid
                
                # Derive LON_FLY from the cached ETF mid
                if self.etf_fv is not None:
                    fv_lon_fly = lon_fly_payoff(self.etf_fv)
                else:
                    fv_lon_fly = None

                # Assemble fair values for all active markets.
                theos = {
                    "TIDE_SPOT": FV_TIDE_SPOT,
                    "TIDE_SWING": FV_TIDE_SWING, 
                    "WX_SPOT": fv_wx_spot,
                    "WX_SUM": fv_wx_sum,
                    "LON_FLY": fv_lon_fly
                }
                
                print(f"\n[Theos] TIDE: {FV_TIDE_SPOT} | SWING: {FV_TIDE_SWING} | WX: {fv_wx_spot} | ETF_MID: {self.etf_fv} | FLY: {fv_lon_fly}")

                local_positions = self.get_positions()
                local_positions = self.check_etf_arbitrage(local_positions)

                self.cancel_all_orders()
                self.place_mm_quotes(theos, local_positions)

                time.sleep(10)
                
            except Exception as e:
                print(f"Main loop error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = UnifiedBot(EXCHANGE_URL, USERNAME, PASSWORD)
    try:
        bot.run_loop()
    except KeyboardInterrupt:
        print("\nShutting down cleanly...")
        bot.cancel_all_orders()
        bot.stop()