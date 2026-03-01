import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import traceback
from dataclasses import dataclass
from threading import Thread
from typing import Optional

# Import the CMI Exchange framework from your bot_template
from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade

# --- Configuration ---
EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
USERNAME = "RATT"
PASSWORD = "ratt67"

AERODATABOX_KEY = "34f9f54137mshaebbce78c55e61dp194894jsnad885717ce27"
AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"
# Settlement session: Sat 12:00 PM to Sun 12:00 PM London 
SESSION_START = pd.to_datetime("2026-02-28 12:00:00").tz_localize("Europe/London")

# --- Strategy Constants ---
CUMULATIVE_PRODUCTS = ["WX_SUM", "TIDE_SWING", "LHR_COUNT"]
IMBALANCE_THRESHOLD = 20.0 
DIR_TRADE_VOLUME = 10 

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ALL_ETF_PRODUCTS = [ETF] + COMPONENTS

POS_LIMIT = 100
MIN_EDGE = 0.5         
UNWIND_EDGE = 0.0      
LIMIT_NEAR = 50        
MAX_ARB_VOL = 7
MAX_SKEW = 3.5  

@dataclass
class TopOfBook:
    bid_px: Optional[float] = None
    bid_sz: int = 0
    ask_px: Optional[float] = None
    ask_sz: int = 0

# --- Helper: Flight Cancellation Filter ---

def get_active_flight_count(data: dict) -> tuple[int, int]:
    """Filters out flights marked as 'Canceled' from API response."""
    arrivals = data.get("arrivals", [])
    departures = data.get("departures", [])
    
    active_arr = [f for f in arrivals if str(f.get("status", "")).lower() != "canceled"]
    active_dep = [f for f in departures if str(f.get("status", "")).lower() != "canceled"]
    
    return len(active_arr), len(active_dep)

# --- Data Fetchers ---

def get_wx_sum_floor() -> float:
    try:
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": 51.5074, "longitude": -0.1278,
            "minutely_15": "temperature_2m,relative_humidity_2m",
            "past_minutely_15": 96, "timezone": "Europe/London",
            "temperature_unit": "fahrenheit" 
        }, timeout=10)
        if not resp.ok: return 0.0
        m = resp.json()["minutely_15"]
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temp": m["temperature_2m"], "humidity": m["relative_humidity_2m"],
        })
        now = pd.Timestamp.now(tz="Europe/London")
        # Market 4: Sum of (T*H)/100 for 15m intervals [cite: 618]
        df_session = df[(df["time"] >= SESSION_START) & (df["time"] < now)].copy()
        if df_session.empty: return 0.0
        return ((df_session["temp"] * df_session["humidity"]) / 100).sum()
    except Exception: return 0.0

def get_lhr_count_floor() -> float:
    """Calculates locked-in LHR_COUNT floor excluding cancellations[cite: 620]."""
    try:
        now = pd.Timestamp.now(tz="Europe/London")
        url = f"https://{AERODATABOX_HOST}/flights/airports/iata/LHR/{SESSION_START.strftime('%Y-%m-%dT%H:%M')}/{now.strftime('%Y-%m-%dT%H:%M')}?direction=Both"
        resp = requests.get(url, headers={"x-rapidapi-host": AERODATABOX_HOST, "x-rapidapi-key": AERODATABOX_KEY}, timeout=15)
        if not resp.ok: return 0.0
        
        arr_cnt, dep_cnt = get_active_flight_count(resp.json())
        return float(arr_cnt + dep_cnt)
    except Exception: return 0.0

# --- THE GOD BOT ---

class GodBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.floors = {"WX_SUM": 0.0, "TIDE_SWING": 0.0, "LHR_COUNT": 0.0}
        self.last_floor_update = 0
        self.last_trade_interval = None
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_ETF_PRODUCTS}
        self._last_arb_time = 0.0
        self._arb_count = 0
        self._theoretical_pnl = 0.0

    def on_orderbook(self, ob: OrderBook):
        if ob.product in ALL_ETF_PRODUCTS:
            top = TopOfBook()
            if ob.buy_orders:
                top.bid_px = ob.buy_orders[0].price
                top.bid_sz = int(ob.buy_orders[0].volume)
            if ob.sell_orders:
                top.ask_px = ob.sell_orders[0].price
                top.ask_sz = int(ob.sell_orders[0].volume)
            self._top[ob.product] = top
            self._maybe_arb()

        if ob.product in CUMULATIVE_PRODUCTS:
            floor = self.floors.get(ob.product, 0.0)
            if floor > 0 and ob.sell_orders:
                best_ask = ob.sell_orders[0]
                # Aggressively buy if market is below known floor [cite: 577]
                if best_ask.price < floor:
                    pos = self.get_positions().get(ob.product, 0)
                    if pos < POS_LIMIT: 
                        vol = min(int(best_ask.volume), POS_LIMIT - pos)
                        if vol > 0:
                            print(f"🚨 FLOOR ARB: {ob.product} @ {best_ask.price} < Floor {floor:.1f}")
                            self._send_ioc(OrderRequest(ob.product, best_ask.price, Side.BUY, vol))

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username or trade.seller == self.username:
            direction = "BOUGHT" if trade.buyer == self.username else "SOLD"
            print(f"✅ Trade: {direction} {trade.volume} {trade.product} @ {trade.price}")

    def _maybe_arb(self):
        """Cross-product Arbitrage between ETF and Components[cite: 576]."""
        now = time.monotonic()
        if now - self._last_arb_time < 0.05: return 

        E, A, B, C = self._top[ETF], self._top[COMPONENTS[0]], self._top[COMPONENTS[1]], self._top[COMPONENTS[2]]
        if any(v.bid_px is None or v.ask_px is None for v in [E, A, B, C]): return

        basket_ask = A.ask_px + B.ask_px + C.ask_px
        basket_bid = A.bid_px + B.bid_px + C.bid_px

        pos = self.get_positions()
        pe, pa, pb, pc = [int(pos.get(p, 0)) for p in ALL_ETF_PRODUCTS]
        
        # Unwind Check
        p0_abs = sum(abs(x) for x in (pe, pa, pb, pc))
        case1_abs = sum(abs(x) for x in (pe-1, pa+1, pb+1, pc+1))
        case2_abs = sum(abs(x) for x in (pe+1, pa-1, pb-1, pc-1))

        skew = (pe / POS_LIMIT) * MAX_SKEW
        
        # Case 1: ETF Rich (Sell ETF, Buy Basket)
        req_1 = UNWIND_EDGE if case1_abs < p0_abs else (MIN_EDGE - skew)
        if (E.bid_px - basket_ask) > req_1:
            vol = min(min(E.bid_sz, A.ask_sz, B.ask_sz, C.ask_sz, MAX_ARB_VOL), 
                      POS_LIMIT + pe, POS_LIMIT - pa, POS_LIMIT - pb, POS_LIMIT - pc)
            if vol > 0:
                self._fire_arb("ETF RICH", vol, [
                    OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol)
                ])

        # Case 2: ETF Cheap (Buy ETF, Sell Basket)
        req_2 = UNWIND_EDGE if case2_abs < p0_abs else (MIN_EDGE + skew)
        if (basket_bid - E.ask_px) > req_2:
            vol = min(min(E.ask_sz, A.bid_sz, B.bid_sz, C.bid_sz, MAX_ARB_VOL),
                      POS_LIMIT - pe, POS_LIMIT + pa, POS_LIMIT + pb, POS_LIMIT + pc)
            if vol > 0:
                self._fire_arb("ETF CHEAP", vol, [
                    OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol)
                ])

    def _fire_arb(self, label, vol, orders):
        print(f"⚡ ARB: {label} Vol: {vol}")
        threads = [Thread(target=self._send_ioc, args=(o,)) for o in orders]
        for t in threads: t.start()
        for t in threads: t.join()
        self._last_arb_time = time.monotonic()

    def _send_ioc(self, order: OrderRequest):
        """Standardized IOC execution."""
        resp = self.send_order(order)
        if resp and resp.status in ["ACTIVE", "PART_FILLED"]:
            if resp.volume > 0:
                self.cancel_order(resp.id)

    def _aggress_book(self, product, side, volume):
        """Safety-checked directional hitting."""
        pos = self.get_positions().get(product, 0)
        safe_vol = min(volume, 100 - pos) if side == Side.BUY else min(volume, 100 + pos)
        if safe_vol <= 0: return

        ob = self.get_orderbook(product)
        if side == Side.BUY and ob.sell_orders:
            self._send_ioc(OrderRequest(product, ob.sell_orders[0].price, Side.BUY, safe_vol))
        elif side == Side.SELL and ob.buy_orders:
            self._send_ioc(OrderRequest(product, ob.buy_orders[0].price, Side.SELL, safe_vol))

    def run_strategy(self):
        self.start()
        last_heartbeat = 0
        while True:
            try:
                now = pd.Timestamp.now(tz="Europe/London")
                if time.time() - self.last_floor_update > 600: # 10 min floor refresh
                    self.floors["WX_SUM"] = get_wx_sum_floor()
                    self.floors["LHR_COUNT"] = get_lhr_count_floor()
                    self.last_floor_update = time.time()

                # Market 6: Airport Metric Front-Running [cite: 621, 734]
                if now.minute in [28, 29, 58, 59]:
                    interval_id = now.strftime("%H%M") + ("30" if now.minute >= 30 else "00")
                    if self.last_trade_interval != interval_id:
                        # Logic to fetch next window and hit LHR_INDEX here
                        self.last_trade_interval = interval_id

                if time.time() - last_heartbeat > 60:
                    pos = self.get_positions()
                    print(f"[{now.strftime('%H:%M')}] 💓 HEARTBEAT | Pos: {pos}")
                    last_heartbeat = time.time()
            except Exception: traceback.print_exc()
            time.sleep(5)

if __name__ == "__main__":
    bot = GodBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_strategy()