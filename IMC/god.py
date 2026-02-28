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
SESSION_START = pd.to_datetime("2026-02-28 12:00:00").tz_localize("Europe/London")

# --- Strategy Constants ---
# Strategy 1: Floor Exploit
CUMULATIVE_PRODUCTS = ["WX_SUM", "TIDE_SWING", "LHR_COUNT"]

# Strategy 2: LHR Front-Runner
IMBALANCE_THRESHOLD = 20.0 
DIR_TRADE_VOLUME = 10 

# Strategy 3: ETF Arb
ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ALL_ETF_PRODUCTS = [ETF] + COMPONENTS

POS_LIMIT = 100
MIN_EDGE = 0.5         # REDUCED from 2.0 for late-stage efficiency
UNWIND_EDGE = 0.0      # Accept zero edge if it flattens our inventory
LIMIT_NEAR = 50        
MAX_ARB_VOL = 7
MAX_SKEW = 3.5  

# --- Shared Data Structures ---

@dataclass
class TopOfBook:
    bid_px: Optional[float] = None
    bid_sz: int = 0
    ask_px: Optional[float] = None
    ask_sz: int = 0

    def __repr__(self) -> str:
        b = f"{self.bid_px:.0f}x{self.bid_sz}" if self.bid_px is not None else "---"
        a = f"{self.ask_px:.0f}x{self.ask_sz}" if self.ask_px is not None else "---"
        return f"{b} / {a}"

# --- External API Data Fetchers ---

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
        df_session = df[(df["time"] >= SESSION_START) & (df["time"] < now)].copy()
        if df_session.empty: return 0.0
        return ((df_session["temp"] * df_session["humidity"]) / 100).sum()
    except Exception: return 0.0

def get_tide_swing_floor() -> float:
    try:
        resp = requests.get(
            "https://environment.data.gov.uk/flood-monitoring/id/measures/0006-level-tidal_level-i-15_min-mAOD/readings",
            params={"_sorted": "", "_limit": 200}, timeout=10
        )
        if not resp.ok: return 0.0
        items = resp.json().get("items", [])
        df = pd.DataFrame(items)[["dateTime", "value"]].rename(columns={"dateTime": "time"})
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
        df = df.sort_values("time").reset_index(drop=True)
        now = pd.Timestamp.now(tz="Europe/London")
        df_session = df[(df["time"] >= SESSION_START) & (df["time"] < now)].copy()
        if df_session.empty: return 0.0
        df_session["diff_cm"] = df_session["value"].diff().abs() * 100
        def strangle(diff):
            if pd.isna(diff): return 0
            return max(0, 20 - diff) + max(0, diff - 25)
        return df_session["diff_cm"].apply(strangle).sum()
    except Exception: return 0.0

def get_lhr_count_floor() -> float:
    try:
        now = pd.Timestamp.now(tz="Europe/London")
        url = f"https://{AERODATABOX_HOST}/flights/airports/iata/LHR/{SESSION_START.strftime('%Y-%m-%dT%H:%M')}/{now.strftime('%Y-%m-%dT%H:%M')}?direction=Both"
        resp = requests.get(url, headers={"x-rapidapi-host": AERODATABOX_HOST, "x-rapidapi-key": AERODATABOX_KEY}, timeout=10)
        if not resp.ok: return 0.0
        data = resp.json()
        return float(len(data.get("arrivals", [])) + len(data.get("departures", [])))
    except Exception: return 0.0


# --- THE GOD BOT ---

class GodBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Strategy 1 State (Floors)
        self.floors = {"WX_SUM": 0.0, "TIDE_SWING": 0.0, "LHR_COUNT": 0.0}
        self.last_floor_update = 0
        
        # Strategy 2 State (LHR Front-Runner)
        self.last_trade_interval = None
        
        # Strategy 3 State (ETF Arb)
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_ETF_PRODUCTS}
        self._last_arb_time = 0.0
        self._arb_count = 0
        self._theoretical_pnl = 0.0

    # ─── CORE EVENT ROUTING ──────────────────────────────────────────────────

    def on_orderbook(self, ob: OrderBook):
        # Route 1: Update Top of Book & Check ETF Arbitrage
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

        # Route 2: Check Cumulative Floor Exploit
        if ob.product in CUMULATIVE_PRODUCTS:
            floor = self.floors.get(ob.product, 0.0)
            if floor > 0 and ob.sell_orders:
                best_ask = ob.sell_orders[0]
                if best_ask.price < floor:
                    try:
                        pos = self.get_positions().get(ob.product, 0)
                        if pos < 100: 
                            trade_volume = min(int(best_ask.volume), 100 - pos)
                            if trade_volume > 0:
                                print(f"🚨 FLOOR EXPLOIT! {ob.product} Ask @ {best_ask.price} < Floor ({floor:.2f})")
                                self._send_ioc(OrderRequest(ob.product, best_ask.price, Side.BUY, trade_volume))
                    except Exception: pass

    def on_trades(self, trade: Trade):
        if trade.buyer == self.username or trade.seller == self.username:
            direction = "BOUGHT" if trade.buyer == self.username else "SOLD"
            sign = 1 if direction == "SOLD" else -1
            cost = sign * trade.volume * trade.price
            print(f"✅ {direction} {trade.volume:>3} {trade.product:<12} @ {trade.price:>7.0f} (Cost: {cost:+.0f})")

    # ─── STRATEGY 3: ETF ARBITRAGE LOGIC ──────────────────────────────────────

    def _total_abs(self, p: tuple[int, int, int, int]) -> int:
        return sum(abs(x) for x in p)

    def _near_limit(self, p: tuple[int, int, int, int]) -> bool:
        return any(abs(x) >= LIMIT_NEAR for x in p)

    def _maybe_arb(self):
        now = time.monotonic()
        if now - self._last_arb_time < 0.05: return # 50ms cooldown

        E, A, B, C = self._top[ETF], self._top[COMPONENTS[0]], self._top[COMPONENTS[1]], self._top[COMPONENTS[2]]

        if any(v is None for v in (E.bid_px, E.ask_px, A.bid_px, A.ask_px, B.bid_px, B.ask_px, C.bid_px, C.ask_px)):
            return

        basket_ask = A.ask_px + B.ask_px + C.ask_px
        basket_bid = A.bid_px + B.bid_px + C.bid_px

        try: pos = self.get_positions()
        except Exception: return

        pe, pa, pb, pc = int(pos.get(ETF, 0)), int(pos.get(COMPONENTS[0], 0)), int(pos.get(COMPONENTS[1], 0)), int(pos.get(COMPONENTS[2], 0))
        p0 = (pe, pa, pb, pc)

        # Unwind logic calculation
        p1_after_1 = (pe - 1, pa + 1, pb + 1, pc + 1)
        p2_after_1 = (pe + 1, pa - 1, pb - 1, pc - 1)
        case1_is_unwind = self._total_abs(p1_after_1) < self._total_abs(p0)
        case2_is_unwind = self._total_abs(p2_after_1) < self._total_abs(p0)

        inventory_skew = (pe / POS_LIMIT) * MAX_SKEW
        required_edge1 = UNWIND_EDGE if case1_is_unwind else (MIN_EDGE - inventory_skew)
        required_edge2 = UNWIND_EDGE if case2_is_unwind else (MIN_EDGE + inventory_skew)

        # Case 1: ETF rich -> sell ETF, buy components
        edge1 = E.bid_px - basket_ask
        if edge1 > required_edge1:
            vol = min(min(E.bid_sz, MAX_ARB_VOL), min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_ARB_VOL),
                      POS_LIMIT + pe, POS_LIMIT - pa, POS_LIMIT - pb, POS_LIMIT - pc)
            if vol >= 1:
                self._fire_arb(f"ETF RICH", edge1, vol, [
                    OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol)
                ])
                self._last_arb_time = now
                return

        # Case 2: ETF cheap -> buy ETF, sell components
        edge2 = basket_bid - E.ask_px
        if edge2 > required_edge2:
            vol = min(min(E.ask_sz, MAX_ARB_VOL), min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_ARB_VOL),
                      POS_LIMIT - pe, POS_LIMIT + pa, POS_LIMIT + pb, POS_LIMIT + pc)
            if vol >= 1:
                self._fire_arb(f"ETF CHEAP", edge2, vol, [
                    OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol)
                ])
                self._last_arb_time = now
                return

    def _fire_arb(self, label: str, edge: float, vol: int, orders: list[OrderRequest]):
        self._arb_count += 1
        self._theoretical_pnl += edge * vol
        print(f"\n⚡ [{datetime.now().strftime('%H:%M:%S')}] ARB #{self._arb_count} {label} | Edge: {edge:.1f} | Vol: {vol}")
        threads = [Thread(target=self._send_ioc, args=(o,)) for o in orders]
        for t in threads: t.start()
        for t in threads: t.join()

    # ─── SHARED EXECUTION ────────────────────────────────────────────────────

    def _send_ioc(self, order: OrderRequest):
        """Used by both Floor Exploit and ETF Arb for safe execution."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            try: self.cancel_order(resp.id)
            except Exception: pass

    def _aggress_book(self, product: str, side: Side, volume: int):
        """Used by LHR Interval Front-Runner with built-in position limits."""
        try:
            # Check current position
            pos = self.get_positions().get(product, 0)
        except Exception:
            return # Skip trade if position API fails to prevent blind firing
            
        ob = self.get_orderbook(product)
        
        if side == Side.BUY and ob.sell_orders:
            # Headroom is 100 minus our current long position
            safe_volume = min(volume, 100 - pos) 
            if safe_volume > 0:
                self._send_ioc(OrderRequest(product, ob.sell_orders[0].price, Side.BUY, safe_volume))
                
        elif side == Side.SELL and ob.buy_orders:
            # If we are short, pos is negative. So 100 + (-40) = 60 headroom.
            safe_volume = min(volume, 100 + pos) 
            if safe_volume > 0:
                self._send_ioc(OrderRequest(product, ob.buy_orders[0].price, Side.SELL, safe_volume))

    # ─── STRATEGY 2: INTERVAL FRONT-RUNNER & MAIN LOOP ───────────────────────

    def get_next_interval_window(self):
        now = pd.Timestamp.now(tz="Europe/London")
        if now.minute < 30:
            return now.replace(minute=30, second=0, microsecond=0), now.replace(minute=30, second=0, microsecond=0) + timedelta(minutes=29)
        return (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0), (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0) + timedelta(minutes=29)

    def run_strategy(self):
        print("🚀 Starting GOD BOT (3 Strategies Active)...")
        self.start()
        last_heartbeat = 0
        
        while True:
            try:
                now = pd.Timestamp.now(tz="Europe/London")
                
                # 1. Update Floors (Every 15 mins)
                if time.time() - self.last_floor_update > 900:
                    print(f"\n🔄 Updating Floors...")
                    self.floors["WX_SUM"] = get_wx_sum_floor()
                    self.floors["TIDE_SWING"] = get_tide_swing_floor()
                    self.floors["LHR_COUNT"] = get_lhr_count_floor()
                    self.last_floor_update = time.time()

                # 2. Check LHR Interval Rollover
                if now.minute in [28, 29, 58, 59]:
                    interval_id = now.strftime("%Y%m%d%H") + ("30" if now.minute >= 30 else "00")
                    if self.last_trade_interval != interval_id:
                        start_time, end_time = self.get_next_interval_window()
                        try:
                            start_str, end_str = start_time.strftime("%Y-%m-%dT%H:%M"), end_time.strftime("%Y-%m-%dT%H:%M")
                            url = f"https://{AERODATABOX_HOST}/flights/airports/iata/LHR/{start_str}/{end_str}?direction=Both"
                            resp = requests.get(url, headers={"x-rapidapi-host": AERODATABOX_HOST, "x-rapidapi-key": AERODATABOX_KEY}, timeout=10)
                            if resp.ok:
                                data = resp.json()
                                arr, dep = len(data.get("arrivals", [])), len(data.get("departures", []))
                                metric = 100.0 * (arr - dep) / max(arr + dep, 1)
                                
                                if metric > IMBALANCE_THRESHOLD:
                                    print(f"🚨 ARRIVAL wave (+{metric:.1f}). Buying LHR_INDEX!")
                                    self._aggress_book("LHR_INDEX", Side.BUY, DIR_TRADE_VOLUME)
                                elif metric < -IMBALANCE_THRESHOLD:
                                    print(f"🚨 DEPARTURE wave ({metric:.1f}). Selling LHR_INDEX!")
                                    self._aggress_book("LHR_INDEX", Side.SELL, DIR_TRADE_VOLUME)
                        except Exception as e: pass
                        self.last_trade_interval = interval_id

                # 3. Heartbeat
                if time.time() - last_heartbeat > 60:
                    try:
                        pos = self.get_positions()
                        pos_str = ", ".join([f"{k}:{v}" for k, v in pos.items() if v != 0]) or "Flat"
                        print(f"[{now.strftime('%H:%M:%S')}] 💓 HEARTBEAT | Arbs: {self._arb_count} | Theo PnL: {self._theoretical_pnl:.0f} | Pos -> {pos_str}")
                    except Exception: pass
                    last_heartbeat = time.time()
                    
            except Exception as e:
                print(f"⚠️ Main Loop Error: {e}")
                traceback.print_exc()
            time.sleep(10)

if __name__ == "__main__":
    bot = GodBot(EXCHANGE_URL, USERNAME, PASSWORD)
    try: bot.run_strategy()
    except KeyboardInterrupt:
        bot.cancel_all_orders()
        bot.stop()