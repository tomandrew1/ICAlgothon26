#!/usr/bin/env python3 -u
"""Combined ETF Arbitrage & Weather Market Making Bot.

Combines risk-free ETF arbitrage (LON_ETF vs components) with directional
market-making and taking on WX_SPOT and WX_SUM based on Open-Meteo forecasts.
"""

from __future__ import annotations

import os
import signal
import sys
import time
import requests
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

os.environ["PYTHONUNBUFFERED"] = "1"

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

# ─── Configuration ──────────────────────────────────────────────────────
EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME = "pog"
PASSWORD = "CHAMP"

# Product Definitions
ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
WX_SPOT = "WX_SPOT"
WX_SUM = "WX_SUM"
ALL_PRODUCTS = [ETF, WX_SUM] + COMPONENTS

# Shared Inventory Limit
POS_LIMIT = 100

# ETF Arb Params (LOWERED FOR TESTING)
ETF_MIN_EDGE = 0.5
ETF_UNWIND_EDGE = 0.0
ETF_LIMIT_UNWIND_EDGE = -1.0
ETF_MAX_SKEW = 3.5
LIMIT_NEAR = 90
MIN_COOLDOWN = 0.01

# Weather Params (LOWERED FOR TESTING)
WX_MIN_EDGE = 3.0     # Minimum edge to act as a Taker
WX_QUOTE_SPREAD = 4.0 # How far from FV to place resting Maker orders
WX_MAX_SKEW = 20.0
WX_COOLDOWN = 1.0

# General Limits
MAX_TRADE_VOL = 7
LONDON_LAT, LONDON_LON = 51.5074, -0.1278
# ────────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

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

class CombinedBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        
        # State tracking
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_PRODUCTS}
        
        # ETF tracking
        self._last_etf_trade_time = 0.0
        self._arb_count = 0
        self._theoretical_pnl = 0.0
        self._start_pnl: Optional[float] = None
        
        # Weather tracking
        self.spot_fv = None
        self.sum_fv = None
        self._last_wx_trade_times = {WX_SPOT: 0.0, WX_SUM: 0.0}
        self._last_wx_quote_time = 0.0
        
        # DYNAMIC DATE CALCULATION (Finds current weekend Sat 12pm -> Sun 12pm)
        now_lon = pd.Timestamp.now(tz="Europe/London")
        days_since_sat = (now_lon.dayofweek - 5) % 7
        if now_lon.dayofweek == 5 and now_lon.hour < 12:
            days_since_sat = 7 # It is Saturday morning, go to previous Saturday
        
        self.session_start = (now_lon - pd.Timedelta(days=days_since_sat)).replace(hour=12, minute=0, second=0, microsecond=0)
        self.session_end = self.session_start + pd.Timedelta(days=1)

        # Start pricing thread
        self._pricing_thread = Thread(target=self._update_fv_loop, daemon=True)
        self._pricing_thread.start()

    # ─── 1. Orderbook & Event Routing ─────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook) -> None:
        product = orderbook.product
        if product not in self._top:
            return

        # Update local cache of Top of Book
        top = TopOfBook()
        if orderbook.buy_orders:
            top.bid_px = orderbook.buy_orders[0].price
            top.bid_sz = int(orderbook.buy_orders[0].volume)
        if orderbook.sell_orders:
            top.ask_px = orderbook.sell_orders[0].price
            top.ask_sz = int(orderbook.sell_orders[0].volume)
        self._top[product] = top

        # Priority 1: Risk-Free ETF Arbitrage
        etf_fired = self._maybe_etf_arb()

        # Priority 2: Directional Weather Trading / Market Making
        if not etf_fired and product in [WX_SPOT, WX_SUM]:
            self._maybe_weather_arb(product)

    def on_trades(self, trade: Trade) -> None:
        direction = "BOT" if trade.buyer == self.username else "SLD"
        sign = 1 if direction == "SLD" else -1
        cost = sign * trade.volume * trade.price
        print(f"  [{ts()}] FILL {direction} {trade.volume:>3} {trade.product:<12} @ {trade.price:>7.0f}  cost={cost:>+10.0f}")

    # ─── 2. ETF Arbitrage Logic ───────────────────────────────────────────

    def _maybe_etf_arb(self) -> bool:
        now = time.monotonic()
        if now - self._last_etf_trade_time < MIN_COOLDOWN:
            return False

        E = self._top[ETF]
        A = self._top[COMPONENTS[0]]
        B = self._top[COMPONENTS[1]]
        C = self._top[COMPONENTS[2]]

        # If orderbook is completely missing a component, we cannot cross the spread
        if any(v is None for v in (E.bid_px, E.ask_px, A.bid_px, A.ask_px, B.bid_px, B.ask_px, C.bid_px, C.ask_px)):
            return False

        basket_ask = A.ask_px + B.ask_px + C.ask_px
        basket_bid = A.bid_px + B.bid_px + C.bid_px

        try: pos = self.get_positions()
        except Exception: return False

        pe = int(pos.get(ETF, 0))
        pa = int(pos.get(COMPONENTS[0], 0))
        pb = int(pos.get(COMPONENTS[1], 0))
        pc = int(pos.get(COMPONENTS[2], 0))
        p0 = (pe, pa, pb, pc)

        p1_after_1 = (pe - 1, pa + 1, pb + 1, pc + 1)
        p2_after_1 = (pe + 1, pa - 1, pb - 1, pc - 1)
        case1_is_unwind = self._total_abs(p1_after_1) < self._total_abs(p0)
        case2_is_unwind = self._total_abs(p2_after_1) < self._total_abs(p0)

        inventory_skew = (pe / POS_LIMIT) * ETF_MAX_SKEW
        required_edge1 = ETF_LIMIT_UNWIND_EDGE if self._near_limit(p0) else (ETF_UNWIND_EDGE if case1_is_unwind else ETF_MIN_EDGE - inventory_skew)
        required_edge2 = ETF_LIMIT_UNWIND_EDGE if self._near_limit(p0) else (ETF_UNWIND_EDGE if case2_is_unwind else ETF_MIN_EDGE + inventory_skew)

        # Case 1: ETF Rich
        edge1 = E.bid_px - basket_ask
        if edge1 > required_edge1:
            vol = min(E.bid_sz, A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL, POS_LIMIT + pe, POS_LIMIT - pa, POS_LIMIT - pb, POS_LIMIT - pc)
            if vol >= 1:
                self._fire_etf_arb("ETF RICH", edge1, vol, [
                    OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol),
                ])
                self._last_etf_trade_time = now
                return True

        # Case 2: ETF Cheap
        edge2 = basket_bid - E.ask_px
        if edge2 > required_edge2:
            vol = min(E.ask_sz, A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL, POS_LIMIT - pe, POS_LIMIT + pa, POS_LIMIT + pb, POS_LIMIT + pc)
            if vol >= 1:
                self._fire_etf_arb("ETF CHEAP", edge2, vol, [
                    OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                    OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                    OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol),
                ])
                self._last_etf_trade_time = now
                return True

        return False

    def _fire_etf_arb(self, label: str, edge_per_lot: float, vol: int, orders: list[OrderRequest]) -> None:
        self._arb_count += 1
        trade_theo_pnl = edge_per_lot * vol
        self._theoretical_pnl += trade_theo_pnl
        print(f"\n[{ts()}] ARB #{self._arb_count} {label} | vol={vol} | PnL=+{trade_theo_pnl:.0f}")
        self._send_ioc_batch(orders)

    @staticmethod
    def _total_abs(p: tuple[int, int, int, int]) -> int: return sum(abs(x) for x in p)

    @staticmethod
    def _near_limit(p: tuple[int, int, int, int]) -> bool: return any(abs(x) >= LIMIT_NEAR for x in p)

    # ─── 3. Weather Arbitrage & Maker Logic ───────────────────────────────

    def _fetch_weather_data(self):
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LONDON_LAT, "longitude": LONDON_LON,
            "minutely_15": "temperature_2m,relative_humidity_2m",
            "past_minutely_15": 96, "forecast_minutely_15": 96,
            "timezone": "Europe/London",
        })
        resp.raise_for_status()
        m = resp.json()["minutely_15"]
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temperature": m["temperature_2m"],
            "humidity": m["relative_humidity_2m"],
        })
        df['temp_F'] = (df['temperature'] * 9/5) + 32
        df['wx_metric'] = df['temp_F'] * df['humidity']
        return df

    def _update_fv_loop(self):
        print(f"[{ts()}] [PRICER] Calibrating window: {self.session_start} to {self.session_end}")
        while True:
            try:
                df = self._fetch_weather_data()
                session_df = df[(df['time'] > self.session_start) & (df['time'] <= self.session_end)]
                
                spot_row = session_df[session_df['time'] == self.session_end]
                if not spot_row.empty: self.spot_fv = spot_row['wx_metric'].iloc[0]
                if not session_df.empty: self.sum_fv = session_df['wx_metric'].sum() / 100.0

                print(f"[{ts()}] [PRICER] Updated FVs -> WX_SPOT: {self.spot_fv:.1f} | WX_SUM: {self.sum_fv:.1f}")
            except Exception as e:
                print(f"[{ts()}] [PRICER ERROR] Failed to fetch weather data: {e}")
            time.sleep(300)

    def _maybe_weather_arb(self, product: str) -> bool:
        now = time.monotonic()
        fv = self.spot_fv if product == WX_SPOT else self.sum_fv
        if fv is None: return False

        top = self._top[product]
        try: current_pos = int(self.get_positions().get(product, 0))
        except Exception: return False

        skew = (current_pos / POS_LIMIT) * WX_MAX_SKEW
        skewed_fv = fv - skew

        # 1. TAKER LOGIC (Cross the spread if edge is met)
        if now - self._last_wx_trade_times[product] >= WX_COOLDOWN:
            if top.ask_px and top.ask_px < (skewed_fv - WX_MIN_EDGE):
                vol = min(top.ask_sz, MAX_TRADE_VOL, POS_LIMIT - current_pos)
                if vol > 0:
                    print(f"\n[{ts()}] WEATHER TAKER BUY {product} | Ask {top.ask_px:.1f} < Skew FV {skewed_fv:.1f}")
                    self._send_ioc_batch([OrderRequest(product, top.ask_px, Side.BUY, vol)])
                    self._last_wx_trade_times[product] = now
                    return True

            if top.bid_px and top.bid_px > (skewed_fv + WX_MIN_EDGE):
                vol = min(top.bid_sz, MAX_TRADE_VOL, POS_LIMIT + current_pos)
                if vol > 0:
                    print(f"\n[{ts()}] WEATHER TAKER SELL {product} | Bid {top.bid_px:.1f} > Skew FV {skewed_fv:.1f}")
                    self._send_ioc_batch([OrderRequest(product, top.bid_px, Side.SELL, vol)])
                    self._last_wx_trade_times[product] = now
                    return True

        # 2. MAKER LOGIC (If no Taker trade happened, quote resting limit orders to provide liquidity)
        # We only update quotes every 5 seconds to avoid spamming the API
        if now - self._last_wx_quote_time > 5.0:
            self._update_weather_quotes(product, skewed_fv, current_pos)
            self._last_wx_quote_time = now

        return False

    def _update_weather_quotes(self, product: str, skewed_fv: float, current_pos: int):
        """Cancels old resting orders and places new ones around the Fair Value."""
        # Cancel existing resting orders for this product to prevent buildup
        existing_orders = self.get_orders(product)
        for order in existing_orders:
            try: self.cancel_order(order["id"])
            except Exception: pass

        new_orders = []
        bid_px = round(skewed_fv - WX_QUOTE_SPREAD)
        ask_px = round(skewed_fv + WX_QUOTE_SPREAD)

        if current_pos < POS_LIMIT: # Room to buy
            new_orders.append(OrderRequest(product, bid_px, Side.BUY, 2))
        if current_pos > -POS_LIMIT: # Room to sell
            new_orders.append(OrderRequest(product, ask_px, Side.SELL, 2))

        if new_orders:
            # We use standard send_order here, NOT IOC, because we want these to rest on the book.
            for o in new_orders:
                Thread(target=self.send_order, args=(o,)).start()

    # ─── 4. Execution & Main Loop ─────────────────────────────────────────

    def _send_ioc_batch(self, orders: list[OrderRequest]) -> None:
        results = []
        def _send(o):
            resp = self.send_order(o)
            if resp and resp.volume > 0:
                try: self.cancel_order(resp.id)
                except Exception: pass
            results.append((o, resp))

        threads = [Thread(target=_send, args=(o,)) for o in orders]
        for t in threads: t.start()
        for t in threads: t.join()

    def run_forever(self) -> None:
        self._start_pnl = self.get_pnl().get("totalProfit", 0.0)
        print(f"[{ts()}] ═══ Combined Arb & Weather Bot Started ═══")
        self.start()

        def _shutdown(sig, frame):
            print(f"\n[{ts()}] ═══ Shutting down ═══")
            self.cancel_all_orders()
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while True:
            time.sleep(30)
            try:
                pnl = self.get_pnl().get("totalProfit", 0.0)
                pos = self.get_positions()
                pos_str = " ".join(f"{k}={v}" for k, v in pos.items() if v != 0)
                print(f"[{ts()}] HEARTBEAT | arbs={self._arb_count} | session={pnl - self._start_pnl:+.0f} | pos=[{pos_str}]")
            except Exception:
                pass

if __name__ == "__main__":
    bot = CombinedBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()