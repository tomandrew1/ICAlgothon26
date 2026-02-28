#!/usr/bin/env python3 -u
"""ETF cross-arbitrage bot with per-trade PnL tracking.

Trades LON_ETF against its components (TIDE_SPOT + WX_SPOT + LHR_COUNT).
Runs indefinitely, prints every trade and theoretical PnL.

Usage:
    python -u etf_arb.py
"""

from __future__ import annotations

import os
import signal
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from threading import Thread

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME = "timgu"
PASSWORD = "1!Qwerty"

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ALL_PRODUCTS = [ETF] + COMPONENTS

POS_LIMIT = 100
MIN_EDGE = 2.0
MIN_COOLDOWN = 0.5
MAX_TRADE_VOL = 5


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


class EtfArbBot(BaseBot):

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_PRODUCTS}
        self._last_trade_time = 0.0
        self._arb_count = 0
        self._theoretical_pnl = 0.0
        self._start_pnl: Optional[float] = None

    # ── callbacks ────────────────────────────────────────────────────────

    def on_orderbook(self, orderbook: OrderBook) -> None:
        if orderbook.product not in self._top:
            return

        top = TopOfBook()
        if orderbook.buy_orders:
            top.bid_px = orderbook.buy_orders[0].price
            top.bid_sz = int(orderbook.buy_orders[0].volume)
        if orderbook.sell_orders:
            top.ask_px = orderbook.sell_orders[0].price
            top.ask_sz = int(orderbook.sell_orders[0].volume)

        self._top[orderbook.product] = top
        self._maybe_arb()

    def on_trades(self, trade: Trade) -> None:
        direction = "BOT" if trade.buyer == self.username else "SLD"
        sign = 1 if direction == "SLD" else -1
        cost = sign * trade.volume * trade.price
        print(
            f"  [{ts()}] FILL {direction} {trade.volume:>3} {trade.product:<12} "
            f"@ {trade.price:>7.0f}  cost={cost:>+10.0f}"
        )

    # ── arbitrage logic ──────────────────────────────────────────────────

    def _maybe_arb(self) -> None:
        now = time.monotonic()
        if now - self._last_trade_time < MIN_COOLDOWN:
            return

        E = self._top[ETF]
        A = self._top[COMPONENTS[0]]
        B = self._top[COMPONENTS[1]]
        C = self._top[COMPONENTS[2]]

        if any(
            v is None
            for v in (E.bid_px, E.ask_px, A.bid_px, A.ask_px, B.bid_px, B.ask_px, C.bid_px, C.ask_px)
        ):
            return

        basket_ask = A.ask_px + B.ask_px + C.ask_px
        basket_bid = A.bid_px + B.bid_px + C.bid_px

        try:
            pos = self.get_positions()
        except Exception as exc:
            print(f"  [{ts()}] WARN  position fetch failed: {exc}")
            return

        pe = pos.get(ETF, 0)
        pa = pos.get(COMPONENTS[0], 0)
        pb = pos.get(COMPONENTS[1], 0)
        pc = pos.get(COMPONENTS[2], 0)

        # Case 1: ETF rich → sell ETF, buy components
        edge1 = E.bid_px - basket_ask
        if edge1 > MIN_EDGE:
            vol = min(
                min(E.bid_sz, MAX_TRADE_VOL),
                min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL),
                POS_LIMIT + pe,
                POS_LIMIT - pa,
                POS_LIMIT - pb,
                POS_LIMIT - pc,
            )
            if vol >= 1:
                self._fire_arb(
                    label="ETF RICH  → sell ETF, buy basket",
                    edge_per_lot=edge1,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol),
                    ],
                    prices={"ETF_sell": E.bid_px, "A_buy": A.ask_px, "B_buy": B.ask_px, "C_buy": C.ask_px},
                    positions=(pe, pa, pb, pc),
                )
                self._last_trade_time = now
                return

        # Case 2: ETF cheap → buy ETF, sell components
        edge2 = basket_bid - E.ask_px
        if edge2 > MIN_EDGE:
            vol = min(
                min(E.ask_sz, MAX_TRADE_VOL),
                min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL),
                POS_LIMIT - pe,
                POS_LIMIT + pa,
                POS_LIMIT + pb,
                POS_LIMIT + pc,
            )
            if vol >= 1:
                self._fire_arb(
                    label="ETF CHEAP → buy ETF, sell basket",
                    edge_per_lot=edge2,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol),
                    ],
                    prices={"ETF_buy": E.ask_px, "A_sell": A.bid_px, "B_sell": B.bid_px, "C_sell": C.bid_px},
                    positions=(pe, pa, pb, pc),
                )
                self._last_trade_time = now
                return

    def _send_ioc(self, order: OrderRequest) -> tuple[OrderRequest, OrderResponse | None]:
        """Send an order and immediately cancel any unfilled remainder (IOC simulation)."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            try:
                self.cancel_order(resp.id)
            except Exception:
                pass
        return order, resp

    def _fire_arb(
        self,
        label: str,
        edge_per_lot: float,
        vol: int,
        orders: list[OrderRequest],
        prices: dict[str, float],
        positions: tuple[int, int, int, int],
    ) -> None:
        self._arb_count += 1
        trade_theo_pnl = edge_per_lot * vol
        self._theoretical_pnl += trade_theo_pnl

        pe, pa, pb, pc = positions
        print(f"\n{'─'*72}")
        print(f"[{ts()}] ARB #{self._arb_count}  {label}")
        print(f"  edge/lot = {edge_per_lot:.1f}  |  vol = {vol}  |  trade PnL = +{trade_theo_pnl:.0f}  |  cum PnL = +{self._theoretical_pnl:.0f}")
        print(f"  pos: ETF={pe} A={pa} B={pb} C={pc}  |  books: ETF {self._top[ETF]}  A {self._top[COMPONENTS[0]]}  B {self._top[COMPONENTS[1]]}  C {self._top[COMPONENTS[2]]}")
        print(f"  sending {len(orders)} legs (IOC):")
        for o in orders:
            print(f"    {str(o.side):>4} {o.volume:>3} {o.product:<12} @ {o.price:>7.0f}")

        results: list[tuple[OrderRequest, OrderResponse | None]] = []
        threads = [Thread(target=lambda o: results.append(self._send_ioc(o)), args=(o,)) for o in orders]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        filled_legs = [(req, resp) for req, resp in results if resp and resp.filled > 0]
        missed_legs = [(req, resp) for req, resp in results if resp is None or resp.filled == 0]

        fill_summary = "  ".join(
            f"{req.product}:{resp.filled if resp else 'X'}/{req.volume}"
            for req, resp in results
        )

        if len(filled_legs) == len(orders):
            print(f"  FILLED  all {len(orders)} legs  ({fill_summary})")
        else:
            print(f"  PARTIAL {len(filled_legs)}/{len(orders)} legs  ({fill_summary})")

        print(f"{'─'*72}")

    # ── run forever ──────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Start the SSE stream and block the main thread indefinitely."""
        pnl_data = self.get_pnl()
        self._start_pnl = pnl_data.get("totalProfit", 0.0)

        print(f"[{ts()}] ═══ ETF Arbitrage Bot Started ═══")
        print(f"  user:       {self.username}")
        print(f"  ETF:        {ETF}  =  {' + '.join(COMPONENTS)}")
        print(f"  params:     MIN_EDGE={MIN_EDGE}  MAX_VOL={MAX_TRADE_VOL}  POS_LIMIT={POS_LIMIT}  COOLDOWN={MIN_COOLDOWN}s")
        print(f"  start PnL:  {self._start_pnl:.0f}")

        pos = self.get_positions()
        pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items())) if pos else "(flat)"
        print(f"  positions:  {pos_str}")
        print(f"  Ctrl+C to stop.\n")

        self.start()

        def _shutdown(sig, frame):
            print(f"\n[{ts()}] ═══ Shutting down ═══")
            self.cancel_all_orders()
            self.stop()
            final = self.get_pnl()
            final_pnl = final.get("totalProfit", 0.0)
            session_pnl = final_pnl - self._start_pnl
            print(f"  final PnL:       {final_pnl:.0f}")
            print(f"  session PnL:     {session_pnl:+.0f}")
            print(f"  theoretical PnL: +{self._theoretical_pnl:.0f}")
            print(f"  arb trades:      {self._arb_count}")
            pos = self.get_positions()
            print(f"  positions:       {pos}")
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while True:
            time.sleep(30)
            try:
                pnl_data = self.get_pnl()
                actual_pnl = pnl_data.get("totalProfit", 0.0)
                session_pnl = actual_pnl - self._start_pnl
                pos = self.get_positions()
                pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items()))
                print(
                    f"[{ts()}] HEARTBEAT  arbs={self._arb_count}  "
                    f"session={session_pnl:+.0f}  theo={self._theoretical_pnl:+.0f}  "
                    f"actual={actual_pnl:.0f}  pos=[{pos_str}]"
                )
            except Exception as exc:
                print(f"[{ts()}] HEARTBEAT error: {exc}")


if __name__ == "__main__":
    bot = EtfArbBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()
