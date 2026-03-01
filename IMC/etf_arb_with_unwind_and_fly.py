#!/usr/bin/env python3 -u
"""ETF cross-arbitrage bot with per-trade PnL tracking + LHR_FLY option structure.

Trades LON_ETF against its components (TIDE_SPOT + WX_SPOT + LHR_COUNT).
Also trades LHR_FLY (option structure: +2P6200 +2C6200 -2C6600 +3C7000)
using LHR_COUNT mid as fair value, subject to spread quality filter.

Key behavior:
- Normal arbs require MIN_EDGE (with mild ETF-position skew).
- If an arb would REDUCE total inventory risk (sum of abs positions),
  we accept a MUCH smaller edge (UNWIND_EDGE), and even smaller when near limits.
- LHR_FLY: compute payoff(LHR_mid), market-take if mispriced by FLY_MIN_EDGE,
  provided LHR spread is <= FLY_MAX_SPREAD_PCT.

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
from threading import Thread
from typing import Optional

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME = "Y"
PASSWORD = "y"

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
LHR_INDEX = "LHR_COUNT"   # underlying for the fly
LHR_FLY   = "LHR_FLY"
ALL_PRODUCTS = [ETF] + COMPONENTS + [LHR_FLY]

# ── ETF arb parameters ───────────────────────────────────────────────────
POS_LIMIT     = 100
MIN_EDGE      = 2.0
MIN_COOLDOWN  = 0.2
MAX_TRADE_VOL = 7
MAX_SKEW      = 1.5   # max edge reduction when ETF pos is at POS_LIMIT

# ── unwind behavior ──────────────────────────────────────────────────────
UNWIND_EDGE       = 1.0   # accept low edge when trade reduces inventory risk
LIMIT_UNWIND_EDGE = 0.0   # accept almost no edge when near limits
LIMIT_NEAR        = 80    # treat |pos| >= LIMIT_NEAR as "near limit"

# ── LHR_FLY parameters ──────────────────────────────────────────────────
# Payoff: +2 P6200  +2 C6200  -2 C6600  +3 C7000
# Strikes
FLY_STRIKES = {
    "P6200": (6200, "put",  +2),
    "C6200": (6200, "call", +2),
    "C6600": (6600, "call", -2),
    "C7000": (7000, "call", +3),
}
FLY_MIN_EDGE        = 2.0   # min edge (fair - market) to trade
FLY_MAX_VOL         = 3     # max lots per trade
FLY_POS_LIMIT       = 30    # separate pos limit for LHR_FLY
FLY_MAX_SPREAD_PCT  = 0.1  # reject LHR quote if spread > 5%
FLY_COOLDOWN        = 0.3   # separate cooldown for fly trades (s)
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


class EtfArbBot(BaseBot):

    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_PRODUCTS}
        self._last_trade_time  = 0.0
        self._last_fly_time    = 0.0
        self._arb_count        = 0
        self._fly_count        = 0
        self._theoretical_pnl  = 0.0
        self._fly_theo_pnl     = 0.0
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
        self._maybe_fly()

    def on_trades(self, trade: Trade) -> None:
        direction = "BOT" if trade.buyer == self.username else "SLD"
        sign = 1 if direction == "SLD" else -1
        cost = sign * trade.volume * trade.price
        print(
            f"  [{ts()}] FILL {direction} {trade.volume:>3} {trade.product:<12} "
            f"@ {trade.price:>7.0f}  cost={cost:>+10.0f}"
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _total_abs(p: tuple[int, int, int, int]) -> int:
        return sum(abs(x) for x in p)

    @staticmethod
    def _near_limit(p: tuple[int, int, int, int]) -> bool:
        return any(abs(x) >= LIMIT_NEAR for x in p)

    # ── LHR_FLY fair value ───────────────────────────────────────────────

    @staticmethod
    def _fly_payoff(s: float) -> float:
        """
        Payoff of +2P6200 +2C6200 -2C6600 +3C7000 at underlying level s.

        By region:
          s < 6200            :  2*(6200-s)          [downside profit]
          6200 <= s < 6600    :  2*(s-6200)           [long the move up]
          6600 <= s < 7000    :  800                  [flat plateau]
          s >= 7000           :  800 + 3*(s-7000)     [re-accelerates]
        """
        return (
            2 * max(6200 - s, 0.0)
            + 2 * max(s - 6200, 0.0)
            - 2 * max(s - 6600, 0.0)
            + 3 * max(s - 7000, 0.0)
        )

    # ── LHR_FLY trading logic ────────────────────────────────────────────

    def _maybe_fly(self) -> None:
        now = time.monotonic()
        if now - self._last_fly_time < FLY_COOLDOWN:
            return

        fly = self._top[LHR_FLY]
        lhr = self._top[LHR_INDEX]

        # Need full quotes on both
        if any(v is None for v in (fly.bid_px, fly.ask_px, lhr.bid_px, lhr.ask_px)):
            return

        # Reject if LHR spread is too wide — mid would be unreliable
        lhr_mid = (lhr.bid_px + lhr.ask_px) / 2.0
        lhr_spread_pct = (lhr.ask_px - lhr.bid_px) / lhr_mid
        if lhr_spread_pct > FLY_MAX_SPREAD_PCT:
            return

        fair = self._fly_payoff(lhr_mid)

        try:
            pos = self.get_positions()
        except Exception as exc:
            print(f"  [{ts()}] WARN  fly position fetch failed: {exc}")
            return

        pos_fly = int(pos.get(LHR_FLY, 0))

        # ── BUY fly: market ask is below fair value ──────────────────────
        edge_buy = fair - fly.ask_px
        if edge_buy > FLY_MIN_EDGE and pos_fly < FLY_POS_LIMIT:
            vol = min(fly.ask_sz, FLY_MAX_VOL, FLY_POS_LIMIT - pos_fly)
            if vol >= 1:
                self._fire_fly(
                    side=Side.BUY,
                    price=fly.ask_px,
                    vol=vol,
                    fair=fair,
                    edge=edge_buy,
                    lhr_mid=lhr_mid,
                    lhr_spread_pct=lhr_spread_pct,
                    pos_fly=pos_fly,
                )
                self._last_fly_time = now
                return

        # ── SELL fly: market bid is above fair value ─────────────────────
        edge_sell = fly.bid_px - fair
        if edge_sell > FLY_MIN_EDGE and pos_fly > -FLY_POS_LIMIT:
            vol = min(fly.bid_sz, FLY_MAX_VOL, FLY_POS_LIMIT + pos_fly)
            if vol >= 1:
                self._fire_fly(
                    side=Side.SELL,
                    price=fly.bid_px,
                    vol=vol,
                    fair=fair,
                    edge=edge_sell,
                    lhr_mid=lhr_mid,
                    lhr_spread_pct=lhr_spread_pct,
                    pos_fly=pos_fly,
                )
                self._last_fly_time = now
                return

    def _fire_fly(
        self,
        side: Side,
        price: float,
        vol: int,
        fair: float,
        edge: float,
        lhr_mid: float,
        lhr_spread_pct: float,
        pos_fly: int,
    ) -> None:
        self._fly_count += 1
        trade_theo = edge * vol
        self._fly_theo_pnl += trade_theo

        direction = "BUY " if side == Side.BUY else "SELL"
        print(f"\n{'═'*72}")
        print(f"[{ts()}] FLY #{self._fly_count}  {direction} LHR_FLY")
        print(
            f"  LHR mid = {lhr_mid:.1f}  spread = {lhr_spread_pct*100:.2f}%  "
            f"fair = {fair:.1f}  market = {price:.0f}  edge = {edge:.1f}"
        )
        print(
            f"  vol = {vol}  trade PnL = +{trade_theo:.0f}  "
            f"cum fly PnL = +{self._fly_theo_pnl:.0f}  pos_fly = {pos_fly}"
        )
        print(f"  sending: {direction} {vol} LHR_FLY @ {price:.0f}")

        order = OrderRequest(LHR_FLY, price, side, vol)
        _, resp = self._send_ioc(order)

        if resp and resp.filled > 0:
            print(f"  FILLED {resp.filled}/{vol}")
        else:
            print(f"  NO FILL (resp={resp})")
        print(f"{'═'*72}")

    # ── ETF arbitrage logic ──────────────────────────────────────────────

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

        pe = int(pos.get(ETF, 0))
        pa = int(pos.get(COMPONENTS[0], 0))
        pb = int(pos.get(COMPONENTS[1], 0))
        pc = int(pos.get(COMPONENTS[2], 0))

        p0 = (pe, pa, pb, pc)

        # Determine if each case is an "unwind" (reduces total inventory risk)
        # Case 1: SELL ETF, BUY A,B,C  => (-1, +1, +1, +1)
        # Case 2: BUY ETF, SELL A,B,C  => (+1, -1, -1, -1)
        p1_after_1 = (pe - 1, pa + 1, pb + 1, pc + 1)
        p2_after_1 = (pe + 1, pa - 1, pb - 1, pc - 1)

        case1_is_unwind = self._total_abs(p1_after_1) < self._total_abs(p0)
        case2_is_unwind = self._total_abs(p2_after_1) < self._total_abs(p0)

        # Normal skew (ETF-only, mild)
        inventory_skew = (pe / POS_LIMIT) * MAX_SKEW
        required_edge1_normal = MIN_EDGE - inventory_skew
        required_edge2_normal = MIN_EDGE + inventory_skew

        # Unwind thresholds (much lower)
        unwind_thresh = LIMIT_UNWIND_EDGE if self._near_limit(p0) else UNWIND_EDGE

        required_edge1 = unwind_thresh if case1_is_unwind else required_edge1_normal
        required_edge2 = unwind_thresh if case2_is_unwind else required_edge2_normal

        # Case 1: ETF rich → sell ETF, buy components
        edge1 = E.bid_px - basket_ask
        if edge1 > required_edge1:
            vol = min(
                min(E.bid_sz, MAX_TRADE_VOL),
                min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL),
                POS_LIMIT + pe,   # SELL ETF headroom to -POS_LIMIT
                POS_LIMIT - pa,   # BUY component headroom to +POS_LIMIT
                POS_LIMIT - pb,
                POS_LIMIT - pc,
            )
            if vol >= 1:
                why = "UNWIND" if case1_is_unwind else "NORMAL"
                self._fire_arb(
                    label=(
                        f"ETF RICH  → sell ETF, buy basket [{why}] "
                        f"(Req Edge: {required_edge1:.1f})"
                    ),
                    edge_per_lot=edge1,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF,          E.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY,  vol),
                        OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY,  vol),
                        OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY,  vol),
                    ],
                    prices={"ETF_sell": E.bid_px, "A_buy": A.ask_px, "B_buy": B.ask_px, "C_buy": C.ask_px},
                    positions=p0,
                )
                self._last_trade_time = now
                return

        # Case 2: ETF cheap → buy ETF, sell components
        edge2 = basket_bid - E.ask_px
        if edge2 > required_edge2:
            vol = min(
                min(E.ask_sz, MAX_TRADE_VOL),
                min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL),
                POS_LIMIT - pe,   # BUY ETF headroom to +POS_LIMIT
                POS_LIMIT + pa,   # SELL component headroom to -POS_LIMIT
                POS_LIMIT + pb,
                POS_LIMIT + pc,
            )
            if vol >= 1:
                why = "UNWIND" if case2_is_unwind else "NORMAL"
                self._fire_arb(
                    label=(
                        f"ETF CHEAP → buy ETF, sell basket [{why}] "
                        f"(Req Edge: {required_edge2:.1f})"
                    ),
                    edge_per_lot=edge2,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF,          E.ask_px, Side.BUY,  vol),
                        OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol),
                    ],
                    prices={"ETF_buy": E.ask_px, "A_sell": A.bid_px, "B_sell": B.bid_px, "C_sell": C.bid_px},
                    positions=p0,
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
        print(
            f"  edge/lot = {edge_per_lot:.1f}  |  vol = {vol}  |  "
            f"trade PnL = +{trade_theo_pnl:.0f}  |  cum PnL = +{self._theoretical_pnl:.0f}"
        )
        print(
            f"  pos: ETF={pe} A={pa} B={pb} C={pc}  |  "
            f"books: ETF {self._top[ETF]}  A {self._top[COMPONENTS[0]]}  "
            f"B {self._top[COMPONENTS[1]]}  C {self._top[COMPONENTS[2]]}"
        )
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
        print(
            f"  params:     MIN_EDGE={MIN_EDGE}  UNWIND_EDGE={UNWIND_EDGE}  "
            f"MAX_VOL={MAX_TRADE_VOL}  POS_LIMIT={POS_LIMIT}  COOLDOWN={MIN_COOLDOWN}s"
        )
        print(
            f"  fly params: FLY_MIN_EDGE={FLY_MIN_EDGE}  FLY_MAX_VOL={FLY_MAX_VOL}  "
            f"FLY_POS_LIMIT={FLY_POS_LIMIT}  MAX_SPREAD={FLY_MAX_SPREAD_PCT*100:.0f}%  "
            f"COOLDOWN={FLY_COOLDOWN}s"
        )
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
            print(f"  theoretical PnL (arb): +{self._theoretical_pnl:.0f}")
            print(f"  theoretical PnL (fly): +{self._fly_theo_pnl:.0f}")
            print(f"  arb trades:      {self._arb_count}")
            print(f"  fly trades:      {self._fly_count}")
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
                    f"[{ts()}] HEARTBEAT  arbs={self._arb_count}  flys={self._fly_count}  "
                    f"session={session_pnl:+.0f}  "
                    f"theo_arb={self._theoretical_pnl:+.0f}  theo_fly={self._fly_theo_pnl:+.0f}  "
                    f"actual={actual_pnl:.0f}  pos=[{pos_str}]"
                )
            except Exception as exc:
                print(f"[{ts()}] HEARTBEAT error: {exc}")


if __name__ == "__main__":
    bot = EtfArbBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()