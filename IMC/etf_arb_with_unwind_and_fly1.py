#!/usr/bin/env python3 -u
"""ETF cross-arbitrage bot with per-trade PnL tracking + LHR_FLY option structure.

Trades LON_ETF against its components (TIDE_SPOT + WX_SPOT + LHR_COUNT).
Also trades LHR_FLY (option structure: +2P6200 +2C6200 -2C6600 +3C7000)
using LHR_COUNT mid as fair value, subject to spread quality filter.

Key behavior:
- Normal arbs require MIN_EDGE (with mild ETF-position skew).
- If an arb would REDUCE total inventory risk (sum of abs positions),
  we accept a MUCH smaller edge (UNWIND_EDGE), and even smaller when near limits.
- LHR_FLY aggressive: market-take if mispriced by FLY_MIN_EDGE vs payoff(LHR_mid).
- LHR_FLY passive: if no aggressive edge, post resting limit orders at
  fair-FLY_QUOTE_OFFSET (bid) and fair+FLY_QUOTE_OFFSET (ask), size FLY_QUOTE_VOL.
  Requotes if resting price drifts more than FLY_REQUOTE_THRESH from current fair.

Usage:
    python -u etf_arb.py
"""

from __future__ import annotations

import os
import signal
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com"
USERNAME = "RATT"
PASSWORD = "ratt67"

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
LHR_INDEX = "LHR_COUNT"
LHR_FLY   = "LHR_FLY"
ALL_PRODUCTS = [ETF] + COMPONENTS + [LHR_FLY]

# ── ETF arb parameters ───────────────────────────────────────────────────
POS_LIMIT     = 100
MIN_EDGE      = 3.0
MIN_COOLDOWN  = 0.2
MAX_TRADE_VOL = 7
MAX_SKEW      = 1.5

# ── unwind behavior ──────────────────────────────────────────────────────
UNWIND_EDGE       = 2.0
LIMIT_UNWIND_EDGE = 1.0
LIMIT_NEAR        = 90

# ── LHR_FLY aggressive parameters ───────────────────────────────────────
FLY_MIN_EDGE       = 2.0
FLY_MAX_VOL        = 3
FLY_POS_LIMIT      = 30
FLY_MAX_SPREAD_PCT = 0.1
FLY_COOLDOWN       = 0.2

# ── LHR_FLY passive quoting ──────────────────────────────────────────────
FLY_QUOTE_VOL      = 3
FLY_QUOTE_OFFSET   = 1
FLY_REQUOTE_THRESH = 2
FLY_QUOTE_COOLDOWN = 0.3

# ── Diagnostics ──────────────────────────────────────────────────────────
DIAG_INTERVAL  = 60    # print full diagnostics every N seconds
EDGE_BUCKET_SZ = 1     # bucket edges into groups of this size (e.g. 1 = "2-3", "3-4" ...)
# ────────────────────────────────────────────────────────────────────────


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

def edge_bucket(edge: float) -> str:
    lo = int(edge // EDGE_BUCKET_SZ) * EDGE_BUCKET_SZ
    return f"{lo}-{lo + EDGE_BUCKET_SZ}"


@dataclass
class TradeStats:
    """Accumulates per-category trade statistics."""
    count:     int   = 0
    vol:       int   = 0
    theo_pnl:  float = 0.0
    edge_sum:  float = 0.0
    # edge bucket histogram: {"2-3": count, ...}
    edge_hist: dict  = field(default_factory=lambda: defaultdict(int))

    def record(self, edge: float, vol: int) -> None:
        self.count    += 1
        self.vol      += vol
        self.theo_pnl += edge * vol
        self.edge_sum += edge
        self.edge_hist[edge_bucket(edge)] += 1

    @property
    def avg_edge(self) -> float:
        return self.edge_sum / self.count if self.count else 0.0

    def summary(self, label: str) -> str:
        if self.count == 0:
            return f"  {label:<28}  (no trades)"
        hist_str = "  ".join(f"{b}:{n}" for b, n in sorted(self.edge_hist.items()))
        return (
            f"  {label:<28}  trades={self.count:>4}  vol={self.vol:>5}  "
            f"theo={self.theo_pnl:>+9.0f}  avg_edge={self.avg_edge:>5.2f}  "
            f"edge_dist=[{hist_str}]"
        )


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
        self._last_quote_time  = 0.0
        self._last_diag_time   = 0.0
        self._start_time       = time.monotonic()
        self._arb_count        = 0
        self._fly_count        = 0
        self._theoretical_pnl  = 0.0
        self._fly_theo_pnl     = 0.0
        self._start_pnl: Optional[float] = None

        # Resting fly quotes
        self._fly_quotes: dict[str, Optional[tuple[str, float]]] = {
            "bid": None, "ask": None,
        }

        # ── Per-category stats ───────────────────────────────────────────
        self._stats: dict[str, TradeStats] = {
            "arb_normal_sell_etf":  TradeStats(),
            "arb_normal_buy_etf":   TradeStats(),
            "arb_unwind_sell_etf":  TradeStats(),
            "arb_unwind_buy_etf":   TradeStats(),
            "fly_aggressive_buy":   TradeStats(),
            "fly_aggressive_sell":  TradeStats(),
            "fly_passive_buy":      TradeStats(),
            "fly_passive_sell":     TradeStats(),
        }

        # Fill tracking: product -> list of (side, price, vol)
        self._fills: list[dict] = []

        # Quote fill counts
        self._quote_fills   = {"bid": 0, "ask": 0}
        self._requote_count = 0
        self._skipped_arb_cooldown  = 0
        self._skipped_fly_spread    = 0

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
        self._maybe_print_diag()

    def on_trades(self, trade: Trade) -> None:
        direction = "BOT" if trade.buyer == self.username else "SLD"
        sign = 1 if direction == "SLD" else -1
        cost = sign * trade.volume * trade.price
        print(
            f"  [{ts()}] FILL {direction} {trade.volume:>3} {trade.product:<12} "
            f"@ {trade.price:>7.0f}  cost={cost:>+10.0f}"
        )
        self._fills.append({
            "time": ts(), "product": trade.product,
            "side": direction, "price": trade.price, "vol": trade.volume,
        })
        if trade.product == LHR_FLY:
            for side in ("bid", "ask"):
                if self._fly_quotes[side] is not None:
                    self._quote_fills[side] += 1
                self._fly_quotes[side] = None

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
          s < 6200         :  2*(6200-s)        [downside profit]
          6200 <= s < 6600 :  2*(s-6200)        [long the move up]
          6600 <= s < 7000 :  800               [flat plateau]
          s >= 7000        :  800 + 3*(s-7000)  [re-accelerates]
        """
        return (
            2 * max(6200 - s, 0.0)
            + 2 * max(s - 6200, 0.0)
            - 2 * max(s - 6600, 0.0)
            + 3 * max(s - 7000, 0.0)
        )

    # ── Diagnostics ──────────────────────────────────────────────────────

    def _maybe_print_diag(self) -> None:
        now = time.monotonic()
        if now - self._last_diag_time < DIAG_INTERVAL:
            return
        self._last_diag_time = now
        self._print_diag()

    def _print_diag(self) -> None:
        elapsed = time.monotonic() - self._start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        try:
            pos = self.get_positions()
            pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items()))
        except Exception:
            pos_str = "?"

        fly = self._top[LHR_FLY]
        lhr = self._top[LHR_INDEX]
        lhr_mid = ((lhr.bid_px + lhr.ask_px) / 2.0) if (lhr.bid_px and lhr.ask_px) else None
        fair    = self._fly_payoff(lhr_mid) if lhr_mid else None
        fly_mid = ((fly.bid_px + fly.ask_px) / 2.0) if (fly.bid_px and fly.ask_px) else None

        bid_q = self._fly_quotes["bid"]
        ask_q = self._fly_quotes["ask"]

        total_arb_trades = sum(
            self._stats[k].count
            for k in ("arb_normal_sell_etf","arb_normal_buy_etf",
                      "arb_unwind_sell_etf","arb_unwind_buy_etf")
        )
        total_fly_trades = sum(
            self._stats[k].count
            for k in ("fly_aggressive_buy","fly_aggressive_sell",
                      "fly_passive_buy","fly_passive_sell")
        )

        print(f"\n{'█'*72}")
        print(f"[{ts()}] DIAGNOSTICS  uptime={mins}m{secs:02d}s")
        print(f"{'─'*72}")

        # ── Market snapshot ──────────────────────────────────────────────
        print(f"  MARKET SNAPSHOT")
        print(f"    ETF     {self._top[ETF]}")
        for c in COMPONENTS:
            print(f"    {c:<12} {self._top[c]}")
        if lhr_mid is not None and fair is not None and fly_mid is not None:
            print(
                f"    LHR_FLY {fly}  |  LHR_mid={lhr_mid:.1f}  "
                f"fair={fair:.1f}  fly_mid={fly_mid:.1f}  "
                f"fair_vs_mid={fly_mid - fair:+.1f}"
            )
        print(f"    resting quotes: bid@{bid_q[1]:.0f}" if bid_q else "    resting quotes: bid=--", end="")
        print(f"  ask@{ask_q[1]:.0f}" if ask_q else "  ask=--")

        # ── Positions ────────────────────────────────────────────────────
        print(f"{'─'*72}")
        print(f"  POSITIONS:  {pos_str}")

        # ── Trade stats ──────────────────────────────────────────────────
        print(f"{'─'*72}")
        print(f"  TRADE STATS  (total arb={total_arb_trades}  fly={total_fly_trades})")
        for key, stat in self._stats.items():
            print(stat.summary(key))

        # ── Quote stats ──────────────────────────────────────────────────
        print(f"{'─'*72}")
        print(
            f"  QUOTE STATS  "
            f"bid_fills={self._quote_fills['bid']}  "
            f"ask_fills={self._quote_fills['ask']}  "
            f"requotes={self._requote_count}  "
            f"skipped_fly_spread={self._skipped_fly_spread}"
        )

        # ── PnL summary ──────────────────────────────────────────────────
        print(f"{'─'*72}")
        total_theo = self._theoretical_pnl + self._fly_theo_pnl
        print(
            f"  THEO PnL  arb={self._theoretical_pnl:+.0f}  "
            f"fly={self._fly_theo_pnl:+.0f}  total={total_theo:+.0f}"
        )
        print(f"{'█'*72}\n")

    # ── LHR_FLY: main entry point ────────────────────────────────────────

    def _maybe_fly(self) -> None:
        now = time.monotonic()

        lhr = self._top[LHR_INDEX]
        fly = self._top[LHR_FLY]

        if any(v is None for v in (fly.bid_px, fly.ask_px, lhr.bid_px, lhr.ask_px)):
            return

        lhr_mid = (lhr.bid_px + lhr.ask_px) / 2.0
        lhr_spread_pct = (lhr.ask_px - lhr.bid_px) / lhr_mid
        if lhr_spread_pct > FLY_MAX_SPREAD_PCT:
            self._skipped_fly_spread += 1
            return

        fair = self._fly_payoff(lhr_mid)

        try:
            pos = self.get_positions()
        except Exception as exc:
            print(f"  [{ts()}] WARN  fly position fetch failed: {exc}")
            return

        pos_fly = int(pos.get(LHR_FLY, 0))

        # ── 1. Aggressive take ───────────────────────────────────────────
        if now - self._last_fly_time >= FLY_COOLDOWN:
            edge_buy  = fair - fly.ask_px
            edge_sell = fly.bid_px - fair

            if edge_buy > FLY_MIN_EDGE and pos_fly < FLY_POS_LIMIT:
                vol = min(fly.ask_sz, FLY_MAX_VOL, FLY_POS_LIMIT - pos_fly)
                if vol >= 1:
                    self._cancel_fly_quote("ask")
                    self._fire_fly(Side.BUY, fly.ask_px, vol, fair, edge_buy,
                                   lhr_mid, lhr_spread_pct, pos_fly, passive=False)
                    self._last_fly_time = now
                    return

            if edge_sell > FLY_MIN_EDGE and pos_fly > -FLY_POS_LIMIT:
                vol = min(fly.bid_sz, FLY_MAX_VOL, FLY_POS_LIMIT + pos_fly)
                if vol >= 1:
                    self._cancel_fly_quote("bid")
                    self._fire_fly(Side.SELL, fly.bid_px, vol, fair, edge_sell,
                                   lhr_mid, lhr_spread_pct, pos_fly, passive=False)
                    self._last_fly_time = now
                    return

        # ── 2. Passive quoting ───────────────────────────────────────────
        self._manage_fly_quotes(now, fair, pos_fly)

    # ── LHR_FLY: passive quote management ───────────────────────────────

    def _manage_fly_quotes(self, now: float, fair: float, pos_fly: int) -> None:
        if now - self._last_quote_time < FLY_QUOTE_COOLDOWN:
            return

        desired_bid = round(fair - FLY_QUOTE_OFFSET)
        desired_ask = round(fair + FLY_QUOTE_OFFSET)

        if pos_fly < FLY_POS_LIMIT:
            q = self._fly_quotes["bid"]
            if q is None:
                self._post_fly_quote("bid", desired_bid)
            elif abs(q[1] - desired_bid) > FLY_REQUOTE_THRESH:
                self._cancel_fly_quote("bid")
                self._post_fly_quote("bid", desired_bid)
                self._requote_count += 1
        else:
            self._cancel_fly_quote("bid")

        if pos_fly > -FLY_POS_LIMIT:
            q = self._fly_quotes["ask"]
            if q is None:
                self._post_fly_quote("ask", desired_ask)
            elif abs(q[1] - desired_ask) > FLY_REQUOTE_THRESH:
                self._cancel_fly_quote("ask")
                self._post_fly_quote("ask", desired_ask)
                self._requote_count += 1
        else:
            self._cancel_fly_quote("ask")

        self._last_quote_time = now

    def _post_fly_quote(self, side: str, price: float) -> None:
        order_side = Side.BUY if side == "bid" else Side.SELL
        try:
            resp = self.send_order(OrderRequest(LHR_FLY, price, order_side, FLY_QUOTE_VOL))
            if resp:
                self._fly_quotes[side] = (resp.id, price)
                print(
                    f"  [{ts()}] FLY QUOTE  {side.upper()} {FLY_QUOTE_VOL} "
                    f"LHR_FLY @ {price:.0f}  (id={resp.id})"
                )
        except Exception as exc:
            print(f"  [{ts()}] WARN  fly quote ({side}) failed: {exc}")

    def _cancel_fly_quote(self, side: str) -> None:
        q = self._fly_quotes[side]
        if q is None:
            return
        order_id, price = q
        try:
            self.cancel_order(order_id)
            print(f"  [{ts()}] FLY CANCEL {side.upper()} @ {price:.0f}  (id={order_id})")
        except Exception as exc:
            print(f"  [{ts()}] WARN  fly cancel ({side}) failed: {exc}")
        finally:
            self._fly_quotes[side] = None

    def _cancel_all_fly_quotes(self) -> None:
        self._cancel_fly_quote("bid")
        self._cancel_fly_quote("ask")

    # ── fire fly trade ───────────────────────────────────────────────────

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
        passive: bool = False,
    ) -> None:
        self._fly_count += 1
        trade_theo = edge * vol
        self._fly_theo_pnl += trade_theo

        # Record stats
        if passive:
            stat_key = "fly_passive_buy" if side == Side.BUY else "fly_passive_sell"
        else:
            stat_key = "fly_aggressive_buy" if side == Side.BUY else "fly_aggressive_sell"
        self._stats[stat_key].record(edge, vol)

        direction = "BUY " if side == Side.BUY else "SELL"
        mode = "PASSIVE" if passive else "AGGRESSIVE"
        print(f"\n{'═'*72}")
        print(f"[{ts()}] FLY #{self._fly_count}  {direction} LHR_FLY  [{mode}]")
        print(
            f"  LHR mid = {lhr_mid:.1f}  spread = {lhr_spread_pct*100:.2f}%  "
            f"fair = {fair:.1f}  market = {price:.0f}  edge = {edge:.1f}"
        )
        print(
            f"  vol = {vol}  trade PnL = +{trade_theo:.0f}  "
            f"cum fly PnL = +{self._fly_theo_pnl:.0f}  pos_fly = {pos_fly}"
        )

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
            self._skipped_arb_cooldown += 1
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

        p1_after_1 = (pe - 1, pa + 1, pb + 1, pc + 1)
        p2_after_1 = (pe + 1, pa - 1, pb - 1, pc - 1)

        case1_is_unwind = self._total_abs(p1_after_1) < self._total_abs(p0)
        case2_is_unwind = self._total_abs(p2_after_1) < self._total_abs(p0)

        inventory_skew = (pe / POS_LIMIT) * MAX_SKEW
        required_edge1_normal = MIN_EDGE - inventory_skew
        required_edge2_normal = MIN_EDGE + inventory_skew

        unwind_thresh = LIMIT_UNWIND_EDGE if self._near_limit(p0) else UNWIND_EDGE

        required_edge1 = unwind_thresh if case1_is_unwind else required_edge1_normal
        required_edge2 = unwind_thresh if case2_is_unwind else required_edge2_normal

        edge1 = E.bid_px - basket_ask
        if edge1 > required_edge1:
            vol = min(
                min(E.bid_sz, MAX_TRADE_VOL),
                min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL),
                POS_LIMIT + pe,
                POS_LIMIT - pa,
                POS_LIMIT - pb,
                POS_LIMIT - pc,
            )
            if vol >= 1:
                is_unwind = case1_is_unwind
                stat_key = "arb_unwind_sell_etf" if is_unwind else "arb_normal_sell_etf"
                self._stats[stat_key].record(edge1, vol)
                why = "UNWIND" if is_unwind else "NORMAL"
                self._fire_arb(
                    label=f"ETF RICH  → sell ETF, buy basket [{why}] (Req Edge: {required_edge1:.1f})",
                    edge_per_lot=edge1, vol=vol,
                    orders=[
                        OrderRequest(ETF,           E.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY,  vol),
                        OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY,  vol),
                        OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY,  vol),
                    ],
                    prices={"ETF_sell": E.bid_px, "A_buy": A.ask_px, "B_buy": B.ask_px, "C_buy": C.ask_px},
                    positions=p0,
                )
                self._last_trade_time = now
                return

        edge2 = basket_bid - E.ask_px
        if edge2 > required_edge2:
            vol = min(
                min(E.ask_sz, MAX_TRADE_VOL),
                min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL),
                POS_LIMIT - pe,
                POS_LIMIT + pa,
                POS_LIMIT + pb,
                POS_LIMIT + pc,
            )
            if vol >= 1:
                is_unwind = case2_is_unwind
                stat_key = "arb_unwind_buy_etf" if is_unwind else "arb_normal_buy_etf"
                self._stats[stat_key].record(edge2, vol)
                why = "UNWIND" if is_unwind else "NORMAL"
                self._fire_arb(
                    label=f"ETF CHEAP → buy ETF, sell basket [{why}] (Req Edge: {required_edge2:.1f})",
                    edge_per_lot=edge2, vol=vol,
                    orders=[
                        OrderRequest(ETF,           E.ask_px, Side.BUY,  vol),
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
            f"FLY_POS_LIMIT={FLY_POS_LIMIT}  MAX_SPREAD={FLY_MAX_SPREAD_PCT*100:.0f}%"
        )
        print(
            f"  fly quotes: offset=±{FLY_QUOTE_OFFSET}  size={FLY_QUOTE_VOL}  "
            f"requote_thresh={FLY_REQUOTE_THRESH}  cooldown={FLY_QUOTE_COOLDOWN}s"
        )
        print(f"  diagnostics every {DIAG_INTERVAL}s")
        print(f"  start PnL:  {self._start_pnl:.0f}")

        pos = self.get_positions()
        pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items())) if pos else "(flat)"
        print(f"  positions:  {pos_str}")
        print(f"  Ctrl+C to stop.\n")

        self.start()

        def _shutdown(sig, frame):
            print(f"\n[{ts()}] ═══ Shutting down ═══")
            self._cancel_all_fly_quotes()
            self.cancel_all_orders()
            self.stop()
            final = self.get_pnl()
            final_pnl = final.get("totalProfit", 0.0)
            session_pnl = final_pnl - self._start_pnl
            print(f"  final PnL:             {final_pnl:.0f}")
            print(f"  session PnL:           {session_pnl:+.0f}")
            print(f"  theoretical PnL (arb): +{self._theoretical_pnl:.0f}")
            print(f"  theoretical PnL (fly): +{self._fly_theo_pnl:.0f}")
            print(f"  arb trades:            {self._arb_count}")
            print(f"  fly trades:            {self._fly_count}")
            pos = self.get_positions()
            print(f"  positions:             {pos}")
            print()
            self._print_diag()
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
                bid_q = self._fly_quotes["bid"]
                ask_q = self._fly_quotes["ask"]
                q_str = (f"bid@{bid_q[1]:.0f}" if bid_q else "bid=--") + \
                        "  " + \
                        (f"ask@{ask_q[1]:.0f}" if ask_q else "ask=--")
                print(
                    f"[{ts()}] HEARTBEAT  arbs={self._arb_count}  flys={self._fly_count}  "
                    f"session={session_pnl:+.0f}  "
                    f"theo_arb={self._theoretical_pnl:+.0f}  theo_fly={self._fly_theo_pnl:+.0f}  "
                    f"actual={actual_pnl:.0f}  quotes=[{q_str}]  pos=[{pos_str}]"
                )
            except Exception as exc:
                print(f"[{ts()}] HEARTBEAT error: {exc}")


if __name__ == "__main__":
    bot = EtfArbBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()