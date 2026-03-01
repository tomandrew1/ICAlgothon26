#!/usr/bin/env python3 -u
"""
ETF cross-arbitrage bot with:
- Dynamic inventory-aware thresholds (ETF + basket tilt, not ETF-only)
- Unwind logic (accepts smaller edge if trade reduces total abs inventory)
- ETF-first execution (fill ETF first, then hedge components to match actual ETF fill)
- Hedge micro-retries + best-effort cleanup if hedges don’t fill
- Simple periodic heartbeat

Trades:
  LON_ETF against its components: TIDE_SPOT + WX_SPOT + LHR_COUNT

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

# ──────────────────────────────────────────────────────────────────────────────
# Connection / products
# ──────────────────────────────────────────────────────────────────────────────
EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
USERNAME = "RATT"
PASSWORD = "ratt67"

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ALL_PRODUCTS = [ETF] + COMPONENTS

# ──────────────────────────────────────────────────────────────────────────────
# Parameters (tuned for: more trades + less stuck inventory + fewer partial disasters)
# ──────────────────────────────────────────────────────────────────────────────
POS_LIMIT = 100
MIN_EDGE = 3.0
MIN_COOLDOWN = 0.15
MAX_TRADE_VOL = 3
MAX_SKEW = 2.0
\
UNWIND_EDGE = 1.0
LIMIT_UNWIND_EDGE = 0.25


HEDGE_RETRIES = 2          # number of extra hedge attempts after the first try
LEG_RETRY_MS = 50          # wait between hedge attempts if no progress
CLEANUP_RETRIES = 1        # try to reduce ETF exposure if left unhedged

MIN_TOP_SZ = 1  # set to 2 or 3 if you want to be more selective

HEARTBEAT_SECS = 30

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

    def has_full_quote(self) -> bool:
        return self.bid_px is not None and self.ask_px is not None

    def has_min_liquidity(self, min_sz: int) -> bool:
        return self.bid_sz >= min_sz and self.ask_sz >= min_sz


class EtfArbBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self._top: dict[str, TopOfBook] = {p: TopOfBook() for p in ALL_PRODUCTS}

        self._last_trade_time = 0.0
        self._arb_count = 0

        self._theoretical_pnl = 0.0
        self._start_pnl: Optional[float] = None

        # Diagnostics counters
        self._partial_events = 0
        self._cleanup_events = 0
        self._skipped_cooldown = 0
        self._skipped_missing_quotes = 0
        self._skipped_liquidity = 0

    # ────────────────────────────────────────────────────────────────────
    # Callbacks
    # ────────────────────────────────────────────────────────────────────

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

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _total_abs(p: tuple[int, int, int, int]) -> int:
        return sum(abs(x) for x in p)

    @staticmethod
    def _near_limit(p: tuple[int, int, int, int]) -> bool:
        return any(abs(x) >= LIMIT_NEAR for x in p)

    def _send_ioc(self, order: OrderRequest) -> tuple[OrderRequest, OrderResponse | None]:
        """
        Send a limit order and immediately cancel any unfilled remainder (IOC simulation).
        Assumes:
          - resp.filled exists
          - resp.volume > 0 indicates there is remainder open (per your earlier template)
        """
        resp = self.send_order(order)
        if resp and getattr(resp, "volume", 0) > 0:
            try:
                self.cancel_order(resp.id)
            except Exception:
                pass
        return order, resp

    def _get_quotes_or_skip(self) -> Optional[tuple[TopOfBook, TopOfBook, TopOfBook, TopOfBook]]:
        E = self._top[ETF]
        A = self._top[COMPONENTS[0]]
        B = self._top[COMPONENTS[1]]
        C = self._top[COMPONENTS[2]]

        if not (E.has_full_quote() and A.has_full_quote() and B.has_full_quote() and C.has_full_quote()):
            self._skipped_missing_quotes += 1
            return None

        if MIN_TOP_SZ > 0:
            if not (
                E.has_min_liquidity(MIN_TOP_SZ)
                and A.has_min_liquidity(MIN_TOP_SZ)
                and B.has_min_liquidity(MIN_TOP_SZ)
                and C.has_min_liquidity(MIN_TOP_SZ)
            ):
                self._skipped_liquidity += 1
                return None

        return E, A, B, C

    # ────────────────────────────────────────────────────────────────────
    # Inventory-aware thresholds
    # ────────────────────────────────────────────────────────────────────

    def _required_edges(
        self,
        p0: tuple[int, int, int, int],
        pe: int,
        pa: int,
        pb: int,
        pc: int,
    ) -> tuple[float, float, bool, bool]:
        """
        Returns:
          required_edge_case1, required_edge_case2, case1_is_unwind, case2_is_unwind

        Case 1: SELL ETF, BUY A,B,C  => (-1, +1, +1, +1)
        Case 2: BUY  ETF, SELL A,B,C => (+1, -1, -1, -1)

        Skew uses ETF minus average basket position (helps unwind component inventory too).
        """
        # Unwind test uses total abs inventory metric
        p_after_case1 = (pe - 1, pa + 1, pb + 1, pc + 1)
        p_after_case2 = (pe + 1, pa - 1, pb - 1, pc - 1)

        case1_is_unwind = self._total_abs(p_after_case1) < self._total_abs(p0)
        case2_is_unwind = self._total_abs(p_after_case2) < self._total_abs(p0)

        # Basket tilt proxy: if components are net long while ETF is flat, we still want to bias
        basket_tilt = (pa + pb + pc) / 3.0
        inv_signal = (pe - basket_tilt) / POS_LIMIT
        inventory_skew = inv_signal * MAX_SKEW

        required_edge1_normal = MIN_EDGE - inventory_skew
        required_edge2_normal = MIN_EDGE + inventory_skew

        unwind_thresh = LIMIT_UNWIND_EDGE if self._near_limit(p0) else UNWIND_EDGE

        required_edge1 = unwind_thresh if case1_is_unwind else required_edge1_normal
        required_edge2 = unwind_thresh if case2_is_unwind else required_edge2_normal

        return required_edge1, required_edge2, case1_is_unwind, case2_is_unwind

    # ────────────────────────────────────────────────────────────────────
    # Execution: ETF-first, hedge to actual fill, retries + cleanup
    # ────────────────────────────────────────────────────────────────────

    def _execute_arb_etf_first(
        self,
        label: str,
        edge_per_lot: float,
        target_vol: int,
        etf_leg: OrderRequest,
        hedge_legs_template: list[OrderRequest],
        positions: tuple[int, int, int, int],
        books_snapshot: tuple[TopOfBook, TopOfBook, TopOfBook, TopOfBook],
    ) -> None:
        self._arb_count += 1
        pe, pa, pb, pc = positions
        E, A, B, C = books_snapshot

        print(f"\n{'─'*72}")
        print(f"[{ts()}] ARB #{self._arb_count}  {label}")
        print(
            f"  edge/lot={edge_per_lot:.2f}  target_vol={target_vol}  "
            f"pos: ETF={pe} A={pa} B={pb} C={pc}"
        )
        print(f"  books: ETF {E}  A {A}  B {B}  C {C}")

        # 1) ETF leg first
        _, etf_resp = self._send_ioc(etf_leg)
        etf_filled = int(etf_resp.filled) if etf_resp else 0

        if etf_filled <= 0:
            print("  ETF leg: no fill -> abort")
            print(f"{'─'*72}")
            return

        print(f"  ETF leg filled {etf_filled}/{target_vol} -> hedging components")

        # 2) Hedge legs sized to ETF actual fill
        remaining: dict[str, int] = {o.product: etf_filled for o in hedge_legs_template}
        filled: dict[str, int] = {o.product: 0 for o in hedge_legs_template}

        for attempt in range(HEDGE_RETRIES + 1):
            any_progress = False

            for tmpl in hedge_legs_template:
                need = remaining[tmpl.product]
                if need <= 0:
                    continue

                req = OrderRequest(tmpl.product, tmpl.price, tmpl.side, need)
                _, resp = self._send_ioc(req)
                got = int(resp.filled) if resp else 0

                if got > 0:
                    filled[tmpl.product] += got
                    remaining[tmpl.product] -= got
                    any_progress = True

            if all(v <= 0 for v in remaining.values()):
                break

            if not any_progress:
                time.sleep(LEG_RETRY_MS / 1000.0)

        # 3) Cleanup if still unhedged
        unhedged_units = sum(max(v, 0) for v in remaining.values())
        if unhedged_units > 0:
            self._partial_events += 1
            print(f"  WARNING: unhedged component units total={unhedged_units} -> cleanup ETF")

            for _ in range(CLEANUP_RETRIES):
                cleanup_side = Side.SELL if etf_leg.side == Side.BUY else Side.BUY
                cleanup_px = self._top[ETF].bid_px if cleanup_side == Side.SELL else self._top[ETF].ask_px
                if cleanup_px is None:
                    print("  cleanup: no ETF quote -> stop cleanup")
                    break

                _, resp = self._send_ioc(OrderRequest(ETF, cleanup_px, cleanup_side, unhedged_units))
                cleaned = int(resp.filled) if resp else 0
                unhedged_units -= cleaned

                print(f"  cleanup ETF filled {cleaned}  remaining_unhedged={unhedged_units}")

                if cleaned > 0:
                    self._cleanup_events += 1

                if unhedged_units <= 0:
                    break

        # Theo accounting on ETF-filled sets only
        trade_theo = edge_per_lot * etf_filled
        self._theoretical_pnl += trade_theo

        hedge_summary = "  ".join(f"{k}:{filled[k]}/{etf_filled}" for k in filled)
        print(
            f"  SUMMARY: ETF_filled={etf_filled}  hedges=[{hedge_summary}]  "
            f"theoPnL=+{trade_theo:.0f}  cumTheo=+{self._theoretical_pnl:.0f}"
        )
        print(f"{'─'*72}")

    # ────────────────────────────────────────────────────────────────────
    # Arb logic
    # ────────────────────────────────────────────────────────────────────

    def _maybe_arb(self) -> None:
        now = time.monotonic()
        if now - self._last_trade_time < MIN_COOLDOWN:
            self._skipped_cooldown += 1
            return

        quotes = self._get_quotes_or_skip()
        if quotes is None:
            return
        E, A, B, C = quotes

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

        req1, req2, case1_is_unwind, case2_is_unwind = self._required_edges(p0, pe, pa, pb, pc)

        # Case 1: ETF rich -> SELL ETF, BUY basket
        edge1 = E.bid_px - basket_ask
        if edge1 > req1:
            vol = min(
                min(E.bid_sz, MAX_TRADE_VOL),
                min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL),
                POS_LIMIT + pe,  # SELL ETF headroom to -POS_LIMIT
                POS_LIMIT - pa,  # BUY comps headroom to +POS_LIMIT
                POS_LIMIT - pb,
                POS_LIMIT - pc,
            )
            if vol >= 1:
                why = "UNWIND" if case1_is_unwind else "NORMAL"
                label = f"ETF RICH -> sell ETF, buy basket [{why}] (req={req1:.2f})"

                self._execute_arb_etf_first(
                    label=label,
                    edge_per_lot=edge1,
                    target_vol=vol,
                    etf_leg=OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                    hedge_legs_template=[
                        OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol),
                    ],
                    positions=p0,
                    books_snapshot=quotes,
                )
                self._last_trade_time = now
                return

        # Case 2: ETF cheap -> BUY ETF, SELL basket
        edge2 = basket_bid - E.ask_px
        if edge2 > req2:
            vol = min(
                min(E.ask_sz, MAX_TRADE_VOL),
                min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL),
                POS_LIMIT - pe,  # BUY ETF headroom to +POS_LIMIT
                POS_LIMIT + pa,  # SELL comps headroom to -POS_LIMIT
                POS_LIMIT + pb,
                POS_LIMIT + pc,
            )
            if vol >= 1:
                why = "UNWIND" if case2_is_unwind else "NORMAL"
                label = f"ETF CHEAP -> buy ETF, sell basket [{why}] (req={req2:.2f})"

                self._execute_arb_etf_first(
                    label=label,
                    edge_per_lot=edge2,
                    target_vol=vol,
                    etf_leg=OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                    hedge_legs_template=[
                        OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol),
                    ],
                    positions=p0,
                    books_snapshot=quotes,
                )
                self._last_trade_time = now
                return

    # ────────────────────────────────────────────────────────────────────
    # Run loop
    # ────────────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        pnl_data = self.get_pnl()
        self._start_pnl = pnl_data.get("totalProfit", 0.0)

        print(f"[{ts()}] ═══ ETF Arbitrage Bot Started ═══")
        print(f"  user:        {self.username}")
        print(f"  products:    {ETF} = {' + '.join(COMPONENTS)}")
        print(
            f"  params:      MIN_EDGE={MIN_EDGE}  UNWIND_EDGE={UNWIND_EDGE}  "
            f"LIMIT_UNWIND_EDGE={LIMIT_UNWIND_EDGE}  LIMIT_NEAR={LIMIT_NEAR}"
        )
        print(
            f"              MAX_VOL={MAX_TRADE_VOL}  POS_LIMIT={POS_LIMIT}  "
            f"COOLDOWN={MIN_COOLDOWN}s  MAX_SKEW={MAX_SKEW}"
        )
        print(
            f"  exec:        ETF-first hedging, HEDGE_RETRIES={HEDGE_RETRIES}, "
            f"LEG_RETRY_MS={LEG_RETRY_MS}, CLEANUP_RETRIES={CLEANUP_RETRIES}"
        )
        print(f"  start PnL:   {self._start_pnl:.0f}")

        pos = self.get_positions()
        pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items())) if pos else "(flat)"
        print(f"  positions:   {pos_str}")
        print("  Ctrl+C to stop.\n")

        self.start()

        def _shutdown(sig, frame):
            print(f"\n[{ts()}] ═══ Shutting down ═══")
            self.cancel_all_orders()
            self.stop()

            final = self.get_pnl()
            final_pnl = final.get("totalProfit", 0.0)
            session_pnl = final_pnl - (self._start_pnl or 0.0)

            print(f"  final PnL:         {final_pnl:.0f}")
            print(f"  session PnL:       {session_pnl:+.0f}")
            print(f"  theoretical PnL:   {self._theoretical_pnl:+.0f}")
            print(f"  arb trades:        {self._arb_count}")
            print(f"  partial events:    {self._partial_events}")
            print(f"  cleanup events:    {self._cleanup_events}")
            print(f"  skipped cooldown:  {self._skipped_cooldown}")
            print(f"  skipped no quote:  {self._skipped_missing_quotes}")
            print(f"  skipped liquidity: {self._skipped_liquidity}")
            pos = self.get_positions()
            print(f"  positions:         {pos}")
            sys.exit(0)

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while True:
            time.sleep(HEARTBEAT_SECS)
            try:
                pnl_data = self.get_pnl()
                actual_pnl = pnl_data.get("totalProfit", 0.0)
                session_pnl = actual_pnl - (self._start_pnl or 0.0)
                pos = self.get_positions()
                pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items()))
                print(
                    f"[{ts()}] HEARTBEAT  arbs={self._arb_count}  "
                    f"session={session_pnl:+.0f}  theo={self._theoretical_pnl:+.0f}  "
                    f"actual={actual_pnl:.0f}  partials={self._partial_events}  "
                    f"cleanups={self._cleanup_events}  pos=[{pos_str}]"
                )
            except Exception as exc:
                print(f"[{ts()}] HEARTBEAT error: {exc}")


if __name__ == "__main__":
    bot = EtfArbBot(EXCHANGE_URL, USERNAME, PASSWORD)
    bot.run_forever()