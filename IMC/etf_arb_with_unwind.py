#!/usr/bin/env python3 -u
"""ETF cross-arbitrage bot with per-trade PnL tracking (with unwind).

Trades LON_ETF against its components (TIDE_SPOT + WX_SPOT + LHR_COUNT).
Runs indefinitely, prints every trade and theoretical PnL.

Key behavior:
- Normal arbs require MIN_EDGE (with mild ETF-position skew).
- If an arb would REDUCE total inventory risk (sum of abs positions),
  we accept a MUCH smaller edge (UNWIND_EDGE), and even smaller when near limits.

Usage:
    python -u etf_arb_with_unwind.py
    DIAG=1 python -u etf_arb_with_unwind.py   # verbose diagnostics
"""

from __future__ import annotations

import os
import signal
import sys
import time

os.environ["PYTHONUNBUFFERED"] = "1"

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Thread
from typing import Optional

import requests

from bot_template import BaseBot, OrderBook, OrderRequest, OrderResponse, Side, Trade

EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
USERNAME = "RATT"
PASSWORD = "ratt67"

ETF = "LON_ETF"
COMPONENTS = ["TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]
ALL_PRODUCTS = [ETF] + COMPONENTS

POS_LIMIT = 100
MIN_EDGE = 2.0
MIN_COOLDOWN = 0.2
MAX_TRADE_VOL = 7
MAX_SKEW = 1.5  # Max edge reduction when ETF position is at POS_LIMIT

# ── unwind behavior ─────────────────────────────────────────────────────
UNWIND_EDGE = 1.0        # accept low edge when trade reduces inventory risk
LIMIT_UNWIND_EDGE = 1.0  # accept almost no edge when near limits
LIMIT_NEAR = 95           # treat |pos| >= 90 as "near limit"
# ───────────────────────────────────────────────────────────────────────

# ── diagnostics ────────────────────────────────────────────────────────
DIAG = os.environ.get("DIAG", "").strip() in ("1", "true", "yes")
IOC_CANCEL_DELAY_S = 0.05  # small delay before cancel to allow exchange to match
SKIP_LOG_INTERVAL_S = 2.0  # min interval between "why no trade" logs when DIAG
# ───────────────────────────────────────────────────────────────────────


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
        self._last_skip_log_time = 0.0
        self._skip_reason: Optional[str] = None

    def _diag(self, msg: str) -> None:
        if DIAG:
            print(f"  [{ts()}] DIAG  {msg}")

    def send_order(self, order: OrderRequest) -> OrderResponse | None:
        """Override to add diagnostics and robust response parsing."""
        payload = asdict(order)
        url = f"{self._cmi_url}/api/order"
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self._auth_headers(),
                timeout=10,
            )
        except Exception as e:
            self._diag(f"send_order EXCEPTION {order.product} {order.side}: {e}")
            return None

        if DIAG:
            print(f"  [{ts()}] DIAG  REQ  {order.product} {order.side} {order.volume} @ {order.price}  payload={payload}")

        if not response.ok:
            print(
                f"  [{ts()}] ORDER FAIL  {order.product} {order.side} vol={order.volume}  "
                f"status={response.status_code}  body={response.text[:500]}"
            )
            return None

        try:
            data = response.json()
        except Exception as e:
            print(f"  [{ts()}] ORDER BAD JSON  {order.product}  {e}  body={response.text[:300]}")
            return None

        # API may return camelCase; normalize to OrderResponse field names
        key_map = {
            "productSymbol": "product",
            "filledVolume": "filled",
            "orderVolume": "volume",
        }
        normalized = dict(data)
        for api_key, our_key in key_map.items():
            if api_key in normalized and our_key not in normalized:
                normalized[our_key] = normalized[api_key]

        fields = set(OrderResponse.__dataclass_fields__)
        filtered = {k: v for k, v in normalized.items() if k in fields}
        try:
            return OrderResponse(**filtered)
        except (TypeError, KeyError) as e:
            print(
                f"  [{ts()}] ORDER PARSE ERR  {order.product}  {e}  "
                f"keys={list(data.keys())}  sample={str(data)[:400]}"
            )
            return None

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

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _total_abs(p: tuple[int, int, int, int]) -> int:
        return sum(abs(x) for x in p)

    @staticmethod
    def _near_limit(p: tuple[int, int, int, int]) -> bool:
        return any(abs(x) >= LIMIT_NEAR for x in p)

    # ── arbitrage logic ──────────────────────────────────────────────────

    def _log_skip(self, reason: str) -> None:
        """Log why we didn't trade (throttled when DIAG)."""
        if not DIAG:
            return
        now = time.monotonic()
        if now - self._last_skip_log_time >= SKIP_LOG_INTERVAL_S:
            self._last_skip_log_time = now
            self._diag(f"no trade: {reason}")

    def _maybe_arb(self) -> None:
        now = time.monotonic()
        if now - self._last_trade_time < MIN_COOLDOWN:
            self._log_skip("cooldown")
            return

        E = self._top[ETF]
        A = self._top[COMPONENTS[0]]
        B = self._top[COMPONENTS[1]]
        C = self._top[COMPONENTS[2]]

        if any(
            v is None
            for v in (E.bid_px, E.ask_px, A.bid_px, A.ask_px, B.bid_px, B.ask_px, C.bid_px, C.ask_px)
        ):
            self._log_skip("missing book (incomplete top of book)")
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

        # Case 1: ETF rich -> sell ETF, buy components (edge = E.bid - basket_ask)
        edge1 = E.bid_px - basket_ask
        if edge1 >= required_edge1:
            vol = min(
                min(E.bid_sz, MAX_TRADE_VOL),
                min(A.ask_sz, B.ask_sz, C.ask_sz, MAX_TRADE_VOL),
                POS_LIMIT + pe,  # SELL ETF headroom to -POS_LIMIT
                POS_LIMIT - pa,  # BUY component headroom to +POS_LIMIT
                POS_LIMIT - pb,
                POS_LIMIT - pc,
            )
            if vol >= 1:
                why = "UNWIND" if case1_is_unwind else "NORMAL"
                self._fire_arb(
                    label=(
                        f"ETF RICH  -> sell ETF, buy basket [{why}] "
                        f"(Req Edge: {required_edge1:.1f})"
                    ),
                    edge_per_lot=edge1,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF, E.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[0], A.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[1], B.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[2], C.ask_px, Side.BUY, vol),
                    ],
                    prices={"ETF_sell": E.bid_px, "A_buy": A.ask_px, "B_buy": B.ask_px, "C_buy": C.ask_px},
                    positions=p0,
                )
                self._last_trade_time = now
                return
            self._log_skip(f"case1 vol=0 (bid_sz={E.bid_sz} ask_sz={A.ask_sz},{B.ask_sz},{C.ask_sz} limits)")
        else:
            self._log_skip(
                f"case1 edge {edge1:.1f} < req {required_edge1:.1f} "
                f"(E.bid={E.bid_px} basket_ask={basket_ask:.0f})"
            )

        # Case 2: ETF cheap -> buy ETF, sell components (edge = basket_bid - E.ask)
        edge2 = basket_bid - E.ask_px
        if edge2 >= required_edge2:
            vol = min(
                min(E.ask_sz, MAX_TRADE_VOL),
                min(A.bid_sz, B.bid_sz, C.bid_sz, MAX_TRADE_VOL),
                POS_LIMIT - pe,  # BUY ETF headroom to +POS_LIMIT
                POS_LIMIT + pa,  # SELL component headroom to -POS_LIMIT
                POS_LIMIT + pb,
                POS_LIMIT + pc,
            )
            if vol >= 1:
                why = "UNWIND" if case2_is_unwind else "NORMAL"
                self._fire_arb(
                    label=(
                        f"ETF CHEAP -> buy ETF, sell basket [{why}] "
                        f"(Req Edge: {required_edge2:.1f})"
                    ),
                    edge_per_lot=edge2,
                    vol=vol,
                    orders=[
                        OrderRequest(ETF, E.ask_px, Side.BUY, vol),
                        OrderRequest(COMPONENTS[0], A.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[1], B.bid_px, Side.SELL, vol),
                        OrderRequest(COMPONENTS[2], C.bid_px, Side.SELL, vol),
                    ],
                    prices={"ETF_buy": E.ask_px, "A_sell": A.bid_px, "B_sell": B.bid_px, "C_sell": C.bid_px},
                    positions=p0,
                )
                self._last_trade_time = now
                return
            self._log_skip(f"case2 vol=0 (ask_sz={E.ask_sz} bid_sz={A.bid_sz},{B.bid_sz},{C.bid_sz} limits)")
        else:
            self._log_skip(
                f"case2 edge {edge2:.1f} < req {required_edge2:.1f} "
                f"(basket_bid={basket_bid:.0f} E.ask={E.ask_px})"
            )

    def _send_ioc(self, order: OrderRequest) -> tuple[OrderRequest, OrderResponse | None]:
        """Send an order and cancel any unfilled remainder after a short delay (IOC-like)."""
        resp = self.send_order(order)
        filled = getattr(resp, "filled", 0) if resp else 0
        if DIAG and resp:
            print(
                f"  [{ts()}] DIAG  RSP  {order.product} {order.side}  id={getattr(resp,'id','?')}  "
                f"status={getattr(resp,'status','?')}  filled={filled}  volume={getattr(resp,'volume',0)}"
            )
        if resp and getattr(resp, "volume", 0) > 0:
            if IOC_CANCEL_DELAY_S > 0:
                time.sleep(IOC_CANCEL_DELAY_S)
            try:
                self.cancel_order(resp.id)
                if DIAG:
                    self._diag(f"cancel sent for {order.product} id={resp.id}")
            except Exception as e:
                if DIAG:
                    self._diag(f"cancel failed {order.product} {resp.id}: {e}")
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
        print(f"\n{'-'*72}")
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

        def _filled(r: OrderResponse | None) -> int:
            return getattr(r, "filled", 0) if r else 0

        filled_legs = [(req, resp) for req, resp in results if resp and _filled(resp) > 0]
        fill_summary = "  ".join(
            f"{req.product}:{_filled(resp)}/{req.volume}" if resp else f"{req.product}:ERR/{req.volume}"
            for req, resp in results
        )

        if len(filled_legs) == len(orders):
            print(f"  FILLED  all {len(orders)} legs  ({fill_summary})")
        else:
            print(f"  PARTIAL {len(filled_legs)}/{len(orders)} legs  ({fill_summary})")
            # Diagnose which legs failed
            for req, resp in results:
                f = _filled(resp) if resp else -1
                if f != req.volume:
                    status = getattr(resp, "status", None) if resp else None
                    oid = getattr(resp, "id", None) if resp else None
                    print(
                        f"    MISS  {req.product} {req.side}  filled={f}/{req.volume}  "
                        f"status={status}  id={oid}"
                    )

        print(f"{'-'*72}")

    # ── run forever ──────────────────────────────────────────────────────

    def run_forever(self) -> None:
        """Start the SSE stream and block the main thread indefinitely."""
        pnl_data = self.get_pnl()
        self._start_pnl = pnl_data.get("totalProfit", 0.0)

        print(f"[{ts()}] === ETF Arbitrage Bot Started ===")
        print(f"  user:       {self.username}")
        print(f"  ETF:        {ETF}  =  {' + '.join(COMPONENTS)}")
        print(
            f"  params:     MIN_EDGE={MIN_EDGE}  UNWIND_EDGE={UNWIND_EDGE}  "
            f"MAX_VOL={MAX_TRADE_VOL}  POS_LIMIT={POS_LIMIT}  COOLDOWN={MIN_COOLDOWN}s"
        )
        print(f"  start PnL:  {self._start_pnl:.0f}")

        pos = self.get_positions()
        pos_str = "  ".join(f"{k}={v}" for k, v in sorted(pos.items())) if pos else "(flat)"
        print(f"  positions:  {pos_str}")
        if DIAG:
            print(f"  DIAG:      ON (skip log every {SKIP_LOG_INTERVAL_S}s, IOC delay {IOC_CANCEL_DELAY_S}s)")
        print(f"  Ctrl+C to stop.\n")

        self.start()

        def _shutdown(sig, frame):
            print(f"\n[{ts()}] === Shutting down ===")
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