#!/usr/bin/env python3 -u
"""Unwind all positions slowly by taking liquidity at best bid/ask.

Runs in a loop until flat. Ctrl+C to stop.
Usage: python -u unwind_positions.py [--dry-run] [--size N] [--delay S]
"""

from __future__ import annotations

import argparse
import signal
import time

from bot_template import BaseBot, OrderRequest, Side

EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com"
USERNAME = "RATT"
PASSWORD = "ratt67"

_stop = False


def _sigint(sig, frame):
    global _stop
    _stop = True
    print("\nStopping after this round...")


class UnwindBot(BaseBot):
    """Minimal bot for REST-only position unwinding."""

    def on_orderbook(self, orderbook):
        pass

    def on_trades(self, trade):
        pass


# Only unwind these (ETF arb legs); ignore other products
UNWIND_PRODUCTS = ["LON_ETF", "TIDE_SPOT", "WX_SPOT", "LHR_COUNT"]


def run_round(bot: UnwindBot, size: int, dry_run: bool) -> bool:
    """Run one unwind round. Returns True if any orders sent."""
    pos = bot.get_positions()
    products_with_pos = [p for p in UNWIND_PRODUCTS if pos.get(p, 0) != 0]
    if not products_with_pos:
        return False

    orders_to_send = []
    for product in sorted(products_with_pos):
        p = pos[product]
        try:
            ob = bot.get_orderbook(product)
        except Exception as e:
            print(f"  WARN: {product}: {e}")
            continue

        vol = min(size, abs(p))
        if vol < 1:
            continue

        if p > 0:
            if not ob.buy_orders:
                continue
            price = ob.buy_orders[0].price
            avail = int(ob.buy_orders[0].volume)
            vol = min(vol, avail)
            if vol >= 1:
                orders_to_send.append((product, OrderRequest(product, price, Side.SELL, vol)))
        else:
            if not ob.sell_orders:
                continue
            price = ob.sell_orders[0].price
            avail = int(ob.sell_orders[0].volume)
            vol = min(vol, avail)
            if vol >= 1:
                orders_to_send.append((product, OrderRequest(product, price, Side.BUY, vol)))

    if not orders_to_send:
        return False

    pos_str = "  ".join(f"{p}={pos.get(p,0)}" for p in sorted(pos.keys()) if pos.get(p, 0))
    print(f"  pos: [{pos_str}]")
    for prod, o in orders_to_send:
        print(f"    {o.side} {o.volume} {prod} @ {o.price:.0f}")

    if dry_run:
        return True

    for prod, order in orders_to_send:
        resp = bot.send_order(order)
        filled = getattr(resp, "filled", 0) if resp else 0
        print(f"    {prod}: filled={filled}/{order.volume}")
        time.sleep(0.5)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print only, no orders")
    parser.add_argument("--size", type=int, default=2, help="Volume per leg per round (default 2)")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds between rounds (default 3)")
    args = parser.parse_args()

    bot = UnwindBot(EXCHANGE_URL, USERNAME, PASSWORD)
    signal.signal(signal.SIGINT, _sigint)

    print("Unwinding all positions slowly. Ctrl+C to stop.\n")

    round_num = 0
    while not _stop:
        round_num += 1
        print(f"[round {round_num}]")
        if not run_round(bot, args.size, args.dry_run):
            print("  All flat.")
            break
        if _stop:
            break
        if not args.dry_run:
            time.sleep(args.delay)

    print("Done.")


if __name__ == "__main__":
    main()
