#!/usr/bin/env python3 -u
"""Weather Market Making & Value Taker Bot.

Trades WX_SPOT (Market 3) and WX_SUM (Market 4) based on Open-Meteo forecasts.
Runs a background thread to continuously update the Fair Value (FV).
"""

import time
import requests
import pandas as pd
from threading import Thread
from datetime import datetime, timezone

from bot_template import BaseBot, OrderBook, OrderRequest, Side, Trade

EXCHANGE_URL = "http://ec2-52-49-69-152.eu-west-1.compute.amazonaws.com/"
USERNAME = "timgu"
PASSWORD = "1!Qwerty"

WX_SPOT = "WX_SPOT"
WX_SUM = "WX_SUM"

LONDON_LAT, LONDON_LON = 51.5074, -0.1278
POS_LIMIT = 100  # Strict position limit of +/- 100 per product
MIN_EDGE = 15.0  # Minimum point difference required between FV and market price to trade
MAX_SKEW = 20.0  # Max amount to artificially shift our FV when approaching position limits
TRADE_COOLDOWN = 1.0 # Seconds between trades for the same product
MAX_TRADE_VOL = 5

class WeatherBot(BaseBot):
    def __init__(self, cmi_url: str, username: str, password: str):
        super().__init__(cmi_url, username, password)
        self.spot_fv = None
        self.sum_fv = None
        
        self._last_trade_times = {WX_SPOT: 0.0, WX_SUM: 0.0}
        
        # Define the exact 24h trading session window
        self.session_start = pd.to_datetime("2026-02-28 12:00:00").tz_localize("Europe/London")
        self.session_end = pd.to_datetime("2026-03-01 12:00:00").tz_localize("Europe/London")

        # Start the background pricing thread
        self._pricing_thread = Thread(target=self._update_fv_loop, daemon=True)
        self._pricing_thread.start()

    # ─── Step 1: Continuous Fair Value Engine ─────────────────────────────

    def _fetch_weather_data(self):
        """Fetches past and forecast 15-minute data from Open-Meteo."""
        variables = "temperature_2m,relative_humidity_2m"
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": LONDON_LAT, "longitude": LONDON_LON,
            "minutely_15": variables,
            "past_minutely_15": 96,     # Grab past 24h to ensure we have realized data
            "forecast_minutely_15": 96, # Grab next 24h for forecast data
            "timezone": "Europe/London",
        })
        resp.raise_for_status()
        m = resp.json()["minutely_15"]
        
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temperature": m["temperature_2m"],
            "humidity": m["relative_humidity_2m"],
        })
        
        # Math from your Jupyter Notebook
        df['temp_F'] = (df['temperature'] * 9/5) + 32
        df['wx_metric'] = df['temp_F'] * df['humidity']
        return df

    def _update_fv_loop(self):
        """Runs continuously to update Fair Values every 5 minutes."""
        while True:
            try:
                df = self._fetch_weather_data()
                
                # Filter strictly for the Saturday 12:00 to Sunday 12:00 window
                session_df = df[(df['time'] > self.session_start) & (df['time'] <= self.session_end)]
                
                # WX_SPOT: Settles at exactly Sunday 12:00 PM
                spot_row = session_df[session_df['time'] == self.session_end]
                if not spot_row.empty:
                    self.spot_fv = spot_row['wx_metric'].iloc[0]
                
                # WX_SUM: Sum of all 15-min intervals divided by 100
                if not session_df.empty:
                    self.sum_fv = session_df['wx_metric'].sum() / 100.0

                print(f"[PRICER] Updated FVs -> WX_SPOT: {self.spot_fv:.1f} | WX_SUM: {self.sum_fv:.1f}")
            
            except Exception as e:
                print(f"[PRICER ERROR] Failed to fetch weather data: {e}")
            
            # Wait 5 minutes before pinging Open-Meteo again
            time.sleep(300)

    # ─── Step 2 & 3: Taker Strategy & Inventory Skewing ───────────────────

    def on_orderbook(self, orderbook: OrderBook) -> None:
        """Evaluates every orderbook tick against our Fair Value."""
        product = orderbook.product
        if product not in [WX_SPOT, WX_SUM]:
            return
            
        now = time.monotonic()
        if now - self._last_trade_times[product] < TRADE_COOLDOWN:
            return

        fv = self.spot_fv if product == WX_SPOT else self.sum_fv
        if fv is None:
            return # Don't trade if we haven't successfully pulled data yet

        # Get top of book
        best_bid = orderbook.buy_orders[0] if orderbook.buy_orders else None
        best_ask = orderbook.sell_orders[0] if orderbook.sell_orders else None

        # Fetch our current position to apply skewing
        positions = self.get_positions()
        current_pos = int(positions.get(product, 0))

        # Calculate Skew: If we are extremely long, skew lowers our FV, making us less likely to buy and more likely to sell.
        inventory_ratio = current_pos / POS_LIMIT
        skew = inventory_ratio * MAX_SKEW
        skewed_fv = fv - skew

        # Check for BUY signal: If the market ask is significantly cheaper than our skewed FV
        if best_ask and best_ask.price < (skewed_fv - MIN_EDGE):
            available_vol = int(best_ask.volume)
            room_to_buy = POS_LIMIT - current_pos
            vol_to_trade = min(available_vol, MAX_TRADE_VOL, room_to_buy)
            
            if vol_to_trade > 0:
                print(f"[{product}] BUY SIGNAL: Market Ask {best_ask.price:.1f} < Skewed FV {skewed_fv:.1f} (Base FV: {fv:.1f})")
                order = OrderRequest(product, best_ask.price, Side.BUY, vol_to_trade)
                self._send_ioc(order)
                self._last_trade_times[product] = now

        # Check for SELL signal: If the market bid is significantly higher than our skewed FV
        if best_bid and best_bid.price > (skewed_fv + MIN_EDGE):
            available_vol = int(best_bid.volume)
            room_to_sell = POS_LIMIT + current_pos # Math works because current_pos would be negative if short
            vol_to_trade = min(available_vol, MAX_TRADE_VOL, room_to_sell)
            
            if vol_to_trade > 0:
                print(f"[{product}] SELL SIGNAL: Market Bid {best_bid.price:.1f} > Skewed FV {skewed_fv:.1f} (Base FV: {fv:.1f})")
                order = OrderRequest(product, best_bid.price, Side.SELL, vol_to_trade)
                self._send_ioc(order)
                self._last_trade_times[product] = now

    def _send_ioc(self, order: OrderRequest):
        """Sends an order and immediately attempts to cancel any unfilled remainder."""
        resp = self.send_order(order)
        if resp and resp.volume > 0:
            try:
                self.cancel_order(resp.id)
            except Exception:
                pass

    def on_trades(self, trade: Trade) -> None:
        direction = "BOT" if trade.buyer == self.username else "SLD"
        print(f"  [FILL] {direction} {trade.volume} {trade.product} @ {trade.price:.0f}")

if __name__ == "__main__":
    bot = WeatherBot(EXCHANGE_URL, USERNAME, PASSWORD)
    print("Starting Weather Fair Value Bot...")
    bot.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bot.cancel_all_orders()
        bot.stop()
        print("Bot stopped.")