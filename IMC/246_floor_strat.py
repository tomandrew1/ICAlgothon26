import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
import traceback

# Import the CMI Exchange framework from your bot_template
from bot_template import BaseBot, OrderBook, OrderRequest, Side

# --- Configuration ---
EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com/"
USERNAME = "RATT"
PASSWORD = "ratt67"

AERODATABOX_KEY = "34f9f54137mshaebbce78c55e61dp194894jsnad885717ce27"
AERODATABOX_HOST = "aerodatabox.p.rapidapi.com"

# The trading session runs Saturday 12:00 PM to Sunday 12:00 PM
SESSION_START = pd.to_datetime("2026-02-28 12:00:00").tz_localize("Europe/London")

# Strategy Constants
TARGET_PRODUCTS = ["WX_SUM", "TIDE_SWING", "LHR_COUNT"]
IMBALANCE_THRESHOLD = 20.0 
TRADE_VOLUME = 10 

# --- Strategy 1: Data Fetching & Floor Calculation ---

def get_wx_sum_floor() -> float:
    try:
        resp = requests.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": 51.5074, 
            "longitude": -0.1278,
            "minutely_15": "temperature_2m,relative_humidity_2m",
            "past_minutely_15": 96, 
            "timezone": "Europe/London",
            "temperature_unit": "fahrenheit" 
        }, timeout=10) # Added Timeout
        
        if not resp.ok:
            return 0.0
            
        m = resp.json()["minutely_15"]
        df = pd.DataFrame({
            "time": pd.to_datetime(m["time"]).tz_localize("Europe/London"),
            "temp": m["temperature_2m"],
            "humidity": m["relative_humidity_2m"],
        })
        
        now = pd.Timestamp.now(tz="Europe/London")
        df_session = df[(df["time"] >= SESSION_START) & (df["time"] < now)].copy()
        
        if df_session.empty: return 0.0
            
        df_session["val"] = (df_session["temp"] * df_session["humidity"]) / 100
        return df_session["val"].sum()
    except Exception:
        return 0.0

def get_tide_swing_floor() -> float:
    try:
        resp = requests.get(
            "https://environment.data.gov.uk/flood-monitoring/id/measures/0006-level-tidal_level-i-15_min-mAOD/readings",
            params={"_sorted": "", "_limit": 200},
            timeout=10 # Added Timeout
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
            
        df_session["strangle"] = df_session["diff_cm"].apply(strangle)
        return df_session["strangle"].sum()
    except Exception:
        return 0.0

def get_lhr_count_floor() -> float:
    try:
        now = pd.Timestamp.now(tz="Europe/London")
        start_str = SESSION_START.strftime("%Y-%m-%dT%H:%M")
        now_str = now.strftime("%Y-%m-%dT%H:%M")
        
        url = f"https://{AERODATABOX_HOST}/flights/airports/iata/LHR/{start_str}/{now_str}?direction=Both"
        resp = requests.get(url, headers={
            "x-rapidapi-host": AERODATABOX_HOST, 
            "x-rapidapi-key": AERODATABOX_KEY
        }, timeout=10) # Added Timeout
        
        if not resp.ok: return 0.0
            
        data = resp.json()
        arrivals = len(data.get("arrivals", []))
        departures = len(data.get("departures", []))
        return float(arrivals + departures)
    except Exception:
        return 0.0


# --- Combined Trading Bot ---

class CombinedStrategyBot(BaseBot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.floors = {"WX_SUM": 0.0, "TIDE_SWING": 0.0, "LHR_COUNT": 0.0}
        self.last_update = 0
        self.last_trade_interval = None

    def update_floors(self):
        now = time.time()
        if now - self.last_update > 900: # 15 minutes
            print(f"\n[{pd.Timestamp.now(tz='Europe/London').strftime('%H:%M:%S')}] 🔄 Updating Market Floors from APIs...")
            self.floors["WX_SUM"] = get_wx_sum_floor()
            self.floors["TIDE_SWING"] = get_tide_swing_floor()
            self.floors["LHR_COUNT"] = get_lhr_count_floor()
            
            for prod in TARGET_PRODUCTS:
                print(f"  -> {prod} Locked-in Floor: {self.floors[prod]:.2f}")
            self.last_update = now

    def on_orderbook(self, ob: OrderBook):
        """Listens to the SSE stream to catch trades dipping below the floor."""
        if ob.product not in TARGET_PRODUCTS:
            return
            
        floor = self.floors.get(ob.product, 0.0)
        if floor <= 0 or not ob.sell_orders:
            return 
            
        best_ask = ob.sell_orders[0]
        
        # Floor Exploit logic
        if best_ask.price < floor:
            try:
                pos = self.get_positions().get(ob.product, 0)
                if pos < 100: 
                    trade_volume = min(int(best_ask.volume), 100 - pos)
                    if trade_volume > 0:
                        print(f"🚨 ARB DETECTED! {ob.product} Ask @ {best_ask.price} is BELOW Floor ({floor:.2f})")
                        self._execute_ioc(ob.product, best_ask.price, trade_volume)
            except Exception as e:
                print(f"Error executing arb: {e}")

    def _execute_ioc(self, product: str, price: float, volume: int):
        order = OrderRequest(product=product, price=price, side=Side.BUY, volume=volume)
        print(f"  -> Sending IOC BUY {volume} {product} @ {price}")
        resp = self.send_order(order)
        
        if resp and resp.volume > 0:
            try:
                self.cancel_order(resp.id)
            except Exception:
                pass 

    def get_next_interval_window(self):
        now = pd.Timestamp.now(tz="Europe/London")
        if now.minute < 30:
            start = now.replace(minute=30, second=0, microsecond=0)
            end = start + timedelta(minutes=29)
        else:
            start = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(minutes=29)
        return start, end

    def fetch_scheduled_imbalance(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
        try:
            start_str = start_time.strftime("%Y-%m-%dT%H:%M")
            end_str = end_time.strftime("%Y-%m-%dT%H:%M")
            
            url = f"https://{AERODATABOX_HOST}/flights/airports/iata/LHR/{start_str}/{end_str}?direction=Both"
            resp = requests.get(url, headers={
                "x-rapidapi-host": AERODATABOX_HOST, 
                "x-rapidapi-key": AERODATABOX_KEY
            }, timeout=10)
            
            if not resp.ok:
                print(f"API Error fetching flights: {resp.status_code}")
                return 0.0
                
            data = resp.json()
            arrivals = len(data.get("arrivals", []))
            departures = len(data.get("departures", []))
            
            metric = 100.0 * (arrivals - departures) / max(arrivals + departures, 1)
            
            print(f"  -> Scheduled for {start_str}: {arrivals} Arrivals, {departures} Departures")
            print(f"  -> Projected Index Contribution: {metric:+.2f}")
            return metric
        except Exception as e:
            print(f"Error calculating scheduled imbalance: {e}")
            return 0.0

    def _aggress(self, product: str, side: Side, volume: int):
        ob = self.get_orderbook(product)
        if side == Side.BUY and ob.sell_orders:
            best_ask = ob.sell_orders[0].price
            order = OrderRequest(product=product, price=best_ask, side=Side.BUY, volume=volume)
            self.send_order(order)
        elif side == Side.SELL and ob.buy_orders:
            best_bid = ob.buy_orders[0].price
            order = OrderRequest(product=product, price=best_bid, side=Side.SELL, volume=volume)
            self.send_order(order)

    def on_trades(self, trade):
        if trade.buyer == self.username or trade.seller == self.username:
            direction = "BOUGHT" if trade.buyer == self.username else "SOLD"
            print(f"✅ {direction} {trade.volume} {trade.product} @ {trade.price}")

    def run_strategy(self):
        print("🚀 Starting Combined Strategy Bot...")
        self.update_floors()
        self.start()
        
        last_heartbeat = 0
        
        while True:
            try:
                # 1. Update cumulative floors (throttled internally to 15 mins)
                self.update_floors()
                
                # 2. Check time for LHR_INDEX Front-Running
                now = pd.Timestamp.now(tz="Europe/London")
                
                if now.minute in [28, 29, 58, 59]:
                    interval_id = now.strftime("%Y%m%d%H") + ("30" if now.minute >= 30 else "00")
                    
                    if self.last_trade_interval != interval_id:
                        print(f"\n[{now.strftime('%H:%M:%S')}] ⏳ Approaching interval rollover! Analyzing next window...")
                        
                        start_time, end_time = self.get_next_interval_window()
                        projected_metric = self.fetch_scheduled_imbalance(start_time, end_time)
                        
                        if projected_metric > IMBALANCE_THRESHOLD:
                            print(f"🚨 Severe ARRIVAL wave incoming (+{projected_metric:.1f}). Buying LHR_INDEX!")
                            self._aggress("LHR_INDEX", Side.BUY, TRADE_VOLUME)
                            self.last_trade_interval = interval_id
                            
                        elif projected_metric < -IMBALANCE_THRESHOLD:
                            print(f"🚨 Severe DEPARTURE wave incoming ({projected_metric:.1f}). Selling LHR_INDEX!")
                            self._aggress("LHR_INDEX", Side.SELL, TRADE_VOLUME)
                            self.last_trade_interval = interval_id
                            
                        else:
                            print(f"⚖️ Market balanced ({projected_metric:.1f}). No trade this interval.")
                            self.last_trade_interval = interval_id

                # 3. Heartbeat Monitor (Prints every 60 seconds)
                current_time = time.time()
                if current_time - last_heartbeat > 60:
                    try:
                        pos = self.get_positions()
                        # Format positions so they are easy to read
                        pos_str = ", ".join([f"{k}: {v}" for k, v in pos.items() if v != 0]) or "Flat (0)"
                        print(f"[{now.strftime('%H:%M:%S')}] 💓 HEARTBEAT: Actively scanning. Open Positions -> {pos_str}")
                    except Exception:
                        pass # Ignore heartbeat API failures
                    last_heartbeat = current_time
                    
            except Exception as e:
                print(f"⚠️ Recovered from Main Loop Error: {e}")
                traceback.print_exc()

            time.sleep(10)

if __name__ == "__main__":
    bot = CombinedStrategyBot(EXCHANGE_URL, USERNAME, PASSWORD)
    try:
        bot.run_strategy()
    except KeyboardInterrupt:
        print("\nStopping bot...")
        bot.cancel_all_orders()
        bot.stop()