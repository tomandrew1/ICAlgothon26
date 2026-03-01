import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class ThamesPredictor:
    """
    One-stop helper for:
      - data fetch (Environment Agency Thames tidal level @15min)
      - TIDE_SWING strangle calc + settlements
      - level z-scores today since 12:00 vs historical same-time
      - expected settlement at next 12:00 using historical same-time averages
      - noon-level sine fit (TIDE_SPOT-style) + prediction
      - plotting methods for all the above
    """
    measure: str = "0006-level-tidal_level-i-15_min-mAOD"
    tz: str = "Europe/London"
    freq: str = "15min"
    settle_hour: int = 12

    # TIDE_SWING strikes (meters)
    K_put: float = 0.20
    K_call: float = 0.25

    # ---------------------------
    # Fetch
    # ---------------------------
    def get_thames(self, limit: int = 1450) -> pd.DataFrame:
        url = f"https://environment.data.gov.uk/flood-monitoring/id/measures/{self.measure}/readings"
        resp = requests.get(url, params={"_sorted": "", "_limit": limit})
        resp.raise_for_status()
        items = resp.json().get("items", [])
        df = (
            pd.DataFrame(items)[["dateTime", "value"]]
            .rename(columns={"dateTime": "time", "value": "level"})
        )
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(self.tz)
        return df.sort_values("time").reset_index(drop=True)

    # ---------------------------
    # Core: diffs + strangle
    # ---------------------------
    def add_strangle_columns(self, zz: pd.DataFrame) -> pd.DataFrame:
        """
        Adds:
          diff = abs(level_t - level_{t-1})
          put_payoff = max(0, K_put - diff)
          call_payoff = max(0, diff - K_call)
          strangle = put + call
        """
        z = zz.sort_values("time").copy()
        z["time"] = pd.to_datetime(z["time"])
        z = z.dropna(subset=["level"])

        z["diff"] = z["level"].diff().abs()
        z["put_payoff"] = np.maximum(0.0, self.K_put - z["diff"])
        z["call_payoff"] = np.maximum(0.0, z["diff"] - self.K_call)
        z["strangle"] = z["put_payoff"] + z["call_payoff"]
        return z

    # ---------------------------
    # Settlement calculations
    # ---------------------------
    def settlement_last_24h_ending_now(self, zz: pd.DataFrame) -> Dict[str, object]:
        z = self.add_strangle_columns(zz)
        end = z["time"].max()
        start = end - pd.Timedelta(hours=24)
        mask = (z["time"] > start) & (z["time"] <= end)
        settlement = 100.0 * float(z.loc[mask, "strangle"].sum())
        return {"start": start, "end": end, "intervals": int(mask.sum()), "settlement": settlement}

    def rolling_settlement_series(self, zz: pd.DataFrame, window: str = "24H") -> pd.Series:
        z = self.add_strangle_columns(zz)
        zt = z.set_index("time").sort_index()
        return 100.0 * zt["strangle"].rolling(window).sum()

    def daily_noon_settlements(self, zz: pd.DataFrame) -> pd.DataFrame:
        rs = self.rolling_settlement_series(zz, window="24H")
        noon = rs.at_time(f"{self.settle_hour:02d}:00")
        out = noon.rename("settlement").reset_index()
        out["date"] = out["time"].dt.date
        return out

    # ---------------------------
    # Z-score: level today since 12:00 vs historical same-time
    # ---------------------------
    def zscore_today_since_noon_level(self, zz: pd.DataFrame) -> pd.DataFrame:
        z = zz.sort_values("time").copy()
        z["time"] = pd.to_datetime(z["time"])
        z = z.dropna(subset=["level"])
        z["date"] = z["time"].dt.date
        z["tod"] = z["time"].dt.strftime("%H:%M")

        today_date = z["date"].max()
        today_start = pd.Timestamp(today_date).tz_localize(self.tz) + pd.Timedelta(hours=self.settle_hour)

        today = z[(z["date"] == today_date) & (z["time"] >= today_start)].copy()
        history = z[z["date"] < today_date].copy()

        stats = history.groupby("tod")["level"].agg(["mean", "std"])
        today = today.merge(stats, left_on="tod", right_index=True, how="left")

        today["zscore"] = (today["level"] - today["mean"]) / today["std"]
        return today[["time", "level", "tod", "mean", "std", "zscore"]]

    # ---------------------------
    # Expected settlement at next 12:00 using historical same-time mean strangle
    # ---------------------------
    def expected_settlement_next_noon(self, zz: pd.DataFrame, *, use_same_weekday: bool = False) -> Dict[str, object]:
        """
        Expected Settle(next_noon) for TIDE_SWING-like contract:
          Settle(T) = 100 * sum_{t in (T-24h, T]} strangle_t

        We compute:
          realised = sum strangle for times in (T-24h, now]
          expected_remaining = sum over future 15-min stamps (now, T] of E[strangle | HH:MM]
        """
        z = self.add_strangle_columns(zz)
        z["tod"] = z["time"].dt.strftime("%H:%M")

        now = z["time"].max()

        next_noon = now.normalize() + pd.Timedelta(hours=self.settle_hour)
        if now >= next_noon:
            next_noon += pd.Timedelta(days=1)

        window_start = next_noon - pd.Timedelta(hours=24)
        window_end = next_noon

        realised_mask = (z["time"] > window_start) & (z["time"] <= now)
        realised_sum = float(z.loc[realised_mask, "strangle"].sum())

        start_future = (now + pd.Timedelta(self.freq)).floor(self.freq)
        future_times = pd.date_range(start=start_future, end=window_end, freq=self.freq)
        future_tods = pd.Series(future_times.strftime("%H:%M"), index=future_times)

        hist = z[z["time"] <= window_start].copy()
        if use_same_weekday:
            wd = next_noon.dayofweek
            hist = hist[hist["time"].dt.dayofweek == wd]

        hist_means = hist.groupby("tod")["strangle"].mean()
        expected_remaining_sum = float(hist_means.reindex(future_tods.values).sum(skipna=True))

        coverage = float(hist_means.reindex(future_tods.values).notna().mean()) if len(future_tods) else np.nan

        return {
            "now": now,
            "next_noon": next_noon,
            "final_window": (window_start, window_end),
            "realised_component": 100.0 * realised_sum,
            "expected_remaining_component": 100.0 * expected_remaining_sum,
            "expected_settlement": 100.0 * (realised_sum + expected_remaining_sum),
            "future_times": future_times,                 # for plotting/debug
            "future_expected_per_tick": hist_means.reindex(future_tods.values).to_numpy(),
            "history_coverage": coverage,
        }

    # ---------------------------
    # Noon-level sine fit (TIDE_SPOT-style)
    # ---------------------------
    def noon_level_series(self, zz: pd.DataFrame, *, resample_to_1min: bool = True) -> pd.DataFrame:
        z = zz.sort_values("time").copy()
        z["time"] = pd.to_datetime(z["time"])
        z = z.dropna(subset=["level"])

        zi = z.set_index("time").sort_index()
        if resample_to_1min:
            zi = zi.resample("1min").interpolate(method="time")

        dates = pd.Series(zi.index.date).unique()
        rows = []
        for d in dates:
            noon = pd.Timestamp(f"{d} {self.settle_hour:02d}:00:00").tz_localize(self.tz)
            if noon in zi.index:
                rows.append({"time": noon, "date": d, "level": float(zi.loc[noon, "level"])})
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)

    def fit_noon_sine_and_predict(self, zz: pd.DataFrame, target_time: Optional[pd.Timestamp] = None) -> Dict[str, object]:
        from scipy.optimize import curve_fit

        df_noon = self.noon_level_series(zz, resample_to_1min=True)
        if len(df_noon) < 6:
            return {"ok": False, "reason": "Not enough noon points to fit robustly.", "n_points": len(df_noon)}

        if target_time is None:
            now = pd.to_datetime(zz["time"]).max()
            target_time = now.normalize() + pd.Timedelta(hours=self.settle_hour)
            if now >= target_time:
                target_time += pd.Timedelta(days=1)

        def sine(x, A, period, phase, offset):
            return A * np.sin(2 * np.pi / period * (x - phase)) + offset

        x0 = df_noon["time"].iloc[0]
        x = (df_noon["time"] - x0).dt.total_seconds().to_numpy() / 3600.0
        y = df_noon["level"].to_numpy()

        # conservative initial guess and bounds
        p0 = [0.8, 24 * 14, 0.0, float(np.mean(y))]
        bounds = ([0.0, 24 * 5, -np.inf, -np.inf], [np.inf, 24 * 60, np.inf, np.inf])

        params, cov = curve_fit(sine, x, y, p0=p0, bounds=bounds, maxfev=20000)

        tx = (target_time - x0).total_seconds() / 3600.0
        yhat = float(sine(tx, *params))

        # build a fitted curve for plotting
        x_fit = np.linspace(float(x.min()), float(tx), 800)
        y_fit = sine(x_fit, *params)
        t_fit = x0 + pd.to_timedelta(x_fit, unit="h")

        return {
            "ok": True,
            "n_points": len(df_noon),
            "params": {"A": float(params[0]), "period_hours": float(params[1]), "phase_hours": float(params[2]), "offset": float(params[3])},
            "target_time": target_time,
            "predicted_level": yhat,
            "predicted_abs_mm": float(abs(yhat) * 1000.0),
            "noon_points": df_noon,
            "fit_curve": (t_fit, y_fit),
        }

    # =====================================================================
    # PLOTTING METHODS
    # =====================================================================

    def plot_zscore_today_since_noon(self, zz: pd.DataFrame) -> pd.DataFrame:
        df = self.zscore_today_since_noon_level(zz)

        plt.figure(figsize=(10, 4.5))
        plt.plot(df["time"], df["zscore"], marker="o")
        plt.axhline(0)
        plt.axhline(2, linestyle="--")
        plt.axhline(-2, linestyle="--")
        plt.title("Z-score of 15-min Level Since 12:00 (vs historical same-time)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return df

    def plot_daily_noon_settlements(self, zz: pd.DataFrame, *, last_n: Optional[int] = None) -> pd.DataFrame:
        df = self.daily_noon_settlements(zz)
        if last_n is not None:
            df = df.tail(last_n).reset_index(drop=True)

        plt.figure(figsize=(10, 4.5))
        plt.plot(df["time"], df["settlement"], marker="o")
        plt.axhline(0)
        plt.title("Daily Settlement at 12:00 (rolling 24h) for TIDE_SWING")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return df

    def plot_expected_settlement_breakdown(self, zz: pd.DataFrame, *, use_same_weekday: bool = False) -> Dict[str, object]:
        out = self.expected_settlement_next_noon(zz, use_same_weekday=use_same_weekday)

        # 1) Bar breakdown
        plt.figure(figsize=(8, 4.5))
        plt.bar(["Realised so far", "Expected remaining"], [out["realised_component"], out["expected_remaining_component"]])
        plt.title(f"Expected Settlement at {out['next_noon']} (breakdown)")
        plt.ylabel("Settlement contribution")
        plt.tight_layout()
        plt.show()

        # 2) Expected remaining per tick (to see where expectation comes from)
        ft = out["future_times"]
        exp_tick = out["future_expected_per_tick"]
        plt.figure(figsize=(10, 4.5))
        plt.plot(ft, 100.0 * exp_tick, marker="o")  # convert to settlement-units per tick
        plt.title("Expected remaining contribution per 15-min tick (100×E[strangle|HH:MM])")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return out

    def plot_noon_sine_fit_and_prediction(self, zz: pd.DataFrame, target_time: Optional[pd.Timestamp] = None) -> Dict[str, object]:
        fit = self.fit_noon_sine_and_predict(zz, target_time=target_time)
        if not fit.get("ok", False):
            print(fit)
            return fit

        df_noon = fit["noon_points"]
        t_fit, y_fit = fit["fit_curve"]
        target_time = fit["target_time"]
        predicted = fit["predicted_level"]

        plt.figure(figsize=(10, 4.5))
        plt.plot(df_noon["time"], df_noon["level"], marker="o", linestyle="None", label="12:00 observations")
        plt.plot(t_fit, y_fit, label="Fitted sine")
        plt.scatter([target_time], [predicted], marker="*", s=180, label=f"Prediction: {predicted:.3f} mAOD")
        plt.axvline(target_time, linestyle="--")
        plt.title("Noon level sine fit + prediction (TIDE_SPOT-style)")
        plt.ylabel("Level (mAOD)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.show()

        return fit


# ==========================
# Example usage
# ==========================
if __name__ == "__main__":
    tp = ThamesPredictor()

    zz = tp.get_thames(limit=1450)

    # 1) Plot z-scores for today's level since 12:00
    zdf = tp.plot_zscore_today_since_noon(zz)
    print(zdf.tail(10).to_string(index=False))

    # 2) Plot historical noon settlements
    daily = tp.plot_daily_noon_settlements(zz, last_n=20)
    print(daily.tail(10).to_string(index=False))

    # 3) Plot expected settlement breakdown for next noon
    exp = tp.plot_expected_settlement_breakdown(zz)
    print({k: exp[k] for k in ["now", "next_noon", "realised_component", "expected_remaining_component", "expected_settlement", "history_coverage"]})

    # 4) Plot noon sine fit + prediction
    fit = tp.plot_noon_sine_fit_and_prediction(zz)
    if fit.get("ok", False):
        print(fit["params"])
        print("Predicted noon level:", fit["predicted_level"])
        print("Predicted |level| mm:", fit["predicted_abs_mm"])