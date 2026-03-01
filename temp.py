from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd


@dataclass(frozen=True)
class LondonTHSettlement:
    """
    Settlement for: round(Fahrenheit temperature) * humidity
    using a 15-min weather DataFrame like your get_weather() returns.

    Required columns: time (tz-aware), temperature (°C), humidity (%).
    """

    tz_name: str = "Europe/London"

    @staticmethod
    def _require_cols(df: pd.DataFrame, cols: list[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Have: {list(df.columns)}")

    @staticmethod
    def c_to_f(c: float) -> float:
        return (9.0 / 5.0) * c + 32.0

    @staticmethod
    def round_fahrenheit(f: float) -> int:
        # Python round uses bankers rounding; for weather settlement it's usually "nearest int"
        # If you want .5 to always round up, replace with: int(f + 0.5) for f>=0.
        return int(round(f))

    def settlement_value(self, temp_c: float, humidity: float) -> int:
        f = self.c_to_f(float(temp_c))
        f_r = self.round_fahrenheit(f)
        return int(f_r * float(humidity))

    def at_time(
        self,
        df_weather: pd.DataFrame,
        when: str | datetime,
        *,
        method: str = "nearest",
        tolerance: str | pd.Timedelta = "7min",
        return_row: bool = False,
    ):
        """
        Compute settlement at a specific time.

        method:
          - "nearest": use nearest 15-min row within tolerance
          - "ffill": use last row at or before time
          - "exact": require exact timestamp match

        tolerance used only for "nearest".
        """
        self._require_cols(df_weather, ["time", "temperature", "humidity"])

        df = df_weather.copy()

        # Ensure tz-aware and aligned
        tz = ZoneInfo(self.tz_name)
        t = pd.Timestamp(when)
        if t.tzinfo is None:
            t = t.tz_localize(tz)
        else:
            t = t.tz_convert(tz)

        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize(tz)
        else:
            df["time"] = df["time"].dt.tz_convert(tz)

        df = df.sort_values("time").reset_index(drop=True)

        if method == "exact":
            hit = df[df["time"] == t]
            if hit.empty:
                raise ValueError(f"No exact match for {t}.")
            row = hit.iloc[0]

        elif method == "ffill":
            hit = df[df["time"] <= t]
            if hit.empty:
                raise ValueError(f"No data at or before {t}.")
            row = hit.iloc[-1]

        elif method == "nearest":
            tol = pd.Timedelta(tolerance) if isinstance(tolerance, str) else tolerance
            # find nearest index
            idx = (df["time"] - t).abs().idxmin()
            row = df.loc[idx]
            if abs(row["time"] - t) > tol:
                raise ValueError(f"Nearest point {row['time']} is outside tolerance {tol} for target {t}.")
        else:
            raise ValueError("method must be one of: 'nearest', 'ffill', 'exact'")

        value = self.settlement_value(row["temperature"], row["humidity"])
        if return_row:
            out = row.copy()
            out["temp_f"] = self.c_to_f(row["temperature"])
            out["temp_f_rounded"] = self.round_fahrenheit(out["temp_f"])
            out["settlement"] = value
            return out
        return value

    def today_noon(
        self,
        df_weather: pd.DataFrame,
        *,
        method: str = "nearest",
        tolerance: str | pd.Timedelta = "7min",
        return_row: bool = True,
    ):
        """
        Settlement at 12:00 local time today (Europe/London).
        """
        tz = ZoneInfo(self.tz_name)
        now = datetime.now(tz)
        noon = datetime(now.year, now.month, now.day, 12, 0, 0, tzinfo=tz)
        return self.at_time(df_weather, noon, method=method, tolerance=tolerance, return_row=return_row)


# ---- Example usage with your df_weather ----
# s = LondonTHSettlement()
# row = s.today_noon(df_weather, method="nearest", tolerance="7min", return_row=True)
# print(row[["time", "temperature", "humidity", "temp_f", "temp_f_rounded", "settlement"]])