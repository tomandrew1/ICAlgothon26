import requests
import time

EXCHANGE_URL = "http://ec2-52-19-74-159.eu-west-1.compute.amazonaws.com"

def ping_server(base_url: str = EXCHANGE_URL, timeout: float = 5.0) -> dict:
    """
    Ping the exchange server and return a structured result.
    Tries a few likely health endpoints; falls back to the base URL.
    """
    candidates = ["/api/health", "/health", "/api/ping", "/ping", "/"]
    last_err = None

    for path in candidates:
        url = base_url.rstrip("/") + path
        try:
            t0 = time.perf_counter()
            r = requests.get(url, timeout=timeout)
            dt_ms = (time.perf_counter() - t0) * 1000

            return {
                "ok": True,
                "url": url,
                "status_code": r.status_code,
                "latency_ms": round(dt_ms, 1),
                "content_type": r.headers.get("content-type", ""),
                "text_sample": r.text[:300],
            }
        except Exception as e:
            last_err = e

    return {"ok": False, "error": repr(last_err)}

if __name__ == "__main__":
    print(ping_server())