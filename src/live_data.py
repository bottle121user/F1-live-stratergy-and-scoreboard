"""
live_data.py
Fetches live and recent F1 data for the scoreboard tab.

Sources:
  - Jolpica API (Ergast replacement) — championship standings, race results
    https://api.jolpi.ca/ergast/f1/
  - OpenF1 API — live session data, positions
    https://api.openf1.org/v1/
"""
from __future__ import annotations

import requests
from datetime import datetime, timezone

_JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
_OPENF1_BASE  = "https://api.openf1.org/v1"
_TIMEOUT      = 8   # seconds


# ── Small request helper ──────────────────────────────────────────────────────
def _get(url: str, params: dict | None = None) -> dict | list | None:
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ── Championship Standings ────────────────────────────────────────────────────
def get_driver_standings(season: str = "current") -> tuple[list[dict], str]:
    """
    Return driver championship standings.
    Falls back to previous year if current season has no data yet.

    Returns (standings_list, season_label)
    """
    def _fetch(s: str) -> list[dict]:
        data = _get(f"{_JOLPICA_BASE}/{s}/driverStandings.json")
        if data is None:
            return []
        try:
            standings_list = data["MRData"]["StandingsTable"]["StandingsLists"]
            if not standings_list:
                return []
            rows = []
            for s_ in standings_list[0]["DriverStandings"]:
                drv = s_["Driver"]
                rows.append({
                    "pos":         int(s_["position"]),
                    "driver_code": drv.get("code", drv["driverId"].upper()[:3]),
                    "driver_name": f"{drv['givenName']} {drv['familyName']}",
                    "nationality": drv.get("nationality", "—"),
                    "team":        s_["Constructors"][0]["name"] if s_.get("Constructors") else "—",
                    "points":      float(s_["points"]),
                    "wins":        int(s_["wins"]),
                })
            return rows
        except (KeyError, IndexError, TypeError):
            return []

    rows = _fetch(season)
    if rows:
        return rows, str(season)
    # Pre-season fallback: try previous calendar year
    prev = str(int(datetime.now(timezone.utc).year) - 1)
    rows = _fetch(prev)
    return rows, f"{prev} (final — new season not started)"


def get_constructor_standings(season: str = "current") -> tuple[list[dict], str]:
    """
    Return constructor championship standings.
    Falls back to previous year if current season has no data yet.

    Returns (standings_list, season_label)
    """
    def _fetch(s: str) -> list[dict]:
        data = _get(f"{_JOLPICA_BASE}/{s}/constructorStandings.json")
        if data is None:
            return []
        try:
            standings_list = data["MRData"]["StandingsTable"]["StandingsLists"]
            if not standings_list:
                return []
            rows = []
            for s_ in standings_list[0]["ConstructorStandings"]:
                con = s_["Constructor"]
                rows.append({
                    "pos":         int(s_["position"]),
                    "team":        con["name"],
                    "nationality": con.get("nationality", "—"),
                    "points":      float(s_["points"]),
                    "wins":        int(s_["wins"]),
                })
            return rows
        except (KeyError, IndexError, TypeError):
            return []

    rows = _fetch(season)
    if rows:
        return rows, str(season)
    prev = str(int(datetime.now(timezone.utc).year) - 1)
    rows = _fetch(prev)
    return rows, f"{prev} (final — new season not started)"


# ── Last Race Results ─────────────────────────────────────────────────────────
def get_last_race_results() -> dict:
    """
    Return the most recent race result.

    Returns:
        {
            "race_name": str,
            "circuit":   str,
            "date":      str,
            "results":   [{"pos", "driver_code", "driver_name", "team",
                            "grid", "laps", "status", "points", "fastest_lap"}]
        }
        or {} if unavailable.
    """
    data = _get(f"{_JOLPICA_BASE}/current/last/results.json")
    if data is None:
        return {}
    try:
        race = data["MRData"]["RaceTable"]["Races"][0]
        results = []
        for r in race.get("Results", []):
            drv = r["Driver"]
            fl  = r.get("FastestLap", {})
            fl_time = fl.get("Time", {}).get("time", "—")
            fl_rank = int(fl.get("rank", 0)) if fl else 0
            results.append({
                "pos":         int(r["position"]) if r["position"].isdigit() else 99,
                "driver_code": drv.get("code", drv["driverId"].upper()[:3]),
                "driver_name": f"{drv['givenName']} {drv['familyName']}",
                "team":        r["Constructor"]["name"],
                "grid":        int(r.get("grid", 0)),
                "laps":        int(r.get("laps", 0)),
                "status":      r.get("status", "—"),
                "points":      float(r.get("points", 0)),
                "fastest_lap": fl_time,
                "fl_rank":     fl_rank,
            })
        return {
            "race_name": race["raceName"],
            "circuit":   race["Circuit"]["circuitName"],
            "date":      race["date"],
            "results":   results,
        }
    except (KeyError, IndexError, TypeError):
        return {}




# ── Season Race Schedule ──────────────────────────────────────────────────────
def get_season_schedule(season: str = "current") -> list[dict]:
    """
    Return race schedule for the given season.

    Each item:
        {"round", "race_name", "circuit", "country", "date", "status"}
    where status is "done", "next", or "upcoming".
    """
    data = _get(f"{_JOLPICA_BASE}/{season}.json")
    if data is None:
        return []
    try:
        races = data["MRData"]["RaceTable"]["Races"]
        today = datetime.now(timezone.utc).date()
        next_flagged = False
        rows = []
        for r in races:
            race_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
            if race_date < today:
                status = "done"
            elif not next_flagged:
                status = "next"
                next_flagged = True
            else:
                status = "upcoming"
            rows.append({
                "round":      int(r["round"]),
                "race_name":  r["raceName"],
                "circuit":    r["Circuit"]["circuitName"],
                "country":    r["Circuit"]["Location"]["country"],
                "date":       r["date"],
                "status":     status,
            })
        return rows
    except (KeyError, IndexError, TypeError):
        return []


# ── Live Session Check (OpenF1) ───────────────────────────────────────────────
def get_live_session() -> dict | None:
    """
    Check whether there is a live or very recent F1 session right now.
    Only returns a session if it started within the last 4 hours
    (i.e., is currently live or just finished).
    Returns session info dict or None.
    """
    data = _get(f"{_OPENF1_BASE}/sessions", params={"session_type": "Race"})
    if not data or not isinstance(data, list):
        return None

    now = datetime.now(timezone.utc)
    # 4-hour window: session started within last 4 hours counts as "live"
    LIVE_WINDOW_HOURS = 4

    # Filter to sessions that started within the window
    recent = []
    for s in data:
        start_str = s.get("date_start")
        if not start_str:
            continue
        try:
            # OpenF1 dates are like "2025-11-23T13:00:00+00:00"
            start_dt = datetime.fromisoformat(start_str)
            # Make offset-aware if naive
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            delta_hours = (now - start_dt).total_seconds() / 3600
            if -1 <= delta_hours <= LIVE_WINDOW_HOURS:
                recent.append((start_dt, s))
        except ValueError:
            continue

    if not recent:
        return None

    # Take the latest
    _, latest = max(recent, key=lambda x: x[0])
    return {
        "session_key":  latest.get("session_key"),
        "meeting_name": latest.get("meeting_name", "—"),
        "circuit":      latest.get("circuit_short_name", "—"),
        "date_start":   latest.get("date_start"),
        "date_end":     latest.get("date_end"),
        "country":      latest.get("country_name", "—"),
    }


def get_live_positions(session_key: int) -> list[dict]:
    """
    Fetch the latest driver positions from a live/recent session.

    Returns list: [{"driver_number", "position", "date"}]
    """
    data = _get(
        f"{_OPENF1_BASE}/position",
        params={"session_key": session_key},
    )
    if not data or not isinstance(data, list):
        return []
    # Keep only the latest position per driver
    latest: dict[int, dict] = {}
    for entry in data:
        drv = entry.get("driver_number")
        if drv is None:
            continue
        if drv not in latest or entry["date"] > latest[drv]["date"]:
            latest[drv] = entry
    return sorted(latest.values(), key=lambda x: x.get("position", 99))


def get_live_drivers(session_key: int) -> dict[int, dict]:
    """
    Fetch driver metadata (name, team, colour) for the session.
    Returns {driver_number: {...}}
    """
    data = _get(
        f"{_OPENF1_BASE}/drivers",
        params={"session_key": session_key},
    )
    if not data or not isinstance(data, list):
        return {}
    result = {}
    for d in data:
        num = d.get("driver_number")
        if num is not None:
            result[num] = {
                "code":        d.get("name_acronym", "???"),
                "full_name":   d.get("full_name", "—"),
                "team":        d.get("team_name", "—"),
                "team_colour": "#" + (d.get("team_colour") or "888888"),
            }
    return result
