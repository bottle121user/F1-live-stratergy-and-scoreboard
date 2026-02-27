"""
exhaustive_test.py
Runs every meaningful input combination for all 4 tabs programmatically
(bypasses Streamlit UI) and writes all inputs + outputs to test_results.txt.
"""
import sys, os, io, textwrap
from pathlib import Path
from datetime import datetime

# ── path setup ────────────────────────────────────────────────────────────────
SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))

from simulator import CIRCUITS, simulate_strategy, compare_strategies, HISTORICAL_METRICS_CACHE
from predict   import predict_pit, recommend_strategy
import live_data

OUT_FILE = Path(__file__).parent / "test_results.txt"

# ── helpers ───────────────────────────────────────────────────────────────────
lines = []

def h(title):
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  {title}")
    lines.append("=" * 80)

def sub(title):
    lines.append(f"\n--- {title} ---")

def row(*args):
    lines.append("  " + " | ".join(str(a) for a in args))

def sep():
    lines.append("  " + "-" * 76)

# ── constants ─────────────────────────────────────────────────────────────────
ALL_CIRCUITS  = list(CIRCUITS.keys())
ALL_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
ALL_TEAMS     = sorted(HISTORICAL_METRICS_CACHE.get("team_pace_offsets", {}).keys()) or \
                ["McLaren", "Red Bull Racing", "Ferrari", "Mercedes", "Aston Martin"]
ALL_DRIVERS   = sorted(HISTORICAL_METRICS_CACHE.get("driver_tyre_factors", {}).keys()) or \
                ["NOR", "VER", "LEC", "HAM", "ALO"]

CIRCUIT_LIST  = ALL_CIRCUITS   # for track_encoded

print(f"Testing {len(ALL_CIRCUITS)} circuits | {len(ALL_COMPOUNDS)} compounds | "
      f"{len(ALL_TEAMS)} teams | {len(ALL_DRIVERS)} drivers")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PIT DECISION
# ALL circuits × ALL compounds × key edge cases
# ══════════════════════════════════════════════════════════════════════════════
h("TAB 1: LIVE PIT DECISION — All Circuits × All Compounds × Edge Cases")
row("Circuit", "Compound", "Lap", "LapsSincePit", "LapsRem", "SC", "Rain",
    "Team", "Driver", "Decision", "PitProb%", "StayProb%", "Confidence%")
sep()

EDGE_CASES = [
    # (lap, laps_since_pit, laps_remaining, is_sc, rain, pos, stint)  label
    (1,  1,  79, 0, 0.0, 5,  1, "lap1_fresh_tyre"),
    (20, 10, 50, 0, 0.0, 8,  1, "mid_race_normal"),
    (30, 20, 30, 0, 0.0, 3,  1, "deg_building"),
    (40, 30, 20, 0, 0.0, 12, 2, "high_deg"),
    (50, 40, 10, 0, 0.0, 1,  2, "leader_late"),
    (70, 50, 5,  0, 0.0, 6,  2, "near_end_old_tyre"),
    (80, 55, 0,  0, 0.0, 15, 3, "last_lap"),
    (25, 5,  45, 1, 0.0, 4,  1, "sc_active"),
    (30, 15, 30, 0, 0.5, 8,  1, "wet_conditions"),
    (35, 10, 35, 0, 1.0, 7,  1, "heavy_rain"),
]

circuit_sample = ALL_CIRCUITS[:5] + [ALL_CIRCUITS[-1]]  # first 5 + last
team_sample    = ALL_TEAMS[:2]
driver_sample  = ALL_DRIVERS[:2]

for circuit in circuit_sample:
    track_enc = CIRCUIT_LIST.index(circuit) if circuit in CIRCUIT_LIST else 0
    n_laps    = CIRCUITS[circuit]["laps"]
    for compound in ALL_COMPOUNDS:
        for (lap, lps, rem, sc, rain, pos, stint, label) in EDGE_CASES:
            # Clamp to circuit lap count
            lap_c = min(lap, n_laps)
            rem_c = max(0, n_laps - lap_c)
            for team in team_sample:
                for driver in driver_sample:
                    try:
                        res = predict_pit(
                            lap_number       = lap_c,
                            laps_since_pit   = float(lps),
                            compound         = compound,
                            lap_time_seconds = 90.0,
                            lap_time_delta   = 0.5,
                            stint_number     = float(stint),
                            laps_remaining   = float(rem_c),
                            is_safety_car    = sc,
                            position         = float(pos),
                            air_temp         = 26.0,
                            track_temp       = 38.0,
                            rainfall         = rain,
                            humidity         = 55.0,
                            wind_speed       = 5.0,
                            track_encoded    = track_enc,
                            team             = team,
                            driver           = driver,
                            pit_threshold    = 0.35,
                        )
                        row(
                            circuit[:25].ljust(25),
                            compound.ljust(12),
                            str(lap_c).rjust(2),
                            str(lps).rjust(3),
                            str(rem_c).rjust(3),
                            "SC" if sc else "  ",
                            f"R{rain:.1f}",
                            team[:12].ljust(12),
                            driver[:5].ljust(5),
                            res["decision"].ljust(8),
                            f"{res['pit_probability']*100:5.1f}",
                            f"{res['stay_probability']*100:5.1f}",
                            f"{res['confidence']*100:5.1f}",
                        )
                    except Exception as e:
                        row(circuit[:25], compound, lap_c, lps, rem_c, sc, rain,
                            team, driver, f"ERROR: {e}", "", "", "")

# Full circuits × all compounds at "typical" race state (lap 30)
sub("ALL 24 Circuits — typical mid-race (Lap 30, 14 laps on SOFT, first driver/team)")
row("Circuit", "Compound", "Decision", "PitProb%", "Confidence%")
sep()
t = ALL_TEAMS[0]
d = ALL_DRIVERS[0]
for circuit in ALL_CIRCUITS:
    track_enc = CIRCUIT_LIST.index(circuit) if circuit in CIRCUIT_LIST else 0
    n = CIRCUITS[circuit]["laps"]
    for compound in ALL_COMPOUNDS:
        try:
            res = predict_pit(
                lap_number=min(30, n), laps_since_pit=14.0, compound=compound,
                lap_time_seconds=90.0, lap_time_delta=0.5, stint_number=1.0,
                laps_remaining=float(max(0, n-30)), is_safety_car=0,
                position=8.0, air_temp=26.0, track_temp=38.0,
                rainfall=0.0, humidity=55.0, wind_speed=5.0,
                track_encoded=track_enc, team=t, driver=d, pit_threshold=0.35,
            )
            row(circuit[:30].ljust(30), compound.ljust(12),
                res["decision"].ljust(8),
                f"{res['pit_probability']*100:5.1f}",
                f"{res['confidence']*100:5.1f}")
        except Exception as e:
            row(circuit[:30], compound, f"ERROR: {e}", "", "")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STRATEGY RECOMMENDER
# All circuits × all starting compounds, dry + wet
# ══════════════════════════════════════════════════════════════════════════════
h("TAB 2: STRATEGY RECOMMENDER — All Circuits × All Starting Compounds")
row("Circuit", "StartCpd", "Rain", "Rank1 Label", "Rank1 Time(s)",
    "Rank1 Pit Schedule", "Rank2 Label", "Rank2 Time(s)")
sep()

for circuit in ALL_CIRCUITS:
    for compound in ALL_COMPOUNDS:
        for rain in [0.0, 0.8]:  # dry and wet
            try:
                ranked = recommend_strategy(
                    circuit, weather={"track_temp": 38.0, "rainfall": rain},
                    top_n=3, team=ALL_TEAMS[0], driver=ALL_DRIVERS[0],
                    starting_compound=compound,
                )
                if not ranked:
                    row(circuit[:25], compound, f"R{rain}", "NO RESULTS", "", "", "", "")
                    continue
                r1 = ranked[0]
                r2 = ranked[1] if len(ranked) > 1 else {}
                pits1 = ", ".join(f"L{p['pit_lap']}->{p['compound'][:1]}" for p in r1["strategy"]) or "no-stop"
                row(
                    circuit[:25].ljust(25),
                    compound.ljust(12),
                    f"R{rain:.1f}",
                    r1["label"][:18].ljust(18),
                    f"{r1['total_time']:,.0f}",
                    pits1[:30].ljust(30),
                    r2.get("label", "")[:18],
                    f"{r2.get('total_time',0):,.0f}" if r2 else "",
                )
            except Exception as e:
                row(circuit[:25], compound, f"R{rain}", f"ERROR: {e}", "", "", "", "")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LAP SIMULATOR
# All circuits × all starting compounds, dry + wet
# ══════════════════════════════════════════════════════════════════════════════
h("TAB 3: LAP SIMULATOR — All Circuits × 4 Strategies × Dry/Wet")
row("Circuit", "StartCpd", "Rain", "Strategy", "TotalTime(s)", "PitLaps", "FastestLap(s)")
sep()

STRAT_DEFS = [
    ("No-stop",       []),
    ("1-stop early",  "early"),
    ("1-stop late",   "late"),
    ("2-stop",        "two"),
]

for circuit in ALL_CIRCUITS:
    n = CIRCUITS[circuit]["laps"]

    def _pl(f): return max(2, min(n - 3, round(n * f)))

    strats = [
        ("No-stop",      []),
        ("1-stop early", [{"pit_lap": _pl(0.35), "compound": "HARD"}]),
        ("1-stop late",  [{"pit_lap": _pl(0.50), "compound": "HARD"}]),
        ("2-stop",       [{"pit_lap": _pl(0.28), "compound": "MEDIUM"},
                          {"pit_lap": _pl(0.58), "compound": "HARD"}]),
    ]
    for compound in ["SOFT", "MEDIUM", "HARD"]:   # skip INT/WET for simulator (physics sim)
        for rain in [0.0, 0.8]:
            weather = {"track_temp": 38.0, "rainfall": rain}
            for label, strat in strats:
                try:
                    r = simulate_strategy(circuit, strat, weather,
                                          team=ALL_TEAMS[0], driver=ALL_DRIVERS[0],
                                          starting_compound=compound)
                    pit_str = ",".join(str(p) for p in r.pit_laps) or "none"
                    fastest = min(lr.time for lr in r.lap_records) if r.lap_records else 0
                    row(
                        circuit[:25].ljust(25),
                        compound.ljust(6),
                        f"R{rain:.1f}",
                        label[:14].ljust(14),
                        f"{r.total_time:,.1f}",
                        pit_str[:20].ljust(20),
                        f"{fastest:.3f}",
                    )
                except Exception as e:
                    row(circuit[:25], compound, f"R{rain}", label, f"ERROR: {e}", "", "")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE SCOREBOARD
# ══════════════════════════════════════════════════════════════════════════════
h("TAB 4: LIVE SCOREBOARD — API Responses")

sub("Driver Standings")
drv, drv_szn = live_data.get_driver_standings()
row(f"Season: {drv_szn}  |  Drivers returned: {len(drv)}")
if drv:
    row("Pos", "Code", "Driver", "Team", "Points", "Wins")
    sep()
    for d in drv:
        row(d["pos"], d["driver_code"], d["driver_name"][:20], d["team"][:20],
            d["points"], d["wins"])

sub("Constructor Standings")
con, con_szn = live_data.get_constructor_standings()
row(f"Season: {con_szn}  |  Teams returned: {len(con)}")
if con:
    row("Pos", "Team", "Points", "Wins")
    sep()
    for c in con:
        row(c["pos"], c["team"][:30], c["points"], c["wins"])

sub("Last Race Results")
last = live_data.get_last_race_results()
if last:
    row(f"Race: {last.get('race_name')} | Circuit: {last.get('circuit')} | Date: {last.get('date')}")
    results = last.get("results", [])
    row(f"Finishers: {len(results)}")
    if results:
        row("Pos", "Code", "Driver", "Team", "Grid", "Laps", "Status", "Points")
        sep()
        for r in results[:10]:
            row(r["pos"], r["driver_code"], r["driver_name"][:20], r["team"][:20],
                r["grid"], r["laps"], r["status"][:15], r["points"])
        if len(results) > 10:
            row(f"... and {len(results)-10} more finishers")
else:
    row("No last race data available (pre-season or API offline)")

sub("Season Calendar")
sched = live_data.get_season_schedule()
row(f"Rounds returned: {len(sched)}")
if sched:
    row("Rd", "Race", "Country", "Date", "Status")
    sep()
    for r in sched:
        row(r["round"], r["race_name"][:30], r["country"][:15], r["date"], r["status"])

sub("Live Session Check (OpenF1)")
sess = live_data.get_live_session()
if sess:
    row(f"LIVE: {sess['meeting_name']} at {sess['circuit']}, {sess['country']}")
    row(f"Session key: {sess['session_key']}  Start: {sess['date_start']}")
    pos = live_data.get_live_positions(sess["session_key"])
    drvs = live_data.get_live_drivers(sess["session_key"])
    row(f"Live positions returned: {len(pos)}")
    if pos:
        row("Pos", "#", "Code", "Driver", "Team")
        sep()
        for p in pos[:10]:
            info = drvs.get(p["driver_number"], {})
            row(p["position"], p["driver_number"], info.get("code","?"),
                info.get("full_name","?")[:20], info.get("team","?")[:20])
else:
    row("No live session active right now (outside race weekend window)")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
h("TEST SUMMARY")
errors = [l for l in lines if "ERROR:" in l]
row(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
row(f"Total output lines: {len(lines)}")
row(f"Lines with errors:  {len(errors)}")
if errors:
    sub("Error lines")
    for e in errors:
        lines.append("  " + e.strip())

# ── write file ────────────────────────────────────────────────────────────────
OUT_FILE.write_text("\n".join(lines), encoding="utf-8")
print(f"\n✅ Done — {len(lines)} lines written to {OUT_FILE}")
print(f"   Errors found: {len(errors)}")
