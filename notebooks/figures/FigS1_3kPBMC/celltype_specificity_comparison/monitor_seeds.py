"""
Live progress dashboard for multi-seed runs.

Usage (in a separate terminal):
    python monitor_seeds.py [--dir scores/logs] [--seeds 0-7] [--interval 5]

Refreshes every N seconds showing per-seed step, timing, and ETA.
"""
import argparse, re, time, sys
from pathlib import Path
from datetime import datetime

# Estimated step durations in seconds (prior; updated from observed completed seeds)
STEP_COST_PRIOR = {
    "PCA":      30,
    "LOCAT":   900,
    "GSPA":    300,
    "LMD":    1200,
    "Hotspot": 180,
    "Haystack": 60,
}
STEPS = ["PCA", "LOCAT", "GSPA", "LMD", "Hotspot", "Haystack"]

parser = argparse.ArgumentParser()
parser.add_argument("--dir",      default="scores/logs", help="Log directory")
parser.add_argument("--seeds",    default="0-7",         help="Seed range, e.g. 0-7 or 0,1,4")
parser.add_argument("--interval", type=int, default=5,   help="Refresh interval in seconds")
args = parser.parse_args()

# Parse seed list
if "-" in args.seeds and "," not in args.seeds:
    lo, hi = map(int, args.seeds.split("-"))
    SEEDS = list(range(lo, hi + 1))
else:
    SEEDS = [int(x) for x in args.seeds.split(",")]

LOG_DIR = Path(args.dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── ANSI helpers ─────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"

def fmt_dur(sec):
    if sec is None or sec < 0:
        return "  ─   "
    m, s = divmod(int(sec), 60)
    h, m2 = divmod(m, 60)
    if h:
        return f"{h}h{m2:02d}m"
    return f"{m}m{s:02d}s"

# ── Log parser ────────────────────────────────────────────────────────────────
def parse_log(path):
    state = {
        "seed": None, "total_start": None,
        "steps_done": {},   # name -> duration_sec
        "current_step": None,  # (step_num, name, start_ts)
        "done": False, "total_dur": None, "error": None,
    }
    if not path.exists():
        return state
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return state

    for line in text.splitlines():
        m = re.match(r"STEP_START\|ts=(\d+)\|seed=(\d+)\|step=(\d+)\|name=(\w+)", line)
        if m:
            ts, seed, step_num, name = int(m[1]), int(m[2]), int(m[3]), m[4]
            state["seed"] = seed
            if state["total_start"] is None:
                state["total_start"] = ts
            state["current_step"] = (step_num, name, ts)
            continue

        m = re.match(r"STEP_END\|ts=(\d+)\|seed=(\d+)\|step=(\d+)\|name=(\w+)\|duration=(\d+)", line)
        if m:
            name, dur = m[4], int(m[5])
            state["steps_done"][name] = dur
            state["current_step"] = None
            continue

        m = re.match(r"DONE\|ts=(\d+)\|seed=(\d+)\|total=(\d+)", line)
        if m:
            state["done"] = True
            state["total_dur"] = int(m[3])
            continue

        m = re.match(r"FAILED\|.*\|error=(.*)", line)
        if m:
            state["error"] = m[1][:50]

    return state

def eta_for_seed(state, now_ts, step_costs):
    if state["done"]:
        return 0
    remaining = 0
    in_progress_name = None
    if state["current_step"]:
        _, name, start_ts = state["current_step"]
        in_progress_name = name
        elapsed_in_step = now_ts - start_ts
        expected = step_costs.get(name, STEP_COST_PRIOR.get(name, 300))
        remaining += max(0, expected - elapsed_in_step)
    for step in STEPS:
        if step not in state["steps_done"] and step != in_progress_name:
            remaining += step_costs.get(step, STEP_COST_PRIOR.get(step, 300))
    return remaining

def infer_step_costs(all_states):
    """Update step cost estimates from completed seeds."""
    costs = dict(STEP_COST_PRIOR)
    observations = {s: [] for s in STEPS}
    for st in all_states.values():
        for name, dur in st["steps_done"].items():
            if name in observations:
                observations[name].append(dur)
    for name, obs in observations.items():
        if obs:
            costs[name] = int(sum(obs) / len(obs))
    return costs

# ── Render ────────────────────────────────────────────────────────────────────
def render(all_states, step_costs, now_ts, refresh_count):
    W = 80
    lines = []
    ts_str = datetime.now().strftime("%H:%M:%S")

    lines.append(f"{BOLD}3k PBMC  Multi-seed Progress{RESET}   {DIM}updated {ts_str}  (refresh #{refresh_count}){RESET}")
    lines.append("")

    col = f"{'Seed':>4}  {'Status':<9}  {'Current step':<20}  {'Step':>6}  {'Total':>6}  {'ETA':>7}"
    lines.append(f"{BOLD}{col}{RESET}")
    lines.append("─" * len(col))

    n_done = 0
    n_error = 0
    eta_list = []

    for seed in SEEDS:
        st = all_states[seed]

        if st["error"]:
            n_error += 1
            status  = f"{RED}ERROR{RESET}    "
            step_s  = st["error"][:20]
            step_t  = ""
            total_t = ""
            eta_s   = ""
        elif not st["total_start"] and not st["done"]:
            status  = f"{DIM}waiting{RESET}  "
            step_s  = "─"
            step_t  = ""
            total_t = ""
            eta_s   = f"~{fmt_dur(sum(step_costs.values()))}"
            eta_list.append(sum(step_costs.values()))
        elif st["done"]:
            n_done += 1
            status  = f"{GREEN}done ✓{RESET}   "
            n_steps = len(STEPS)
            step_s  = f"all {n_steps} steps"
            step_t  = ""
            total_t = fmt_dur(st["total_dur"])
            eta_s   = f"{DIM}─{RESET}"
        else:
            status = f"{YELLOW}running{RESET}  "
            if st["current_step"]:
                num, name, start_ts = st["current_step"]
                elapsed_step = now_ts - start_ts
                step_s = f"[{num}/6] {name}"
                step_t = fmt_dur(elapsed_step)
            else:
                step_s = "starting..."
                step_t = ""
            total_elapsed = now_ts - st["total_start"] if st["total_start"] else 0
            total_t = fmt_dur(total_elapsed)
            eta = eta_for_seed(st, now_ts, step_costs)
            eta_list.append(eta)
            eta_s = f"~{fmt_dur(eta)}"

        sid = st["seed"] if st["seed"] is not None else seed
        lines.append(
            f"{sid:>4}  {status:<9}  {step_s:<20}  {step_t:>6}  {total_t:>6}  {eta_s:>7}"
        )

    lines.append("─" * len(col))

    # Overall summary
    n_running = len(SEEDS) - n_done - n_error - sum(
        1 for s in SEEDS if not all_states[s]["total_start"] and not all_states[s]["done"]
    )
    wall_eta = max(eta_list) if eta_list else 0  # parallel → bottleneck is slowest
    lines.append(
        f"  {n_done}/{len(SEEDS)} done"
        + (f"  {n_error} failed" if n_error else "")
        + f"   Wall ETA: ~{fmt_dur(wall_eta)}"
        + f"   (step costs: LOCAT={fmt_dur(step_costs['LOCAT'])} "
        + f"LMD={fmt_dur(step_costs['LMD'])} GSPA={fmt_dur(step_costs['GSPA'])})"
    )
    lines.append("")
    lines.append(f"  {DIM}Ctrl-C to exit monitor (jobs continue in background){RESET}")

    return "\n".join(lines), n_done == len(SEEDS)

# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("\033[?25l", end="", flush=True)   # hide cursor
    try:
        refresh_count = 0
        while True:
            now_ts = int(time.time())
            all_states = {s: parse_log(LOG_DIR / f"seed_{s}.log") for s in SEEDS}
            step_costs = infer_step_costs(all_states)

            dashboard, all_done = render(all_states, step_costs, now_ts, refresh_count)

            print("\033[H\033[J" + dashboard, end="", flush=True)  # clear + redraw
            refresh_count += 1

            if all_done:
                print(f"\n\n{GREEN}{BOLD}All {len(SEEDS)} seeds complete!{RESET}\n", flush=True)
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h", end="", flush=True)   # restore cursor
        print()

if __name__ == "__main__":
    main()
