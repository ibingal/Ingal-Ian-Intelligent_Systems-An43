import csv
import math
import os
import sys

CSV_FILE = "disaster_cases.csv"


# =====================================================================
# 1. CASE REPRESENTATION
# =====================================================================
class DisasterCase:

    PROBLEM_FIELDS = ["hazard", "setting", "signal_or_magnitude",
                      "flood_depth_m", "population_at_risk", "hours_before_impact"]
    SOLUTION_FIELDS = ["evacuees", "shelters", "relief_packs",
                       "rescue_teams", "medical_units", "strategy"]

    def __init__(self, name, problem, solution=None):
        self.name = name
        self.problem = problem          # dict of PROBLEM_FIELDS
        self.solution = solution or {}  # dict of SOLUTION_FIELDS

    def to_row(self):
        return {"name": self.name, **self.problem, **self.solution}

    @classmethod
    def from_row(cls, row):
        p = {
            "hazard": row["hazard"],
            "setting": row["setting"],
            "signal_or_magnitude": float(row["signal_or_magnitude"]),
            "flood_depth_m": float(row["flood_depth_m"]),
            "population_at_risk": int(row["population_at_risk"]),
            "hours_before_impact": int(row["hours_before_impact"]),
        }
        s = {
            "evacuees": int(row["evacuees"]),
            "shelters": int(row["shelters"]),
            "relief_packs": int(row["relief_packs"]),
            "rescue_teams": int(row["rescue_teams"]),
            "medical_units": int(row["medical_units"]),
            "strategy": row["strategy"],
        }
        return cls(row["name"], p, s)


# Starting case base (5 past events)
SEED_CASES = [
    DisasterCase("Typhoon Ursula 2019",
                 {"hazard": "typhoon", "setting": "coastal", "signal_or_magnitude": 3,
                  "flood_depth_m": 0.5, "population_at_risk": 80000, "hours_before_impact": 48},
                 {"evacuees": 16000, "shelters": 40, "relief_packs": 48000, "rescue_teams": 12,
                  "medical_units": 6, "strategy": "Pre-emptive evacuation of shoreline barangays"}),
    DisasterCase("Typhoon Odette 2021",
                 {"hazard": "typhoon", "setting": "urban", "signal_or_magnitude": 4,
                  "flood_depth_m": 1.2, "population_at_risk": 250000, "hours_before_impact": 36},
                 {"evacuees": 62000, "shelters": 130, "relief_packs": 186000, "rescue_teams": 40,
                  "medical_units": 20, "strategy": "Forced evacuation of low-lying zones; schools as shelters"}),
    DisasterCase("Typhoon Kristine 2024",
                 {"hazard": "typhoon", "setting": "urban", "signal_or_magnitude": 3,
                  "flood_depth_m": 2.0, "population_at_risk": 300000, "hours_before_impact": 24},
                 {"evacuees": 90000, "shelters": 180, "relief_packs": 270000, "rescue_teams": 70,
                  "medical_units": 28, "strategy": "Rooftop and boat rescue; vertical evacuation to upper floors"}),
    DisasterCase("Habagat Flood 2023",
                 {"hazard": "flood", "setting": "urban", "signal_or_magnitude": 0,
                  "flood_depth_m": 1.5, "population_at_risk": 150000, "hours_before_impact": 12},
                 {"evacuees": 35000, "shelters": 70, "relief_packs": 105000, "rescue_teams": 30,
                  "medical_units": 10, "strategy": "Boat rescue along riverbanks; gyms used as shelters"}),
    DisasterCase("Taal Volcano 2020",
                 {"hazard": "volcano", "setting": "rural", "signal_or_magnitude": 4,
                  "flood_depth_m": 0.0, "population_at_risk": 120000, "hours_before_impact": 6},
                 {"evacuees": 45000, "shelters": 90, "relief_packs": 135000, "rescue_teams": 20,
                  "medical_units": 15, "strategy": "Total evacuation of 14-km danger zone; N95 masks distributed"}),
]


class CaseBase:
    """Loads and saves cases from/to a CSV file."""

    def __init__(self, path=CSV_FILE):
        self.path = path
        if not os.path.exists(path):
            self.reset()
        self.cases = self.load()

    def reset(self):
        self.save_all(SEED_CASES)

    def load(self):
        with open(self.path, newline="", encoding="utf-8") as f:
            return [DisasterCase.from_row(r) for r in csv.DictReader(f)]

    def save_all(self, cases):
        fields = ["name"] + DisasterCase.PROBLEM_FIELDS + DisasterCase.SOLUTION_FIELDS
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for c in cases:
                w.writerow(c.to_row())

    def add(self, case):
        self.cases.append(case)
        self.save_all(self.cases)


# =====================================================================
# 2. SIMILARITY ASSESSMENT & RETRIEVAL
# =====================================================================
# How similar two hazard types are (1.0 = same)
HAZARD_SIM = {
    ("typhoon", "flood"): 0.7,    # typhoons often cause floods
    ("typhoon", "volcano"): 0.1,
    ("flood", "volcano"): 0.0,
}
# How similar two settings are
SETTING_SIM = {("urban", "coastal"): 0.5, ("urban", "rural"): 0.3, ("coastal", "rural"): 0.5}

# Gaussian "spread" for numeric features: a difference equal to the
# spread gives similarity of about 0.61; bigger differences drop fast.
SPREAD = {"signal_or_magnitude": 1.5, "flood_depth_m": 1.0,
          "population_at_risk": 150000, "hours_before_impact": 24}

WEIGHTS = {"hazard": 0.25, "setting": 0.15, "signal_or_magnitude": 0.15,
           "flood_depth_m": 0.20, "population_at_risk": 0.15, "hours_before_impact": 0.10}


def table_sim(table, a, b):
    if a == b:
        return 1.0
    return table.get((a, b), table.get((b, a), 0.0))


def gaussian_sim(a, b, spread):
    return math.exp(-((a - b) ** 2) / (2 * spread ** 2))


def local_sim(feature, a, b):
    if feature == "hazard":
        return table_sim(HAZARD_SIM, a, b)
    if feature == "setting":
        return table_sim(SETTING_SIM, a, b)
    return gaussian_sim(a, b, SPREAD[feature])


def similarity(new, old):
    """Global similarity = sum of weight x local similarity (weights add to 1)."""
    return sum(w * local_sim(f, new[f], old[f]) for f, w in WEIGHTS.items())


def retrieve(new_problem, case_base):
    scored = [(similarity(new_problem, c.problem), c) for c in case_base.cases]
    return sorted(scored, key=lambda t: t[0], reverse=True)


# =====================================================================
# 3. CASE ADAPTATION
# Step A: blend the TOP-2 cases. Each resource is converted to a
#         per-1,000-people rate, and the two rates are averaged using
#         the similarity scores as weights.
# Step B: multiply by the new population and by a severity index.
# Step C: pick the strategy of the best case, plus rule-based notes.
# =====================================================================
def severity_index(p):
    """Rough 0.8 - 1.4 multiplier from signal level, flood depth, and lead time."""
    s = 1.0
    s += 0.05 * max(0, p["signal_or_magnitude"] - 3)
    s += 0.10 * max(0, p["flood_depth_m"] - 1.0)
    if p["hours_before_impact"] < 24:
        s += 0.10                      # less time = more people stranded
    return round(min(max(s, 0.8), 1.4), 2)


def adapt(new_problem, top2):
    (s1, c1), (s2, c2) = top2
    w1, w2 = s1 / (s1 + s2), s2 / (s1 + s2)
    sev = severity_index(new_problem)
    pop_k = new_problem["population_at_risk"] / 1000
    notes = [f"Blend weights: {c1.name} = {w1:.2f}, {c2.name} = {w2:.2f}",
             f"Severity index = {sev}"]

    adapted = {}
    for field in ["evacuees", "shelters", "relief_packs", "rescue_teams", "medical_units"]:
        rate1 = c1.solution[field] / (c1.problem["population_at_risk"] / 1000)
        rate2 = c2.solution[field] / (c2.problem["population_at_risk"] / 1000)
        blended_rate = w1 * rate1 + w2 * rate2
        adapted[field] = math.ceil(round(blended_rate * pop_k * sev, 6))
        notes.append(f"{field:<14} rate/1k: {rate1:7.2f} & {rate2:7.2f} -> "
                     f"{blended_rate:7.2f} x {pop_k:.0f}k x {sev} = {adapted[field]:,}")

    # Strategy adaptation rules
    strategy = c1.solution["strategy"]
    extra = []
    low = strategy.lower()               # avoid repeating what is already there
    if new_problem["flood_depth_m"] >= 1.5 and "boat" not in low:
        extra.append("deploy rubber boats to flood-prone streets")
    if new_problem["hours_before_impact"] < 24:
        extra.append("start evacuation IMMEDIATELY (less than 24 hrs lead time)")
    if new_problem["setting"] == "urban" and "school" not in low:
        extra.append("open schools and covered courts as shelters")
    if extra:
        strategy += "; " + "; ".join(extra)
    adapted["strategy"] = strategy
    return adapted, notes


# =====================================================================
# 4. LEARNING (RETAIN)
# =====================================================================
def retain(case_base, name, problem, solution):
    case_base.add(DisasterCase(name, problem, solution))


# =====================================================================
# Display helpers and input
# =====================================================================
def hr(ch="-"):
    print(ch * 76)


def show_cases(case_base):
    print(f"   {'#':<3}{'Event':<26}{'Hazard':<9}{'Setting':<9}{'Depth':>6}{'Pop. at risk':>14}")
    for i, c in enumerate(case_base.cases, 1):
        p = c.problem
        print(f"   {i:<3}{c.name[:25]:<26}{p['hazard']:<9}{p['setting']:<9}"
              f"{p['flood_depth_m']:>5}m{p['population_at_risk']:>14,}")


def demo_event():
    return "Incoming Urban Typhoon", {
        "hazard": "typhoon", "setting": "urban", "signal_or_magnitude": 4,
        "flood_depth_m": 1.8, "population_at_risk": 200000, "hours_before_impact": 18}


def typed_event():
    print("Describe the new event:")
    name = input("   Event name: ") or "New Event"
    p = {"hazard": input("   Hazard (typhoon/flood/volcano): ").strip().lower(),
         "setting": input("   Setting (urban/coastal/rural): ").strip().lower(),
         "signal_or_magnitude": float(input("   Signal no. / alert level: ")),
         "flood_depth_m": float(input("   Expected flood depth (m): ")),
         "population_at_risk": int(input("   Population at risk: ")),
         "hours_before_impact": int(input("   Hours before impact: "))}
    return name, p


# =====================================================================
# MAIN - CBR cycle
# =====================================================================
def main():
    cb = CaseBase()
    if "--reset" in sys.argv:
        cb.reset()
        print("Case base reset to 5 seed cases.")
        return

    hr("=")
    print("  DISASTER RESPONSE DSS  -  Case-Based Reasoning (CDRRMO planning tool)")
    hr("=")
    print(f"Case base loaded from {CSV_FILE}: {len(cb.cases)} cases")
    show_cases(cb)

    name, problem = typed_event() if "--new" in sys.argv else demo_event()
    hr()
    print(f"NEW PROBLEM: {name}")
    for k, v in problem.items():
        print(f"   {k:<22}: {v:,}" if isinstance(v, int) else f"   {k:<22}: {v}")

    # ---- Retrieve ----
    hr()
    print("SIMILARITY ASSESSMENT & RETRIEVAL")
    ranked = retrieve(problem, cb)
    for rank, (score, c) in enumerate(ranked, 1):
        print(f"   Rank {rank}: {c.name:<24} similarity = {score:.3f}")
    top2 = ranked[:2]
    print(f"   >> Retrieved: {top2[0][1].name} and {top2[1][1].name}")

    # ---- Adapt ----
    hr()
    print("CASE ADAPTATION (blend of top-2 cases)")
    solution, notes = adapt(problem, top2)
    for n in notes:
        print(f"   {n}")

    hr()
    print("ADAPTED FINAL SOLUTION")
    for k in DisasterCase.SOLUTION_FIELDS:
        v = solution[k]
        print(f"   {k:<14}: {v:,}" if isinstance(v, int) else f"   {k:<14}: {v}")

    # ---- Retain ----
    retain(cb, name, problem, solution)
    hr()
    print(f"LEARNING: New case saved -> '{name}' written to {CSV_FILE}")
    print(f"Updated case base ({len(cb.cases)} cases):")
    show_cases(cb)
    hr("=")


if __name__ == "__main__":
    main()
