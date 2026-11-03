medical_facts = {
    
    "has_fever": True,
    "has_cough": True,
    "shortness_of_breath": False,
    
        "heart_rate": 90,            
    
            "duration_of_symptoms": 5,     
}


def display_facts(facts):
    """Print the knowledge base so the input is visible in the output."""
    for key, value in facts.items():
        print(f"  {key:<22}: {value}")


#  RBR

def rule_cold_or_flu(facts):
    if (facts["has_fever"] and facts["has_cough"]
            and not facts["shortness_of_breath"]
            and facts["duration_of_symptoms"] < 7):
        return "You may have a common cold or flu. Rest and stay hydrated."


# IF symptoms have lasted 7 days or more THEN see a doctor
def rule_long_duration(facts):
    if facts["duration_of_symptoms"] >= 7:
        return "Your symptoms have lasted a week or more. Please see a doctor."


#IF shortness of breath THEN seek medical attention immediately
def rule_breathing(facts):
    if facts["shortness_of_breath"]:
        return "Shortness of breath is concerning. Please seek medical attention immediately."


#IF heart rate is above 100 bpm THEN have it checked
def rule_high_heart_rate(facts):
    if facts["heart_rate"] > 100:
        return "Your heart rate is high. Go to the doctor."


#IF fever AND no cough THEN monitor temperature
def rule_fever_only(facts):
    if facts["has_fever"] and not facts["has_cough"]:
        return "Fever without cough: monitor your temperature and drink fluids."


# All rules that the inference engine will check, in order
rules = [
    rule_cold_or_flu,
    rule_long_duration,
    rule_breathing,
    rule_high_heart_rate,
    rule_fever_only,
]


def inference_engine(facts, rules):
    actions = []
    for rule in rules:
        result = rule(facts)
        if result:                         
            print(f"  [fired]     {rule.__name__}")
            actions.append(result)
        else:
            print(f"  [not fired] {rule.__name__}")

    if not actions:
        actions.append("Your symptoms are not specific enough for a diagnosis. "
                       "Monitor your health and consult a doctor if necessary.")
    return actions


def run_rbr(facts):
    print("-" * 65)
    print("PART 2 - RULE-BASED REASONING (RBR)")
    print("-" * 65)
    print("Input (working memory):")
    display_facts(facts)

    print("\nProcess (checking each rule):")
    actions = inference_engine(facts, rules)

    print("\nRule-Based Reasoning Actions:")
    for action in actions:
        print("-", action)
    print()
    return actions



#(CBR)

case_base = [
    {"problem": {"has_fever": True, "has_cough": True, "shortness_of_breath": False,
                 "heart_rate": 85, "duration_of_symptoms": 4},
     "solution": "Common cold / flu: rest, drink fluids, take paracetamol for fever."},

    {"problem": {"has_fever": True, "has_cough": True, "shortness_of_breath": True,
                 "heart_rate": 110, "duration_of_symptoms": 6},
     "solution": "Possible pneumonia: see a doctor immediately for a chest X-ray."},

    {"problem": {"has_fever": False, "has_cough": True, "shortness_of_breath": False,
                 "heart_rate": 75, "duration_of_symptoms": 21},
     "solution": "Chronic cough: consult a doctor to check for allergies or TB."},

    {"problem": {"has_fever": False, "has_cough": False, "shortness_of_breath": True,
                 "heart_rate": 120, "duration_of_symptoms": 1},
     "solution": "Possible asthma attack or heart problem: go to the emergency room."},

    {"problem": {"has_fever": True, "has_cough": False, "shortness_of_breath": False,
                 "heart_rate": 95, "duration_of_symptoms": 2},
     "solution": "Viral fever: rest, drink fluids, and monitor your temperature."},
]

# Expected spread of each numeric feature, used to scale differences to 0-1
NUMERIC_RANGES = {
    "heart_rate": 80,              # roughly 60 to 140 bpm
    "duration_of_symptoms": 30,    # 0 to 30 days
}


def similarity(new_problem, old_problem):
    """Return a similarity score from 0.0 (nothing alike) to 1.0 (identical).

    - True/False features score 1 if they match, 0 if they do not.
    - Numeric features score 1 minus the scaled difference
      (e.g. heart rate 90 vs 85 is very similar, 90 vs 120 is not).
    The final score is the average over all features.
    """
    total = 0.0
    for feature, new_value in new_problem.items():
        old_value = old_problem[feature]
        if isinstance(new_value, bool):
            total += 1.0 if new_value == old_value else 0.0
        else:
            diff = abs(new_value - old_value) / NUMERIC_RANGES[feature]
            total += max(0.0, 1.0 - diff)
    return total / len(new_problem)


def retrieve(new_problem, case_base):
    print("Retrieve (similarity to each past case):")
    best_case, best_score = None, -1.0
    for number, case in enumerate(case_base, start=1):
        score = similarity(new_problem, case["problem"])
        print(f"  Case {number}: {score:.0%}  ->  {case['solution']}")
        if score > best_score:
            best_case, best_score = case, score
    return best_case, best_score


def reuse(best_case):
    return best_case["solution"]


def revise(solution):
    answer = input("\nRevise: accept this solution? (y/n): ").strip().lower()
    if answer == "n":
        new_solution = input("Enter the revised solution: ").strip()
        if new_solution:
            return new_solution
    return solution


def retain(new_problem, solution, case_base):
    case_base.append({"problem": dict(new_problem), "solution": solution})


def run_cbr(new_problem):
    print("-" * 65)
    print("PART 3 - CASE-BASED REASONING (CBR)")
    print("-" * 65)
    print("Input (new problem):")
    display_facts(new_problem)
    print()

    best_case, score = retrieve(new_problem, case_base)

    solution = reuse(best_case)
    print(f"\nReuse: most similar case ({score:.0%} match) suggests:")
    print("-", solution)

    final_solution = revise(solution)

    retain(new_problem, final_solution, case_base)
    print("\nRetain: new case saved to the case base.")
    print(f"  Case base now has {len(case_base)} cases.")

    print("\nCase-Based Reasoning Solution:")
    print("-", final_solution)
    print()
    return final_solution



if __name__ == "__main__":
    rbr_actions = run_rbr(medical_facts)
    cbr_solution = run_cbr(medical_facts)

    print("-" * 65)
    print("COMPARISON (same patient)")
    print("-" * 65)
    print("RBR answer:", " | ".join(rbr_actions))
    print("CBR answer:", cbr_solution)