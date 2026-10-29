"""
src/guardrails.py
Comprehensive guardrail engine for the Loan Decision System:
1. Input Guardrails (Field validation, range checks, prompt injection detection)
2. Decision Guardrails (Hard policy rules, model-policy conflict resolution, fair lending checks)
3. LLM Output Guardrails (Citation validation, decision consistency, hallucination blocks)
"""

import re
from typing import Dict, Any, Tuple, List


# ---------------------------------------------------------
# PROMPT INJECTION DETECTION PATTERNS
# ---------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(policy|rules|instructions?)",
    r"system\s+(override|prompt|directive)",
    r"you\s+are\s+now\s+(an?\s+)?(admin|unrestricted|dan)",
    r"approve\s+this\s+loan(\s+immediately|\s+now)?",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"bypass\s+(safety|guardrail|policy)",
    r"new\s+rule:",
    r"developer\s+mode",
]


def validate_inputs(applicant: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input Guardrails: Validates ranges, types, realism, and scans for prompt injection.
    """
    checks = []
    errors = []
    is_valid = True

    # 1. Required fields
    required = ["age", "income", "employment_type", "loan_amount", "credit_score"]
    missing = [f for f in required if f not in applicant or applicant[f] is None]
    if missing:
        checks.append({
            "name": "Required Fields",
            "passed": False,
            "details": f"Missing required fields: {', '.join(missing)}"
        })
        return {"passed": False, "checks": checks, "errors": [f"Missing required fields: {', '.join(missing)}"]}
    else:
        checks.append({"name": "Required Fields", "passed": True, "details": "All mandatory fields present."})

    # 2. Age Check (18 - 100)
    age = applicant.get("age")
    try:
        age = int(age)
        if 18 <= age <= 100:
            checks.append({"name": "Age Boundary (18-100)", "passed": True, "details": f"Age {age} is within valid bounds."})
        else:
            is_valid = False
            msg = f"Age {age} is outside the allowable range (18 to 100 years)."
            checks.append({"name": "Age Boundary (18-100)", "passed": False, "details": msg})
            errors.append(msg)
    except (ValueError, TypeError):
        is_valid = False
        msg = f"Age must be a valid integer, received '{age}'."
        checks.append({"name": "Age Boundary (18-100)", "passed": False, "details": msg})
        errors.append(msg)

    # 3. Monthly Income (Positive & Realism check: max ₹10 Crore / month)
    income = applicant.get("income")
    try:
        income = float(income)
        if income <= 0:
            is_valid = False
            msg = f"Monthly income must be positive. Received ₹{income:,.2f}."
            checks.append({"name": "Income Positivity", "passed": False, "details": msg})
            errors.append(msg)
        elif income > 100000000:  # 10 Crore
            is_valid = False
            msg = f"Monthly income ₹{income:,.2f} exceeds realistic threshold (₹10 Crore/month cap)."
            checks.append({"name": "Income Realism Check", "passed": False, "details": msg})
            errors.append(msg)
        else:
            checks.append({"name": "Income Realism & Range", "passed": True, "details": f"Income of ₹{income:,.2f}/mo is valid."})
    except (ValueError, TypeError):
        is_valid = False
        msg = f"Income must be numeric, received '{income}'."
        checks.append({"name": "Income Realism & Range", "passed": False, "details": msg})
        errors.append(msg)

    # 4. Credit Score (300 - 900)
    cs = applicant.get("credit_score")
    try:
        cs = int(cs)
        if 300 <= cs <= 900:
            checks.append({"name": "Credit Score Range (300-900)", "passed": True, "details": f"Credit score {cs} is valid."})
        else:
            is_valid = False
            msg = f"Credit score {cs} is outside bureau boundaries (300 to 900)."
            checks.append({"name": "Credit Score Range (300-900)", "passed": False, "details": msg})
            errors.append(msg)
    except (ValueError, TypeError):
        is_valid = False
        msg = f"Credit score must be an integer, received '{cs}'."
        checks.append({"name": "Credit Score Range (300-900)", "passed": False, "details": msg})
        errors.append(msg)

    # 5. Loan Amount (Positive & Reasonable Cap: max ₹10 Crore)
    amount = applicant.get("loan_amount")
    try:
        amount = float(amount)
        if amount <= 0:
            is_valid = False
            msg = f"Loan amount must be strictly greater than 0. Received ₹{amount:,.2f}."
            checks.append({"name": "Loan Amount Range", "passed": False, "details": msg})
            errors.append(msg)
        elif amount > 100000000:
            is_valid = False
            msg = f"Loan amount ₹{amount:,.2f} exceeds max retail exposure cap (₹10 Crore)."
            checks.append({"name": "Loan Amount Range", "passed": False, "details": msg})
            errors.append(msg)
        else:
            checks.append({"name": "Loan Amount Range", "passed": True, "details": f"Requested amount ₹{amount:,.2f} is valid."})
    except (ValueError, TypeError):
        is_valid = False
        msg = f"Loan amount must be numeric, received '{amount}'."
        checks.append({"name": "Loan Amount Range", "passed": False, "details": msg})
        errors.append(msg)

    # 6. Prompt Injection Defense on free text fields (loan_purpose, notes, comments)
    text_fields = ["loan_purpose", "notes", "applicant_notes", "purpose"]
    injection_found = False
    flagged_terms = []

    for field in text_fields:
        val = str(applicant.get(field, ""))
        if not val or val.lower() == "none":
            continue
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, val, re.IGNORECASE):
                injection_found = True
                flagged_terms.append(pattern)

    if injection_found:
        is_valid = False
        msg = "Adversarial prompt injection attempt detected in text input field."
        checks.append({"name": "Prompt Injection Defense", "passed": False, "details": msg})
        errors.append(msg)
    else:
        checks.append({"name": "Prompt Injection Defense", "passed": True, "details": "Clean free-text fields with zero adversarial prompt injection patterns."})

    return {
        "passed": is_valid,
        "checks": checks,
        "errors": errors
    }


def check_decision_guardrails(applicant: Dict[str, Any], model_prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision Guardrails:
    - Enforces hard bank policy overrides (Credit Score < 600 or DTI > 50% cannot auto-approve).
    - Checks for model-policy conflicts and routes to Human Review.
    - Ensures sensitive demographic attributes are strictly excluded.
    """
    checks = []
    override = None
    override_reasons = []

    cs = int(applicant.get("credit_score", 0))
    dti = float(applicant.get("dti_ratio") or model_prediction.get("dti_ratio") or 0.0)
    if dti == 0.0 and applicant.get("income") and applicant.get("existing_debt"):
        inc = float(applicant["income"])
        debt = float(applicant["existing_debt"])
        dti = round((debt / inc) * 100.0, 2) if inc > 0 else 0.0

    pdef = int(applicant.get("past_defaults", 0))
    amount = float(applicant.get("loan_amount", 0.0))
    emp_type = str(applicant.get("employment_type", ""))
    model_decision = model_prediction.get("decision", "Rejected")
    prob = model_prediction.get("approval_probability", 0.0)

    # 1. Fair Lending / Sensitive Attributes Check
    prohibited = ["gender", "sex", "religion", "caste", "race", "marital_status", "sexual_orientation"]
    used_prohibited = [p for p in prohibited if p in applicant]
    if used_prohibited:
        checks.append({
            "name": "Fair Lending Compliance",
            "passed": False,
            "details": f"Policy Section 7 Violation: Prohibited sensitive features present ({', '.join(used_prohibited)})."
        })
    else:
        checks.append({
            "name": "Fair Lending Compliance",
            "passed": True,
            "details": "Policy Section 7: Zero protected attributes used in model inference."
        })

    # 2. Hard Policy Rule: Credit Score < 600 (Section 2.4)
    if cs < 600:
        if model_decision == "Approved":
            override = "Rejected"
            override_reasons.append("Section 2.4 Hard Rule: Credit score < 600 overrides model approval to auto-reject.")
        checks.append({
            "name": "Credit Score Policy Cap (Sec 2.4)",
            "passed": False,
            "details": f"Credit score {cs} < 600 triggers mandatory Section 2.4 rejection threshold."
        })
    else:
        checks.append({
            "name": "Credit Score Policy Cap (Sec 2.4)",
            "passed": True,
            "details": f"Credit score {cs} satisfies minimum threshold (≥ 600)."
        })

    # 3. Hard Policy Rule: DTI > 50% (Section 3.3)
    if dti > 50.0:
        if model_decision == "Approved":
            override = "Rejected"
            override_reasons.append("Section 3.3 Hard Rule: DTI ratio > 50% overrides model approval to auto-reject.")
        checks.append({
            "name": "DTI Leverage Limit (Sec 3.3)",
            "passed": False,
            "details": f"DTI ratio {dti:.1f}% exceeds Section 3.3 statutory ceiling (> 50%)."
        })
    else:
        checks.append({
            "name": "DTI Leverage Limit (Sec 3.3)",
            "passed": True,
            "details": f"DTI ratio {dti:.1f}% meets Section 3 prudent leverage ceiling (≤ 50%)."
        })

    # 4. Multiple Defaults Hard Rule (Section 5.3)
    if pdef >= 2:
        if model_decision == "Approved":
            override = "Rejected"
            override_reasons.append(f"Section 5.3 Hard Rule: {pdef} past defaults mandates rejection.")
        checks.append({
            "name": "Past Defaults Limit (Sec 5.3)",
            "passed": False,
            "details": f"{pdef} past defaults exceeds allowable tolerance (Sec 5.3)."
        })
    else:
        checks.append({
            "name": "Past Defaults Limit (Sec 5.3)",
            "passed": True,
            "details": f"{pdef} defaults complies with Section 5 delinquency standards."
        })

    # 5. High-Value Loan Trigger (Section 4.2: > ₹25 Lakhs)
    high_value_flag = False
    if amount > 2500000:
        high_value_flag = True
        checks.append({
            "name": "High-Value Quantum Check (Sec 4.2)",
            "passed": False,
            "details": f"Loan amount ₹{amount:,.0f} exceeds ₹25 Lakhs; mandates collateral & senior officer approval."
        })
    else:
        checks.append({
            "name": "High-Value Quantum Check (Sec 4.2)",
            "passed": True,
            "details": f"Loan amount ₹{amount:,.0f} is within standard unsecured limits (≤ ₹25 Lakhs)."
        })

    # 6. Policy Coverage Gap Check (Section 1.3: Freelance / Gig Economy)
    policy_gap_flag = False
    if "freelance" in emp_type.lower() or "gig" in emp_type.lower():
        policy_gap_flag = True
        checks.append({
            "name": "Employment Policy Scope (Sec 1.3)",
            "passed": False,
            "details": "Section 1.3 Gap: Gig worker / Freelance income is outside standard automated underwriting."
        })
    else:
        checks.append({
            "name": "Employment Policy Scope (Sec 1.3)",
            "passed": True,
            "details": f"Employment type '{emp_type}' is fully covered by Section 1.2 standard criteria."
        })

    # 7. Model Borderline Probability (Section 6.1: 0.40 <= prob <= 0.60)
    borderline_flag = 0.40 <= prob <= 0.60
    if borderline_flag:
        checks.append({
            "name": "Borderline Probability Check (Sec 6.1)",
            "passed": False,
            "details": f"Approval probability {prob:.1%} lies within borderline band (40% - 60%)."
        })
    else:
        checks.append({
            "name": "Borderline Probability Check (Sec 6.1)",
            "passed": True,
            "details": f"Approval probability {prob:.1%} is outside borderline band."
        })

    # Determine if human checkpoint escalation is required
    requires_human_review = (
        borderline_flag or
        high_value_flag or
        policy_gap_flag or
        (override is not None)
    )

    final_decision = override if override else model_decision

    all_passed = (override is None) and (not high_value_flag) and (not policy_gap_flag) and (not borderline_flag)

    return {
        "passed": all_passed,
        "requires_human_review": requires_human_review,
        "escalation_reasons": override_reasons + (
            ["High loan quantum (> ₹25L) requires manual sanction."] if high_value_flag else []
        ) + (
            ["Freelance/gig policy gap requires human underwriting appraisal."] if policy_gap_flag else []
        ) + (
            ["Borderline model confidence (40-60%) mandates human checkpoint."] if borderline_flag else []
        ),
        "hard_override": override,
        "final_decision": final_decision,
        "checks": checks
    }


def validate_llm_output(
    explanation: str,
    target_decision: str,
    retrieved_sections: List[str],
    similarity_score: float,
    min_similarity_threshold: float = 0.35
) -> Dict[str, Any]:
    """
    LLM Output Guardrails:
    - Mandatory policy section citations (e.g. 'Section 1', 'Section 2.4').
    - Decision consistency: Explanation must align with final decision.
    - Prohibited promises, hallucinatory terms, or protected attributes.
    - Low similarity retrieval fallback.
    """
    checks = []
    issues = []
    needs_regen = False

    # 1. Section Citation Check (Section 8.2)
    citation_pattern = r"(Section\s+[1-8](\.[1-9])?|Sec\.\s+[1-8])"
    citations_found = re.findall(citation_pattern, explanation, re.IGNORECASE)
    if citations_found:
        checks.append({
            "name": "Policy Section Citations (Sec 8.2)",
            "passed": True,
            "details": f"Valid section citations identified: {[c[0] for c in citations_found[:3]]}"
        })
    else:
        needs_regen = True
        msg = "Explanation lacks required policy section citations (Section 8.2)."
        checks.append({"name": "Policy Section Citations (Sec 8.2)", "passed": False, "details": msg})
        issues.append(msg)

    # 2. Decision Consistency Check
    lower_text = explanation.lower()
    if target_decision.lower() == "approved":
        if "rejected" in lower_text or "declined" in lower_text or "disapproved" in lower_text:
            if not ("not rejected" in lower_text or "past default was rejected" in lower_text):
                msg = "Decision contradiction: Decision is Approved, but explanation text discusses rejection."
                checks.append({"name": "Decision Consistency", "passed": False, "details": msg})
                issues.append(msg)
                needs_regen = True
            else:
                checks.append({"name": "Decision Consistency", "passed": True, "details": "Matches Approved decision."})
        else:
            checks.append({"name": "Decision Consistency", "passed": True, "details": "Matches Approved decision."})
    elif target_decision.lower() == "rejected":
        if "your loan has been approved" in lower_text or "application is approved" in lower_text:
            msg = "Decision contradiction: Decision is Rejected, but explanation claims approval."
            checks.append({"name": "Decision Consistency", "passed": False, "details": msg})
            issues.append(msg)
            needs_regen = True
        else:
            checks.append({"name": "Decision Consistency", "passed": True, "details": "Matches Rejected decision."})
    else:
        checks.append({"name": "Decision Consistency", "passed": True, "details": "Accurately represents Human Review state."})

    # 3. Forbidden Promises & Hallucinatory Terms Check
    hallucination_patterns = [
        r"100%\s+guarantee",
        r"definitely\s+(get\s+)?approved",
        r"we\s+promise\s+approval",
        r"guaranteed\s+loan",
        r"no\s+repayment\s+required",
    ]
    hallucination_found = False
    for pat in hallucination_patterns:
        if re.search(pat, lower_text):
            hallucination_found = True
            break

    if hallucination_found:
        msg = "Hallucination guardrail failed: Unlawful guarantee or promise detected."
        checks.append({"name": "Hallucination & Promise Guard", "passed": False, "details": msg})
        issues.append(msg)
        needs_regen = True
    else:
        checks.append({"name": "Hallucination & Promise Guard", "passed": True, "details": "No unlawful guarantees or hallucinatory commitments."})

    # 4. Fair Lending Mention Guard
    protected_terms = ["gender", "religion", "caste", "marital status", "sexual orientation", "hindu", "muslim", "christian"]
    found_protected = [t for t in protected_terms if t in lower_text]
    if found_protected:
        msg = f"Fair lending output guardrail: Explanation referenced protected demographic terms ({', '.join(found_protected)})."
        checks.append({"name": "Protected Attributes Exclusion", "passed": False, "details": msg})
        issues.append(msg)
    else:
        checks.append({"name": "Protected Attributes Exclusion", "passed": True, "details": "Zero protected demographic terms in customer response."})

    # 5. Low Retrieval Similarity / Out-of-Scope Guard
    if similarity_score < min_similarity_threshold:
        msg = f"Low retrieval confidence ({similarity_score:.2f} < {min_similarity_threshold}). Policy may not cover this query."
        checks.append({"name": "Policy Grounding Confidence", "passed": False, "details": msg})
        issues.append(msg)
    else:
        checks.append({"name": "Policy Grounding Confidence", "passed": True, "details": f"Strong semantic policy match ({similarity_score:.2f})."})

    all_passed = len(issues) == 0

    return {
        "passed": all_passed,
        "needs_regeneration": needs_regen,
        "checks": checks,
        "flagged_issues": issues
    }
