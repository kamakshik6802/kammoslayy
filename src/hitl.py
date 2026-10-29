"""
src/hitl.py
Human-in-the-Loop (HITL) Checkpoint and Audit Logging module.
Manages the loan officer review queue, decision overrides, and immutable audit logs.
"""

import os
import csv
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
AUDIT_LOG_PATH = os.path.join(LOGS_DIR, "audit_log.csv")

AUDIT_LOG_COLUMNS = [
    "timestamp",
    "application_id",
    "age",
    "income",
    "employment_type",
    "loan_amount",
    "loan_term",
    "dti_ratio",
    "credit_score",
    "past_defaults",
    "collateral",
    "ai_decision",
    "ai_probability",
    "guardrails_passed",
    "escalation_reasons",
    "final_status",
    "officer_decision",
    "officer_comment",
    "is_override",
    "policy_sections_cited"
]


def init_audit_log():
    """Initializes audit_log.csv with headers if not present."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(AUDIT_LOG_COLUMNS)


def log_application(
    applicant: Dict[str, Any],
    decision_result: Dict[str, Any],
    guardrail_result: Dict[str, Any],
    rag_result: Dict[str, Any],
    application_id: Optional[str] = None
) -> str:
    """
    Appends an application decision to the audit log.
    If human review is required, status is marked as 'Escalated / Pending Officer Review'.
    """
    init_audit_log()

    if not application_id:
        application_id = f"APP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prob = decision_result.get("approval_probability", 0.0)
    ai_dec = decision_result.get("decision", "Rejected")
    guardrail_passed = guardrail_result.get("passed", True)
    requires_officer = guardrail_result.get("requires_human_review", False)
    escalations = "; ".join(guardrail_result.get("escalation_reasons", [])) or "None"

    retrieved = rag_result.get("retrieved_chunks", [])
    sections = ", ".join([f"Sec {c['section_number']}" for c in retrieved]) or "None"

    if requires_officer:
        final_status = "Pending Officer Review"
        officer_dec = "Pending"
        officer_comment = "Awaiting manual underwriting evaluation."
        is_override = False
    else:
        final_status = guardrail_result.get("final_decision", ai_dec)
        officer_dec = "N/A (Automated)"
        officer_comment = "Auto-sanctioned within established policy parameters."
        is_override = False

    row = [
        timestamp,
        application_id,
        applicant.get("age", 0),
        applicant.get("income", 0.0),
        applicant.get("employment_type", "Unknown"),
        applicant.get("loan_amount", 0.0),
        applicant.get("loan_term", 0),
        applicant.get("dti_ratio", 0.0),
        applicant.get("credit_score", 0),
        applicant.get("past_defaults", 0),
        applicant.get("collateral", "No"),
        ai_dec,
        f"{prob:.4f}",
        "Yes" if guardrail_passed else "No",
        escalations,
        final_status,
        officer_dec,
        officer_comment,
        "Yes" if is_override else "No",
        sections
    ]

    with open(AUDIT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

    return application_id


def record_officer_action(
    application_id: str,
    officer_decision: str,
    officer_comment: str
) -> bool:
    """
    Updates an escalated application in the audit log with the loan officer's decision.
    """
    init_audit_log()
    df = pd.read_csv(AUDIT_LOG_PATH)

    match_idx = df.index[df["application_id"] == application_id].tolist()
    if not match_idx:
        return False

    idx = match_idx[-1]  # Update most recent match
    ai_dec = df.at[idx, "ai_decision"]
    is_override = (officer_decision != ai_dec)

    df.at[idx, "officer_decision"] = officer_decision
    df.at[idx, "officer_comment"] = officer_comment.strip()
    df.at[idx, "final_status"] = officer_decision
    df.at[idx, "is_override"] = "Yes" if is_override else "No"
    df.at[idx, "timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(AUDIT_LOG_PATH, index=False)
    return True


def get_pending_reviews() -> List[Dict[str, Any]]:
    """Returns list of all applications currently in the Officer Review Queue."""
    init_audit_log()
    df = pd.read_csv(AUDIT_LOG_PATH)
    pending_df = df[df["final_status"] == "Pending Officer Review"]
    return pending_df.to_dict(orient="records")


def load_audit_log() -> pd.DataFrame:
    """Loads full audit log DataFrame."""
    init_audit_log()
    return pd.read_csv(AUDIT_LOG_PATH)

