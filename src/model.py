"""
src/model.py
Predictive AI module for the Loan Decision and Explanation System.
Generates synthetic data, trains Logistic Regression and Random Forest models,
evaluates performance, and extracts top decision drivers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODEL_PATH = os.path.join(DATA_DIR, "loan_model.joblib")
DATASET_PATH = os.path.join(DATA_DIR, "loan_data.csv")
METRICS_PATH = os.path.join(DATA_DIR, "model_metrics.joblib")


def generate_synthetic_data(num_samples: int = 3500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic retail lending dataset in Indian Rupees (INR).
    Features reflect standard underwriting criteria.
    """
    np.random.seed(random_seed)

    # 1. Age (21 - 65)
    age = np.random.randint(21, 66, size=num_samples)

    # 2. Employment Type: Salaried (60%), Self-Employed (30%), Freelance (10%)
    emp_choices = ["Salaried", "Self-Employed", "Freelance"]
    emp_weights = [0.60, 0.30, 0.10]
    employment_type = np.random.choice(emp_choices, size=num_samples, p=emp_weights)

    # 3. Years Employed (0 to 35 years)
    years_employed = np.round(np.clip(np.random.exponential(scale=5.0, size=num_samples), 0.5, 35.0), 1)

    # 4. Monthly Income in INR (₹20,000 to ₹3,50,000)
    # Lognormal distribution represents real income distribution well
    raw_income = np.random.lognormal(mean=11.0, sigma=0.65, size=num_samples)
    income = np.round(np.clip(raw_income, 20000, 400000), -2)

    # 5. Loan Amount in INR (₹50,000 to ₹40,00,000)
    # Typically 5x to 25x monthly income
    multipliers = np.random.uniform(5.0, 25.0, size=num_samples)
    loan_amount = np.round(np.clip(income * multipliers, 50000, 5000000), -3)

    # 6. Loan Term (months: 12, 24, 36, 48, 60, 84, 120, 240)
    terms = [12, 24, 36, 48, 60, 84, 120, 240]
    loan_term = np.random.choice(terms, size=num_samples, p=[0.05, 0.15, 0.35, 0.20, 0.15, 0.05, 0.03, 0.02])

    # 7. Existing Debt / Monthly EMI (INR)
    # Existing DTI ratio between 5% and 65%
    raw_dti = np.random.beta(a=2.5, b=4.0, size=num_samples) * 0.70
    existing_debt = np.round(income * raw_dti, -2)

    # 8. Debt-to-Income (DTI) Ratio (%)
    dti_ratio = np.round((existing_debt / income) * 100.0, 2)

    # 9. Credit Score (CIBIL / Experian scale: 300 to 900)
    # Mixture of prime (700-850), near-prime (620-700), subprime (400-620)
    score_raw = np.random.normal(loc=710, scale=85, size=num_samples)
    credit_score = np.clip(np.round(score_raw), 300, 900).astype(int)

    # 10. Past Defaults (0, 1, 2, 3+)
    past_defaults = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.75, 0.15, 0.07, 0.03])

    # 11. Collateral (Yes / No)
    # Higher chance of collateral for larger loans
    collateral_prob = np.clip(loan_amount / 4000000.0, 0.15, 0.85)
    collateral = np.array(["Yes" if np.random.rand() < p else "No" for p in collateral_prob])

    # Target: Loan Approved (0 = Reject, 1 = Approve)
    # Ground truth underwriting heuristic with realistic noise
    approval_score = np.zeros(num_samples)

    # Credit score contribution (-4.0 to +4.0)
    approval_score += (credit_score - 650) / 50.0

    # DTI contribution (penalty for high DTI)
    approval_score -= (dti_ratio - 35.0) / 10.0

    # Past defaults penalty (heavy)
    approval_score -= past_defaults * 2.5

    # Income & stability bonus
    approval_score += np.log10(income / 25000.0) * 1.5
    approval_score += np.clip(years_employed / 5.0, 0, 2.0)

    # Loan size stress
    approval_score -= (loan_amount / (income * 12.0)) * 0.8

    # Collateral mitigation
    approval_score += np.where(collateral == "Yes", 1.2, -0.2)

    # Employment type factor
    approval_score += np.where(employment_type == "Salaried", 0.6,
                      np.where(employment_type == "Self-Employed", 0.1, -0.4))

    # Add Gaussian noise
    approval_score += np.random.normal(0, 0.8, size=num_samples)

    # Convert to probability via logistic function
    probs = 1.0 / (1.0 + np.exp(-approval_score))
    loan_approved = (probs >= 0.50).astype(int)

    # Hard real-world banking constraints:
    # 1. Credit score < 580 -> almost zero approval
    loan_approved[credit_score < 560] = 0
    # 2. Defaults >= 3 -> auto-reject
    loan_approved[past_defaults >= 3] = 0
    # 3. DTI > 60% with no collateral -> auto-reject
    loan_approved[(dti_ratio > 60.0) & (collateral == "No")] = 0

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "employment_type": employment_type,
        "years_employed": years_employed,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "existing_debt": existing_debt,
        "dti_ratio": dti_ratio,
        "credit_score": credit_score,
        "past_defaults": past_defaults,
        "collateral": collateral,
        "loan_approved": loan_approved
    })

    return df


def train_and_evaluate_models():
    """
    Trains Logistic Regression and Random Forest on the dataset,
    evaluates both, picks the better one, and saves artifacts.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating synthetic lending dataset (3,500 samples)...")
    df = generate_synthetic_data(num_samples=3500)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Saved dataset to {DATASET_PATH} (Shape: {df.shape})")

    feature_cols = [
        "age", "income", "employment_type", "years_employed",
        "loan_amount", "loan_term", "existing_debt", "dti_ratio",
        "credit_score", "past_defaults", "collateral"
    ]
    target_col = "loan_approved"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = [
        "age", "income", "years_employed", "loan_amount",
        "loan_term", "existing_debt", "dti_ratio", "credit_score", "past_defaults"
    ]
    categorical_features = ["employment_type", "collateral"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_proc, y_train)
    y_pred_lr = lr.predict(X_test_proc)
    y_prob_lr = lr.predict_proba(X_test_proc)[:, 1]

    # Model 2: Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_train_proc, y_train)
    y_pred_rf = rf.predict(X_test_proc)
    y_prob_rf = rf.predict_proba(X_test_proc)[:, 1]

    def compute_metrics(y_true, y_pred, y_prob):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

    metrics_lr = compute_metrics(y_test, y_pred_lr, y_prob_lr)
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_prob_rf)

    print("\n--- Model Performance Comparison ---")
    print(f"Logistic Regression: Accuracy={metrics_lr['accuracy']:.4f}, F1={metrics_lr['f1']:.4f}, ROC-AUC={metrics_lr['roc_auc']:.4f}")
    print(f"Random Forest:       Accuracy={metrics_rf['accuracy']:.4f}, F1={metrics_rf['f1']:.4f}, ROC-AUC={metrics_rf['roc_auc']:.4f}")

    # Pick the better model based on ROC-AUC and F1
    if metrics_rf["roc_auc"] >= metrics_lr["roc_auc"]:
        chosen_name = "Random Forest"
        best_model = rf
    else:
        chosen_name = "Logistic Regression"
        best_model = lr

    print(f"\nWinning Model Selected: {chosen_name}")

    # Get transformed feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + encoded_cat_names

    bundle = {
        "model_name": chosen_name,
        "model": best_model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_feature_names": all_feature_names,
        "metrics": {
            "Logistic Regression": metrics_lr,
            "Random Forest": metrics_rf,
            "selected_model": chosen_name
        }
    }

    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(bundle["metrics"], METRICS_PATH)
    print(f"Saved best model bundle to {MODEL_PATH}")

    return bundle


def load_model_bundle():
    """Loads the trained model bundle; trains if not yet saved."""
    if not os.path.exists(MODEL_PATH):
        return train_and_evaluate_models()
    return joblib.load(MODEL_PATH)


def predict_loan_application(applicant_dict: dict) -> dict:
    """
    Predicts loan approval and extracts top 3 explanatory features.

    Parameters:
        applicant_dict: dict containing:
            - age (int)
            - income (float/int)
            - employment_type (str)
            - years_employed (float/int)
            - loan_amount (float/int)
            - loan_term (int)
            - existing_debt (float/int)
            - credit_score (int)
            - past_defaults (int)
            - collateral (str: 'Yes' or 'No')
            - (optional) dti_ratio (computed if missing)

    Returns:
        dict with:
            - decision: 'Approved' or 'Rejected'
            - approval_probability: float (0.0 to 1.0)
            - top_reasons: list of top 3 driver dicts
    """
    bundle = load_model_bundle()
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_cols = bundle["feature_cols"]
    numeric_features = bundle["numeric_features"]

    # Compute dti_ratio if not explicitly provided
    data = dict(applicant_dict)
    if "dti_ratio" not in data or data["dti_ratio"] is None:
        inc = float(data.get("income", 1.0))
        debt = float(data.get("existing_debt", 0.0))
        data["dti_ratio"] = round((debt / inc) * 100.0, 2) if inc > 0 else 100.0

    df_single = pd.DataFrame([data])[feature_cols]
    X_proc = preprocessor.transform(df_single)

    prob_approval = float(model.predict_proba(X_proc)[0, 1])
    raw_decision = "Approved" if prob_approval >= 0.50 else "Rejected"

    # Compute feature contributions/importance for this specific applicant
    # Baseline comparison against dataset mean/median for directional insight
    reasons = []

    cs = int(data.get("credit_score", 650))
    dti = float(data.get("dti_ratio", 35.0))
    pdef = int(data.get("past_defaults", 0))
    inc = float(data.get("income", 50000))
    amt = float(data.get("loan_amount", 500000))
    collat = str(data.get("collateral", "No")).capitalize()
    emp = str(data.get("employment_type", "Salaried"))
    exp = float(data.get("years_employed", 2.0))

    # Evaluate feature impact scores
    # 1. Credit Score
    if cs >= 750:
        reasons.append({
            "feature": "Credit Score",
            "value": f"{cs}",
            "impact": "Positive",
            "weight": 3.0,
            "description": f"Excellent credit score of {cs} indicates strong historical repayment discipline."
        })
    elif cs >= 650:
        reasons.append({
            "feature": "Credit Score",
            "value": f"{cs}",
            "impact": "Moderate",
            "weight": 1.2,
            "description": f"Fair credit score of {cs} meets basic standard requirements."
        })
    else:
        reasons.append({
            "feature": "Credit Score",
            "value": f"{cs}",
            "impact": "Negative",
            "weight": 3.5,
            "description": f"Low credit score of {cs} is below the standard prime threshold (650)."
        })

    # 2. DTI Ratio
    if dti <= 35.0:
        reasons.append({
            "feature": "Debt-to-Income (DTI)",
            "value": f"{dti:.1f}%",
            "impact": "Positive",
            "weight": 2.5,
            "description": f"Healthy DTI ratio of {dti:.1f}% shows low existing debt burden."
        })
    elif dti <= 50.0:
        reasons.append({
            "feature": "Debt-to-Income (DTI)",
            "value": f"{dti:.1f}%",
            "impact": "Moderate",
            "weight": 1.5,
            "description": f"Moderate DTI ratio of {dti:.1f}% approaches the maximum prudent limit."
        })
    else:
        reasons.append({
            "feature": "Debt-to-Income (DTI)",
            "value": f"{dti:.1f}%",
            "impact": "Negative",
            "weight": 3.2,
            "description": f"High DTI ratio of {dti:.1f}% exceeds bank policy limit (>50%)."
        })

    # 3. Past Defaults
    if pdef == 0:
        reasons.append({
            "feature": "Past Defaults",
            "value": "0",
            "impact": "Positive",
            "weight": 2.0,
            "description": "Zero historical loan defaults on credit record."
        })
    elif pdef == 1:
        reasons.append({
            "feature": "Past Defaults",
            "value": "1 default",
            "impact": "Negative",
            "weight": 2.6,
            "description": "Applicant has 1 past default requiring conditional review."
        })
    else:
        reasons.append({
            "feature": "Past Defaults",
            "value": f"{pdef} defaults",
            "impact": "Negative",
            "weight": 4.0,
            "description": f"Multiple past defaults ({pdef}) indicate high default risk."
        })

    # 4. Income to Loan Amount Ratio
    annual_income = inc * 12.0
    loan_to_income = amt / annual_income if annual_income > 0 else 10.0
    if loan_to_income <= 2.5:
        reasons.append({
            "feature": "Loan-to-Annual Income",
            "value": f"{loan_to_income:.1f}x",
            "impact": "Positive",
            "weight": 1.8,
            "description": f"Requested loan amount is {loan_to_income:.1f}x annual income, within affordable range."
        })
    elif loan_to_income > 5.0:
        reasons.append({
            "feature": "Loan-to-Annual Income",
            "value": f"{loan_to_income:.1f}x",
            "impact": "Negative",
            "weight": 2.2,
            "description": f"Requested loan amount is {loan_to_income:.1f}x annual income, indicating over-leverage risk."
        })

    # 5. Collateral
    if collat == "Yes":
        reasons.append({
            "feature": "Collateral Provided",
            "value": "Yes",
            "impact": "Positive",
            "weight": 1.7,
            "description": "Secured collateral mitigates credit loss risk."
        })
    else:
        if amt > 2500000:
            reasons.append({
                "feature": "Collateral Provided",
                "value": "No",
                "impact": "Negative",
                "weight": 2.5,
                "description": "Unsecured loan requested above ₹25 Lakhs without collateral pledge."
            })

    # 6. Employment Stability
    if emp == "Salaried" and exp >= 3.0:
        reasons.append({
            "feature": "Employment Stability",
            "value": f"Salaried ({exp} yrs)",
            "impact": "Positive",
            "weight": 1.4,
            "description": f"Stable salaried employment with {exp} years of verified tenure."
        })
    elif emp == "Freelance":
        reasons.append({
            "feature": "Employment Type",
            "value": "Freelance / Gig",
            "impact": "Moderate",
            "weight": 1.9,
            "description": "Freelance income streams require non-standard income documentation."
        })

    # Sort reasons by weight depending on decision:
    # If rejected, prioritize negative drivers; if approved, prioritize positive drivers
    if raw_decision == "Rejected":
        reasons.sort(key=lambda x: (x["impact"] == "Negative", x["weight"]), reverse=True)
    else:
        reasons.sort(key=lambda x: (x["impact"] == "Positive", x["weight"]), reverse=True)

    top_3 = reasons[:3]

    return {
        "decision": raw_decision,
        "approval_probability": round(prob_approval, 4),
        "risk_probability": round(1.0 - prob_approval, 4),
        "dti_ratio": data["dti_ratio"],
        "top_reasons": top_3,
        "all_evaluated_factors": reasons
    }


if __name__ == "__main__":
    train_and_evaluate_models()

