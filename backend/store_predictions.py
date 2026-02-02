import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sqlalchemy import text
""" # from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error
"""

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

# -------------------- DB --------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database import engine

# -------------------- ROUND ORDER --------------------
ROUND_ORDER = ["R1", "R2", "R3", "R4"]

# -------------------- GENERATE PREDICTIONS --------------------
def generate_predictions():
    print("🔄 Generating predictions using ML-based trend analysis...")

    query = """
    SELECT
        year,
        round,
        college_code,
        college_name,
        branch,
        category,
        closing_rank
    FROM COMEDK_MASTER_2021_2025
    WHERE closing_rank IS NOT NULL
    """

    df = pd.read_sql(query, engine)

    if df.empty:
        print("❌ No data available")
        return

    # ---------------- PREPROCESS ----------------
    df["round"] = df["round"].astype(str).str.strip().str.upper()
    df["category"] = df["category"].replace({"GM/KKR": "GM", "HKR": "KKR"})
    df["category"] = df["category"].str.strip().str.upper()
    # Normalize branch names: remove extra spaces
    df["branch"] = df["branch"].str.replace(r'\s+', ' ', regex=True).str.strip()

    df["round_num"] = df["round"].str.extract(r"(\d+)")[0].astype(int)
    df = df.dropna(subset=["round_num", "closing_rank"])

    # ---------------- ACTIVE BRANCH FILTER (2025) ----------------
    recent_df = df[df["year"] == 2025]
    active_keys = set(zip(recent_df["college_code"], recent_df["branch"]))

    df = df[df.apply(
        lambda r: (r["college_code"], r["branch"]) in active_keys,
        axis=1
    )]

    preds = []
    mae_records = []

    groups = df.groupby(
        ["college_code", "college_name", "branch", "category"]
    )

    print(f"✅ Processing {len(groups)} college–branch–category groups")

    # ---------------- GROUP LOOP ----------------
    for (c_code, c_name, branch, category), group in groups:

        available_rounds = sorted(group["round_num"].unique())

        for r in available_rounds:
            hist = group[group["round_num"] == r].sort_values("year")

            years = hist["year"].values
            cutoffs = hist["closing_rank"].values
            last_val = cutoffs[-1]

            # ---------------- ML + MAE ----------------
            if len(years) >= 3:
                # Train on all but last year
                train_years = years[:-1]
                train_cutoffs = cutoffs[:-1]

                test_year = years[-1]
                test_actual = cutoffs[-1]

                X_train = pd.DataFrame({
                    "year": train_years,
                    "year_index": np.arange(len(train_years))
                })

                X_test = pd.DataFrame({
                    "year": [test_year],
                    "year_index": [len(train_years)]
                })

                model = XGBRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=0.8,
                    objective="reg:squarederror",
                    random_state=42
                )

                model.fit(X_train, train_cutoffs)

                test_pred = model.predict(X_test)[0]
                mae_records.append({
                    "college_code": c_code,
                    "college_name": c_name,
                    "branch": branch,
                    "category": category,
                    "round": f"R{r}",
                    "mae": abs(test_actual - test_pred)
                })

                # Retrain on full data for 2026
                X_full = pd.DataFrame({
                    "year": years,
                    "year_index": np.arange(len(years))
                })

                model.fit(X_full, cutoffs)

                X_future = pd.DataFrame({
                    "year": [2026],
                    "year_index": [len(years)]
                })

                pred_val = model.predict(X_future)[0]

            elif len(years) == 2:
                # Linear ML fallback
                model = XGBRegressor(
                    booster="gblinear",
                    objective="reg:squarederror",
                    learning_rate=1.0,
                    random_state=42
                )

                model.fit(years[:-1].reshape(-1, 1), cutoffs[:-1])
                test_pred = model.predict([[years[-1]]])[0]

                mae_records.append({
                    "college_code": c_code,
                    "college_name": c_name,
                    "branch": branch,
                    "category": category,
                    "round": f"R{r}",
                    "mae": abs(cutoffs[-1] - test_pred)
                })

                model.fit(years.reshape(-1, 1), cutoffs)
                pred_val = model.predict([[2026]])[0]

            else:
                pred_val = last_val

            # ---------------- DOMAIN RULES ----------------
            is_rv_cse = (
                "R V College" in c_name and
                "Computer Science" in branch
            )

            if last_val < 2000:
                if is_rv_cse:
                    pred_val = max(pred_val, last_val + 5)
                    pred_val = min(pred_val, last_val * 1.15)
                else:
                    pred_val = max(1, last_val - 5)
            else:
                pred_val = max(pred_val, last_val + 5)
                pred_val = min(pred_val, last_val * 1.3)

            pred_val = int(max(1, pred_val))

            preds.append({
                "year": 2026,
                "round": f"R{r}",
                "college_code": c_code,
                "college_name": c_name,
                "branch": branch,
                "category": category,
                "predicted_closing_rank": pred_val
            })

    # ---------------- ROUND GAP ENFORCEMENT ----------------
    pred_df = pd.DataFrame(preds)
    final_rows = []

    for _, grp in pred_df.groupby(
        ["college_code", "branch", "category"]
    ):
        grp = grp.sort_values("round")
        prev = None

        for _, row in grp.iterrows():
            val = row["predicted_closing_rank"]
            if prev is not None:
                gap = max(50, int(prev * 0.05))
                if val < prev + gap:
                    val = prev + gap
            row["predicted_closing_rank"] = val
            prev = val
            final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    # ---------------- STORE RESULTS ----------------
    final_df.to_sql(
        "predictions_2026",
        engine,
        if_exists="replace",
        index=False
    )

    if mae_records:
        mae_df = pd.DataFrame(mae_records)
        mae_df.to_sql(
            "model_mae_2025_validation",
            engine,
            if_exists="replace",
            index=False
        )

        print("📊 MAE (Mean Absolute Error):", mae_df["mae"].mean())

    print("✅ Predictions & MAE stored successfully")


# -------------------- FETCH USER RESULTS --------------------
def fetch_predictions(rank, category, course_type="engineering"):
    sql = text("""
        SELECT
            college_code,
            college_name,
            branch,
            round,
            predicted_closing_rank
        FROM predictions_2026
        WHERE category = :category
          AND predicted_closing_rank >= :rank
    """)

    df = pd.read_sql_query(sql, engine, params={
        "rank": rank,
        "category": category
    })

    if not df.empty and course_type:
        # Filter based on course_type
        is_arch = df["branch"].str.contains(r"Architecture|B\.Arch", case=False, regex=True)
        if course_type == "architecture":
            df = df[is_arch]
        else:
            df = df[~is_arch]

    if df.empty:
        return []

    df["round_num"] = df["round"].str.extract(r"(\d+)")[0].astype(int)
    df.sort_values(["college_name", "branch", "round_num"], inplace=True)
    df.drop_duplicates(
        subset=["college_name", "branch"],
        keep="first",
        inplace=True
    )
    df.sort_values("predicted_closing_rank", inplace=True)

    return df.to_dict(orient="records")


# -------------------- ROUTE --------------------
@app.route("/", methods=["GET", "POST"])
def predictor():
    results = None
    error = None

    if request.method == "POST":
        try:
            rank = int(request.form["rank"])
            category = request.form["category"]
            course_type = request.form.get("course_type", "engineering")
            results = fetch_predictions(rank, category, course_type)
        except Exception:
            error = "Invalid input"

    return render_template("index.html", results=results, error=error)


# -------------------- MAIN --------------------
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        generate_predictions()
    app.run(debug=True)
