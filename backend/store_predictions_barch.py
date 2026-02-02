import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sqlalchemy import text
""" # from xgboost import XGBRegressor
"""

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

# -------------------- DB --------------------
        # Do not run generate_predictions_barch on startup
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.database import engine

# -------------------- ROUND ORDER --------------------
ROUND_ORDER = ["R1", "R2", "R3", "R4"]

# -------------------- ARCH / DESIGN CHECK --------------------
def is_arch_or_design(branch):
    return bool(
        pd.Series(branch)
        .str.contains(r"Architecture|B\.Arch|Design|B\.Des", case=False, regex=True)
        .iloc[0]
    )

# -------------------- COURSE CODE MAPPER --------------------
def get_course_code(branch):
    b = str(branch).upper()

    if "ARCH" in b:
        return "AT"      # Architecture
    if "DESIGN" in b:
        return "BDC"     # Bachelor of Design
    return b

# -------------------- GENERATE PREDICTIONS --------------------
def generate_predictions_barch():
    print("🏛 Generating Architecture & Design predictions...")

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
    df["round"] = df["round"].astype(str).str.upper().str.strip()
    df["category"] = df["category"].replace({"GM/KKR": "GM", "HKR": "KKR"})
    df["category"] = df["category"].str.upper().str.strip()
    df["branch"] = df["branch"].str.replace(r"\s+", " ", regex=True).str.strip()

    # ---------- FILTER ARCH / DESIGN ----------
    df = df[df["branch"].apply(is_arch_or_design)]

    df["round_num"] = df["round"].str.extract(r"(\d+)")[0].astype(int)
    df = df.dropna(subset=["round_num", "closing_rank"])

    # ---------- ACTIVE COLLEGES (2025) ----------
    recent_df = df[df["year"] == 2025]
    active_keys = set(zip(recent_df["college_code"], recent_df["branch"]))

    df = df[df.apply(
        lambda r: (r["college_code"], r["branch"]) in active_keys,
        axis=1
    )]

    preds = []

    groups = df.groupby(
        ["college_code", "college_name", "branch", "category"]
    )

    print(f"✅ Processing {len(groups)} Architecture / Design groups")

    # ---------------- GROUP LOOP ----------------
    for (c_code, c_name, branch, category), group in groups:

        available_rounds = sorted(group["round_num"].unique())

        for r in available_rounds:
            hist = group[group["round_num"] == r].sort_values("year")

            years = hist["year"].values
            cutoffs = hist["closing_rank"].values
            last_val = cutoffs[-1]

            # ---------------- ML LOGIC ----------------
            if len(years) >= 3:
                X_train = pd.DataFrame({
                    "year": years[:-1],
                    "year_index": np.arange(len(years) - 1)
                })
                y_train = cutoffs[:-1]

                model = XGBRegressor(
                    n_estimators=50,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=0.8,
                    objective="reg:squarederror",
                    random_state=42
                )

                model.fit(X_train, y_train)

                X_future = pd.DataFrame({
                    "year": [2026],
                    "year_index": [len(years)]
                })

                pred_val = model.predict(X_future)[0]

            elif len(years) == 2:
                model = XGBRegressor(
                    booster="gblinear",
                    objective="reg:squarederror",
                    learning_rate=1.0,
                    random_state=42
                )
                model.fit(years.reshape(-1, 1), cutoffs)
                pred_val = model.predict([[2026]])[0]

            else:
                pred_val = last_val

            # ---------------- DOMAIN RULES ----------------
            if last_val < 1500:
                pred_val = max(pred_val, last_val + 5)
                pred_val = min(pred_val, last_val * 1.15)
            else:
                pred_val = max(pred_val, last_val + 10)
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
                gap = max(100, int(prev * 0.08))
                if val < prev + gap:
                    val = prev + gap
            row["predicted_closing_rank"] = val
            prev = val
            final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    # ---------------- STORE ----------------
    final_df.to_sql(
        "predictions_2026_arch_design",
        engine,
        if_exists="replace",
        index=False
    )

    print("✅ Architecture & Design predictions stored")

# -------------------- FETCH RESULTS --------------------
def fetch_predictions_arch(rank, category):
    sql = text("""
        SELECT
            college_code,
            college_name,
            branch,
            round,
            predicted_closing_rank
        FROM predictions_2026_arch_design
        WHERE category = :category
          AND predicted_closing_rank >= :rank
    """)

    df = pd.read_sql_query(sql, engine, params={
        "rank": rank,
        "category": category
    })

    if df.empty:
        return []

    # Convert branch → course code
    df["course_code"] = df["branch"].apply(get_course_code)

    df["round_num"] = df["round"].str.extract(r"(\d+)")[0].astype(int)
    df.sort_values(["college_name", "round_num"], inplace=True)

    df.drop_duplicates(
        subset=["college_name", "course_code"],
        keep="first",
        inplace=True
    )

    df.sort_values("predicted_closing_rank", inplace=True)

    df.drop(columns=["branch"], inplace=True)
    df.rename(columns={"course_code": "branch"}, inplace=True)

    # Ensure 'category' is present in the output for each row
    if 'category' not in df.columns:
        df['category'] = category

    # Add course_full_name using code-to-name mapping from branches_data
    from backend.branches_data import architecture_branches, engineering_branches
    code_to_name = {b['code'].upper(): b['name'] for b in architecture_branches + engineering_branches}
    df['course_full_name'] = df['branch'].apply(lambda code: code_to_name.get(str(code).upper(), str(code)))

    return df.to_dict(orient="records")

# -------------------- ROUTE --------------------
@app.route("/barch", methods=["GET", "POST"])
def barch_predictor():
    results = None
    error = None

    if request.method == "POST":
        try:
            rank = int(request.form["rank"])
            category = request.form["category"]
            results = fetch_predictions_arch(rank, category)
        except Exception:
            error = "Invalid input"

    return render_template("barch.html", results=results, error=error)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        generate_predictions_barch()
    app.run(debug=True)
