import json
import os
import re

import pandas as pd
from flask import jsonify, render_template, request
from sqlalchemy import create_engine, inspect, text

from backend.app import app
from backend.branches_data import architecture_branches, engineering_branches
from backend.chatbot_ai import chatbot
from backend.college_agent import college_agent
from backend.colleges_data import architecture_colleges, colleges_list

# Removed unused imports
# from backend.prediction import get_college_courses
# from backend.prediction_2025 import predictor_2025
from backend.database import engine

# engine = create_engine(
#     "mssql+pyodbc://@localhost\\SQLEXPRESS/COMEDK_DB"
#     "?driver=ODBC+Driver+17+for+SQL+Server"
#     "&trusted_connection=yes"
# )

# Load enriched data if available
ENRICHED_DATA_FILE = os.path.join(os.path.dirname(__file__), 'college_data_enriched.json')
ENRICHED_DATA = {}

def load_enriched_data():
    global ENRICHED_DATA
    if os.path.exists(ENRICHED_DATA_FILE):
        try:
            with open(ENRICHED_DATA_FILE, 'r', encoding='utf-8') as f:
                ENRICHED_DATA = json.load(f)
            print(f"Loaded enriched data for {len(ENRICHED_DATA)} colleges.")
        except Exception as e:
            print(f"Error loading enriched data: {e}")

# Initial load
load_enriched_data()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})

@app.route('/api/college-data/<college_code>')
def get_college_data_api(college_code):
    # Reload data if empty or if college not found (to pick up new prefetch data)
    if not ENRICHED_DATA or college_code not in ENRICHED_DATA:
        load_enriched_data()

    # Check enriched data first (Part 3: Prefetched content)
    if college_code in ENRICHED_DATA:
        data = ENRICHED_DATA[college_code]
        return jsonify({
            "source": "explicit",
            "links": data['links'],
            "courses": data['courses']
        })

    # First check for explicit data (Part 1 & 2 of requirement)
    explicit_data = get_college_explicit_data(college_code)
    if explicit_data:
        return jsonify({
            "source": "explicit",
            "links": explicit_data['links'],
            "courses": explicit_data['courses']
        })

    # Fallback: Return basic info from static list without live fetching
    college = next((c for c in colleges_list if c['code'] == college_code), None)
    if not college:
        college = next((c for c in architecture_colleges if c['code'] == college_code), None)
    
    if not college:
        return jsonify({"error": "College not found"}), 404

    # Return basic structure if no enriched data available
    # The seeding agent should be run to populate this data
    return jsonify({
        "source": "explicit",
        "links": {
            "placement": None,
            "hostel": None,
            "infrastructure": None,
            "academics": None,
            "admissions": None,
            "contact": None
        },
        "courses": [],
        "website": college.get('website')
    })

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/exam-details')
def exam_details():
    return render_template('exam_details.html')

@app.route('/colleges')
def colleges():
    return render_template('colleges.html', engineering_colleges=colleges_list, architecture_colleges=architecture_colleges)


def get_college_courses_db(college_code):
    try:
        # Fetch unique branches for 2026 predictions (which represent 2025 active courses)
        with engine.connect() as conn:
            # Check columns first
            inspector = inspect(engine)
            columns = {col['name'] for col in inspector.get_columns('predictions_2026')}
            
            if 'branch_code' in columns:
                query = text("SELECT DISTINCT branch, branch_code FROM predictions_2026 WHERE college_code = :code AND branch NOT LIKE '%Arch%' ORDER BY branch")
                result = conn.execute(query, {"code": college_code})
                courses = [{"name": row[0], "code": row[1]} for row in result]
            else:
                query = text("SELECT DISTINCT branch FROM predictions_2026 WHERE college_code = :code AND branch NOT LIKE '%Arch%' ORDER BY branch")
                result = conn.execute(query, {"code": college_code})
                courses = [{"name": row[0], "code": None} for row in result]
        return courses
    except Exception as e:
        print(f"Error fetching courses for {college_code}: {e}")
        return []

@app.route('/college/<college_code>')
def college_details(college_code):
    # Find college details from the static list
    college = next((c for c in colleges_list if c['code'] == college_code), None)
    if not college:
        college = next((c for c in architecture_colleges if c['code'] == college_code), None)
        
    if not college:
        return "College not found", 404
        
    # Use DB function instead of deleted module
    courses = get_college_courses_db(college_code)

    # Always reload enriched data to ensure latest updates are shown
    load_enriched_data()
    
    enriched_data = None
    if college_code in ENRICHED_DATA:
        enriched_data = ENRICHED_DATA[college_code]
    
    if not enriched_data:
         # Empty structure
         enriched_data = {
            "links": {
                "placement": None, "hostel": None, "infrastructure": None,
                "academics": None, "admissions": None, "contact": None
            },
            "courses": []
         }


    # Merge explicit data (prioritize explicit links)
    # EXPLICIT DATA LOGIC REMOVED AS REQUESTED
    
    return render_template('college_details.html', college=college, courses=courses, enriched_data=None)

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    error = None
    results = []

    if request.method == 'POST':
        try:
            rank = int(request.form.get('rank', '').strip())
        except (TypeError, ValueError):
            rank = None
            error = "Please enter a valid numeric rank."

        category = request.form.get('category')
        course_type = request.form.get('course_type', 'engineering')

        if rank is not None and category:
            if course_type == 'architecture':
                # Use B.Arch/Architecture predictions from store_predictions_barch
                from backend.store_predictions_barch import fetch_predictions_arch
                results = fetch_predictions_arch(rank, category)
            else:
                inspector = inspect(engine)
                columns = {col['name'] for col in inspector.get_columns('predictions_2026')}

                # Fallback if legacy table lacks category column
                if 'category' in columns:
                    # Check for branch_code column
                    has_branch_code = 'branch_code' in columns
                    select_clause = "college_code, college_name, branch, " + ("branch_code, " if has_branch_code else "") + "round, category, predicted_closing_rank"
                    sql = text(
                        f"""
                        SELECT {select_clause}
                        FROM predictions_2026
                        WHERE predicted_closing_rank >= :rank
                          AND category = :category
                        ORDER BY predicted_closing_rank ASC
                        """
                    )
                    params = {"rank": rank, "category": category}
                else:
                    # Check for branch_code column
                    has_branch_code = 'branch_code' in columns
                    select_clause = "college_code, college_name, branch, " + ("branch_code, " if has_branch_code else "") + "round, NULL AS category, predicted_closing_rank"
                    sql = text(
                        f"""
                        SELECT {select_clause}
                        FROM predictions_2026
                        WHERE predicted_closing_rank >= :rank
                        ORDER BY predicted_closing_rank ASC
                        """
                    )
                    params = {"rank": rank}

                with engine.begin() as conn:
                    df = pd.read_sql_query(sql, conn, params=params)
                # ...existing code for filtering, mapping, and results...
                if not df.empty:
                    arch_codes = [b['code'] for b in architecture_branches]
                    branch_col = 'branch'
                    if 'branch_code' in df.columns:
                        branch_col = 'branch_code'
                    branch_name_col = None
                    for col in ['normalized_branch', 'branch', 'branch_name']:
                        if col in df.columns:
                            branch_name_col = col
                            break
                    if branch_name_col is None:
                        branch_name_col = branch_col
                    if course_type == 'engineering':
                        df = df[~(
                            (df[branch_col].astype(str).str.upper() == 'AT') |
                            (df[branch_name_col].astype(str).str.upper().str.contains('B.ARCH|ARCHITECTURE', regex=True))
                        )]
                    df['round_num'] = df['round'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
                    all_colleges = colleges_list + architecture_colleges
                    code_to_name = {str(c['code']).strip().upper(): c['name'] for c in all_colleges}
                    name_to_code = {c['name'].strip().upper(): c['code'] for c in all_colleges}
                    if 'college_code' not in df.columns:
                        df['college_code'] = df['branch'].apply(lambda x: 'UNKNOWN')
                    if 'college_name' not in df.columns:
                        def infer_college_name(row):
                            branch = str(row.get('branch', '')).strip().upper()
                            for name in name_to_code:
                                if branch in name:
                                    return name
                            return 'Unknown College'
                        df['college_name'] = df.apply(infer_college_name, axis=1)
                    def get_official_name(row):
                        code = str(row.get('college_code', '')).strip().upper()
                        name = str(row.get('college_name', '')).strip()
                        if code in code_to_name:
                            return code_to_name[code]
                        if name:
                            return name
                        return 'Unknown College'
                    df['college_name'] = df.apply(get_official_name, axis=1)
                    def get_official_code(row):
                        name = str(row.get('college_name', '')).strip().upper()
                        if name in name_to_code:
                            return name_to_code[name]
                        return row.get('college_code', 'UNKNOWN')
                    df['college_code'] = df.apply(get_official_code, axis=1)
                    
                    # Sort Descending first to prioritize higher (optimistic) ranks during deduplication
                    df.sort_values(by=['predicted_closing_rank'], ascending=False, inplace=True)
                    
                    df['branch'] = df['branch'].replace(['B.ARCH', 'B.Arch', 'Architecture'], 'Bachelor of Architecture (B.Arch)')
                    results = df[['college_name', 'college_code', 'branch', branch_col, branch_name_col, 'round', 'category', 'predicted_closing_rank']].to_dict(orient='records')
                     
                    # 1. Create lookup map for standardizing branch names by CODE
                    standard_branches_by_code = {b['code']: b['name'] for b in engineering_branches + architecture_branches}
                    
                    # 2. Create lookup map for standardizing branch names by SIMPLIFIED TEXT
                    # This handles "Communicati on" etc by matching "electronics&communicationengineering"
                    def simplify_text(text):
                        if not isinstance(text, str): return ""
                        # Remove all non-alphanumeric characters and lowercase
                        return re.sub(r'[^a-z0-9]', '', text.lower())

                    standard_branches_by_name = {}
                    for b in engineering_branches + architecture_branches:
                        # Map simplified name -> (Code, Standard Name)
                        rectified_name = b['name']
                        simple = simplify_text(rectified_name)
                        if simple:
                            standard_branches_by_name[simple] = (b['code'], rectified_name)

                    # List of common broken words to fix via regex if lookup fails
                    # (Fallback for when the branch standard isn't in our list)
                    common_fixes = [
                        (r'Telecomm[- ]?unication', 'Telecommunication'),
                        (r'Communicati[- ]?on', 'Communication'),
                        (r'Engin[- ]?eering', 'Engineering'),
                        (r'Techno[- ]?logy', 'Technology'),
                        (r'Inform[- ]?ation', 'Information'),
                        (r'Artific[- ]?ial', 'Artificial'),
                        (r'Intell[- ]?lgence', 'Intelligence'),
                        (r'Mechan[- ]?ical', 'Mechanical'),
                        (r'Electr[- ]?ical', 'Electrical'),
                        (r'Electr[- ]?onics', 'Electronics'),
                        (r'Comp[- ]?uter', 'Computer'),
                        (r'Scien[- ]?ce', 'Science'),
                    ]

                    # Normalize branch names for deduplication
                    for r in results:
                        if 'branch' in r and isinstance(r['branch'], str):
                            raw_branch = r['branch'].strip()
                            clean_branch = raw_branch.replace('- ', '-').strip()
                            
                            found_standard = False
                            
                            # Strategy A: Check Code Prefix (Most Reliable)
                            # e.g. "EC - Electronics..." -> "EC"
                            parts = clean_branch.split('-', 1)
                            if len(parts) > 1:
                                code_prefix = parts[0].strip().upper()
                                # Check if code matches a known standard
                                if code_prefix in standard_branches_by_code:
                                    # Replace with standard name: CODE-Standard Name
                                    r['branch'] = f"{code_prefix}-{standard_branches_by_code[code_prefix]}"
                                    found_standard = True

                            # Strategy B: Simplified Name Match (Robust against spaces/hyphens)
                            # e.g. "EC-Electronics & Communicati on" -> "electronicscommunication"
                            if not found_standard:
                                # Try matching the text part (after hyphen if exists, else full string)
                                text_part = parts[1] if len(parts) > 1 else clean_branch
                                simple_text = simplify_text(text_part)
                                
                                if simple_text in standard_branches_by_name:
                                    code, name = standard_branches_by_name[simple_text]
                                    r['branch'] = f"{code}-{name}"
                                    found_standard = True
                                else:
                                    # Try simplifying the *entire* string (ignoring potential code prefix if it was garbled)
                                    simple_full = simplify_text(clean_branch)
                                    if simple_full in standard_branches_by_name:
                                         code, name = standard_branches_by_name[simple_full]
                                         r['branch'] = f"{code}-{name}"
                                         found_standard = True

                            # Strategy C: Regex Pattern Fixes (Fallback)
                            # Fixes "Bro ken Wor ds" even if we can't map to a standard branch
                            if not found_standard:
                                fixed_branch = clean_branch
                                for pattern, replacement in common_fixes:
                                    fixed_branch = re.sub(pattern, replacement, fixed_branch, flags=re.IGNORECASE)
                                r['branch'] = fixed_branch

                    # Remove duplicates based on college_name, college_code, branch, round, category
                    seen = set()
                    unique_results = []
                    for r in results:
                        key = (
                            r.get('college_name'),
                            r.get('college_code'),
                            r.get('branch'),
                            r.get(branch_col),
                            r.get(branch_name_col),
                            r.get('round'),
                            r.get('category')
                        )
                        if key not in seen:
                            seen.add(key)
                            unique_results.append(r)
                    results = unique_results
                    # Only show the earliest round for each (college_name, college_code, branch)
                    filtered = {}
                    round_order = {'R1': 1, 'R2': 2, 'R3': 3, 'R4': 4}
                    for r in results:
                        key = (r.get('college_name'), r.get('college_code'), r.get('branch'))
                        round_val = str(r.get('round', '')).upper()
                        round_num = round_order.get(round_val, 99)
                        if key not in filtered or round_num < filtered[key][1]:
                            filtered[key] = (r, round_num)
                    results = [v[0] for v in filtered.values()]
                    
                    # Sort final results by rank (Best/Lowest Cutoff first)
                    results.sort(key=lambda x: x.get('predicted_closing_rank', 999999))
                else:
                    results = []
        elif not error:
            error = "Rank and category are required."

        return render_template('results.html', results=results, rank=rank, category=category, branch="All Courses", error=error)
    return render_template('predictor.html', engineering_branches=engineering_branches, architecture_branches=architecture_branches)

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/courses')
def courses():
    return render_template('courses.html', engineering_branches=engineering_branches, architecture_branches=architecture_branches)
