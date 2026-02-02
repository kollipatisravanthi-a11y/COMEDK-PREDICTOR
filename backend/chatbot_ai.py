import json
import random
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from backend.colleges_data import colleges_list, architecture_colleges
# from backend.utils import load_comedk_data
# from backend.prediction import predict_colleges
from sqlalchemy import create_engine, text
from backend.database import engine

# DB Configuration - Removed hardcoded MSSQL details
# DATABASE = 'COMEDK_DB'
# engine = create_engine(f'mssql+pyodbc://@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes')

# Common abbreviations mapping
COLLEGE_ABBREVIATIONS = {
    'rvce': 'RV College of Engineering',
    'msrit': 'M.S. Ramaiah Institute of Technology',
    'bmsce': 'BMS College of Engineering',
    'pesit': 'PES Institute of Technology',
    'dsce': 'Dayananda Sagar College of Engineering',
    'bit': 'Bangalore Institute of Technology',
    'sit': 'Siddaganga Institute of Technology',
    'nie': 'National Institute of Engineering',
    'uvce': 'University Visvesvaraya College of Engineering',
    'jss': 'JSS Science and Technology University'
}

def predict_colleges(rank, branch=None, category='GM', course_type=None):
    """
    Predict colleges based on rank using the predictions_2026 table.
    """
    try:
        with engine.connect() as conn:
            # Construct query
            query_str = """
            SELECT college_name, branch, predicted_closing_rank, round 
            FROM predictions_2026 
            WHERE predicted_closing_rank >= :rank 
            """
            params = {"rank": rank}
            
            if category:
                query_str += " AND category = :category"
                params["category"] = category
                
            if branch:
                query_str += " AND branch = :branch"
                params["branch"] = branch

            # Course Type Filtering
            if course_type:
                if course_type == 'architecture':
                    # Include Arch, Design, Planning
                    query_str += " AND (branch LIKE '%Arch%' OR branch LIKE '%Design%' OR branch LIKE '%Plan%')"
                elif course_type == 'engineering':
                    # Exclude Arch, Design, Planning
                    query_str += " AND branch NOT LIKE '%Arch%' AND branch NOT LIKE '%Design%' AND branch NOT LIKE '%Plan%'"
            
            # Order by Cutoff ASC (Better colleges first)
            query_str += " ORDER BY predicted_closing_rank ASC"
            
            # Limit results - increased to allow grouping
            query_str += " LIMIT 100" if 'sqlite' in str(engine.url) else " OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY"
            
            result = conn.execute(text(query_str), params)
            predictions = []
            
            for row in result:
                predictions.append({
                    "college": row[0],
                    "branch": row[1],
                    "cutoff": row[2],
                    "round": row[3],
                    "location": "Karnataka", 
                    "probability": "High"
                })
            return predictions
    except Exception as e:
        print(f"Prediction Error: {e}")
        return []

class ChatBot:
    def __init__(self):
        self.intents = []
        self.model = None
        self.colleges_df = None
        self.enriched_web_data = {}
        self.model_name = "GPT-5.1-Codex-Max"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.base_dir, 'chatbot_model.pkl')
        
        print(f"Enabled {self.model_name} for all clients")
        self.load_resources()
        
        # Try to load existing model, otherwise train
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print("Loaded saved chatbot model.")
            except Exception as e:
                print(f"Failed to load model ({e}), retraining...")
                self.train_model()
        else:
            self.train_model()

    def load_resources(self):
        # Load intents
        intents_path = os.path.join(self.base_dir, 'intents.json')
        try:
            with open(intents_path, 'r') as f:
                data = json.load(f)
            self.intents = data['intents']
        except Exception as e:
            print(f"Error loading intents: {e}")

        # Load college data
        enriched_path = os.path.join(self.base_dir, '../data/processed/linear_model_results_enriched.csv')
        if os.path.exists(enriched_path):
             try:
                 self.colleges_df = pd.read_csv(enriched_path)
                 print(f"Chatbot loaded enriched linear model results from {enriched_path}")
             except Exception as e:
                 print(f"Error loading csv: {e}")
                 self.colleges_df = pd.DataFrame()
        else:
             print("Chatbot: Enriched data not found. Please run result generation.")
             self.colleges_df = pd.DataFrame() 
             
        # Load Enriched Web Data (Placements, Hostels etc.)
        enriched_json_path = os.path.join(self.base_dir, 'college_data_enriched.json')
        if os.path.exists(enriched_json_path):
            try:
                with open(enriched_json_path, 'r', encoding='utf-8') as f:
                    self.enriched_web_data = json.load(f)
                print(f"Chatbot loaded enriched web data for {len(self.enriched_web_data)} colleges.")
            except Exception as e:
                print(f"Error loading enriched web data: {e}")

    def train_model(self):
        print("Training chatbot model...")
        patterns = []
        tags = []
        
        try:
            for intent in self.intents:
                for pattern in intent['patterns']:
                    patterns.append(pattern)
                    tags.append(intent['tag'])
            
            # --- Augment training data with College names and Ranks ---
            
            # Add abbreviations to training data
            for abbr, full_name in COLLEGE_ABBREVIATIONS.items():
                patterns.append(abbr)
                tags.append('colleges')
                patterns.append(f"cutoff for {abbr}")
                tags.append('cutoff')
                # Removed "courses in {abbr}" to avoid biasing generic "courses" query towards 'colleges' intent.
                # Specific queries like "courses in rvce" are handled by get_college_info logic anyway.
                patterns.append(f"Tell me about {abbr}")
                tags.append('colleges')

            for college in colleges_list:
                full_name = college.get('name', '')
                main_name = full_name.split('-')[0].strip()
                
                if main_name:
                    patterns.append(main_name)
                    tags.append('colleges')
                    patterns.append(f"Tell me about {main_name}")
                    tags.append('colleges')
                    patterns.append(f"cutoff for {main_name}")
                    tags.append('cutoff')
                    # Removed "courses in {main_name}" to avoid biasing generic "courses" query towards 'colleges' intent.
                
                code = college.get('code', '')
                if code:
                    patterns.append(code)
                    tags.append('colleges')

            # Add Rank patterns
            for r in [1, 100, 1000, 5000, 10000, 20000, 50000, 100000]:
                patterns.append(f"Rank {r}")
                tags.append('rank')
                patterns.append(f"My rank is {r}")
                tags.append('rank')
                patterns.append(f"predict for {r}")
                tags.append('rank')

            # Create and train pipeline
            self.model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
            self.model.fit(patterns, tags)
            
            # Save model
            joblib.dump(self.model, self.model_path)
            print(f"Chatbot model trained successfully and saved to {self.model_path}")

        except Exception as e:
            print(f"Error training chatbot: {e}")
            self.model = None

    def get_college_info(self, message):
        message = message.lower()
        found_college = None
        
        # Check for abbreviations
        for abbr, full_name in COLLEGE_ABBREVIATIONS.items():
            # Check for abbr as a distinct word
            if re.search(r'\b' + re.escape(abbr) + r'\b', message):
                # Replace abbr with part of full name to help matching below
                # or just use full_name for matching logic
                message = message.replace(abbr, full_name.lower())
        
        # Check against the comprehensive list from colleges_data.py
        for college in colleges_list:
            # Simple fuzzy matching: check if college name parts are in message
            # Or check if a significant part of the college name is in the message
            college_name_lower = college['name'].lower()
            
            # Extract main name (e.g., "BMS College of Engineering" from "BMS College of Engineering-Basavanagudi...")
            main_name = college_name_lower.split('-')[0].strip()
            
            if main_name in message:
                found_college = college
                break
            
            # Also check for common abbreviations if possible, or just partial match
            # For now, let's stick to the main name check
            
        if found_college:
            response = f"**{found_college['name']}**\n"
            response += f"Location: {found_college['location']}\n"
            
            # Check for specific info types in user message
            info_type = None
            if any(w in message for w in ["placement", "package", "salary", "recruiters"]):
                info_type = "placement"
            elif any(w in message for w in ["hostel", "accommodation", "dorm"]):
                info_type = "hostel"
            elif any(w in message for w in ["infrastructure", "campus", "facilities"]):
                info_type = "infrastructure"
            elif any(w in message for w in ["academic", "curriculum", "faculty"]):
                info_type = "academics"
            elif any(w in message for w in ["admission", "eligibility", "process"]):
                info_type = "admissions"
                
            # If enriched data exists and users asks for specific info
            code = found_college['code']
            if info_type and self.enriched_web_data and code in self.enriched_web_data:
                details = self.enriched_web_data[code].get('links', {}).get(info_type)
                if details:
                    if details.get('url'):
                        response += f"\nHere is the official {info_type} information: {details['url']}\n"
                    if details.get('content') and len(details['content']) > 20:
                        # Extract a snippet
                        snippet = details['content'][:300].replace('\n', ' ') + "..."
                        response += f"Snippet: {snippet}\n"
                    else:
                        response += f"\n(Further details available on the official website)\n"
                else:
                    response += f"\nSpecific {info_type} details are not currently indexed. Please visit: {found_college.get('website', 'official website')}\n"
            
            # Look up courses in the CSV if no specific info requested or just general
            elif not info_type and not self.colleges_df.empty:
                # Normalize columns for access
                df = self.colleges_df.copy()
                df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
                
                # Check column existence
                code_col = 'college_code' if 'college_code' in df.columns else None
                course_col = 'course' if 'course' in df.columns else None
                
                if code_col and course_col:
                    college_courses = df[df[code_col] == found_college['code']]
                    
                    if not college_courses.empty:
                        response += "\nAvailable Courses:\n"
                        courses = college_courses[course_col].unique()
                        for course in courses:
                            response += f"- {course}\n"
            
            # Helper text
            if not info_type:
                 response += "\nYou can also ask about **placements**, **hostels**, or **fees** for this college."
                
            return response
            
        return None

    def get_response(self, message):
        if not self.model:
            return "I am currently under maintenance. Please try again later."
            
        if not message:
            return "Please say something."
            
        try:
            msg_lower = message.lower()
            
            # --- General Category Counts (User Request) ---
            # Check for pure category queries (without rank numbers)
            is_rank_query = bool(re.search(r'\d+', msg_lower))
            
            if not is_rank_query:
                if "b.arch" in msg_lower or "architecture" in msg_lower:
                    return f"We have data on **{len(architecture_colleges)}** Architecture colleges. You can explore the list of all participating colleges in the **Colleges** section."
                
                
                
                if any(k in msg_lower for k in ["b.e", "b.tech", "engineering", "btech", "be"]):
                    return f"We have data on **{len(colleges_list)}** Engineering colleges including top institutes like RVCE, BMSCE, and MSRIT. Check the **Colleges** tab for more details."

            # Check for rank prediction request
            # Look for "rank" followed by number, or just a number if it looks like a rank
            rank_match = re.search(r'rank\s*[:is]?\s*(\d+)', msg_lower)
            if not rank_match:
                # Try to find just a number if the message is short (e.g. "5000")
                if message.strip().isdigit():
                    rank_match = re.search(r'(\d+)', message)
            
            if rank_match:
                rank = int(rank_match.group(1))

                # Identify requested course type
                msg_lower = message.lower()
                course_type = None
                
                arch_keywords = ["architecture", "b.arch", "b arch", "at"]
                design_keywords = ["design", "b.des", "b des"] # Usually handled within architecture type logic for grouping
                eng_keywords = ["engineering", "b.e", "b.tech", "b tech", "be", "technology"]

                if any(k in msg_lower for k in arch_keywords + design_keywords):
                    course_type = 'architecture'
                elif any(k in msg_lower for k in eng_keywords):
                    course_type = 'engineering'

                # Call prediction logic with detected type
                results = predict_colleges(rank, None, 'GM', course_type=course_type)
                
                # --- Categorize Results ---
                # Keywords for categorization
                # Re-defining here for grouping logic
                arch_keys = ["architecture", "b.arch"]
                design_keys = ["bachelor of design", "b.des", "design"]
                planning_keys = ["planning", "b.plan", "urban"]

                eng_list = []
                arch_list = []
                design_list = []
                plan_list = []

                for res in results:
                    branch_lower = res['branch'].lower()
                    if any(k in branch_lower for k in design_keys):
                        design_list.append(res)
                    elif any(k in branch_lower for k in planning_keys):
                        plan_list.append(res)
                    elif any(k in branch_lower for k in arch_keys):
                        arch_list.append(res)
                    else:
                        eng_list.append(res)

                response = f"Entered Rank: {rank}\n"
                if course_type:
                     response += f" (Filtered for {course_type.title()})\n\n"
                else:
                     response += "\n"

                # Helper to format list
                def format_list(lst, limit=10):
                    txt = ""
                    # Sort primarily by cutoff
                    lst.sort(key=lambda x: x['cutoff']) # Ascending cutoff (Better rank first)
                    for i, r in enumerate(lst[:limit], 1):
                         txt += f"{i}. {r['college']} – {r['branch']}\n   (Cutoff: {r['cutoff']}, Round: {r['round']})\n"
                    return txt

                has_content = False

                # 1. Architecture Results
                if arch_list:
                    response += "**Architecture (B.Arch)**\n"
                    response += format_list(arch_list)
                    response += "\n"
                    has_content = True

                # 2. Design Results
                if design_list:
                    response += "**Design Courses**\n"
                    response += format_list(design_list)
                    response += "\n*Note: Design ranks may differ significantly from Architecture norms due to intake patterns.*\n"
                    has_content = True
                
                # 3. Planning Results
                if plan_list:
                    response += "**Planning Courses**\n"
                    response += format_list(plan_list)
                    response += "\n*Note: Planning ranks sort separately.*\n"
                    has_content = True

                # 4. Engineering
                # If filtered for Architecture, this list should be empty or ignored unless specific overlap
                if eng_list:
                    # Only show engineering if:
                    # a) User asked for Engineering
                    # b) User didn't specify type (show generic)
                    # c) User asked for Arch but somehow we got Eng results (shouldn't happen with SQL filter)
                    
                    if course_type == 'engineering' or not course_type:
                        if has_content:
                            response += "\n**Engineering Courses** (Top matches)\n"
                        else:
                            response += "**Eligible Engineering Colleges**:\n"
                        response += format_list(eng_list, limit=10)
                        has_content = True

                if not has_content:
                    response += "Based on historical data, no colleges found for this rank."

                return response

            # First, check if the user is asking about a specific college
            college_response = self.get_college_info(message)
            if college_response:
                return college_response

            # Predict intent
            probs = self.model.predict_proba([message])[0]
            max_prob = np.max(probs)
            predicted_tag = self.model.classes_[np.argmax(probs)]
            
            if max_prob < 0.3: 
                return "I'm not sure I understand. You can ask me about COMEDK exam, colleges, cutoffs, or how to use this predictor."
                
            for intent in self.intents:
                if intent['tag'] == predicted_tag:
                    return random.choice(intent['responses'])
                    
            return "I'm sorry, can you ask about colleges, cutoffs etc?"
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Sorry, I encountered an error."

chatbot = ChatBot()

if __name__ == '__main__':
    bot = ChatBot()
    bot.train_model()
