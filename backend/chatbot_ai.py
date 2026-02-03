import json
import random
import os
import re
import joblib
import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from backend.colleges_data import colleges_list, architecture_colleges
from backend.branches_data import engineering_branches, architecture_branches
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
        self.context = {}  # Stores session context: {'last_college': None, 'topic': None}
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

            all_colleges = colleges_list + architecture_colleges
            for college in all_colleges:
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

    def find_college_match(self, message):
        """
        Robust fuzzy matching for college names using a scoring system.
        """
        message_lower = message.lower()
        # Normalization for spaced initials (r v -> rv, b m s -> bms, m s -> ms)
        message_clean = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', message_lower)
        message_clean = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', message_clean) # Repeat for 3 letters (b m s)
        message_clean = re.sub(r'[^a-z0-9\s]', '', message_clean)
        
        best_college = None
        best_score = 0
        
        # Stopwords: Common words that don't help much in distinguishing colleges
        # "college", "institute" are kept out of stopwords as they might help distinguish "BMS College" from "BMS Institute"
        stopwords = {"of", "and", "the", "in", "at", "for", "dist", "road", "main", "cross", "dr", "sri", "smt", "shri", "bengaluru", "bangalore", "mysore", "karnataka"}

        message_tokens = set(message_clean.split())
        relevant_message_tokens = message_tokens - stopwords

        # Combine both lists for search
        all_colleges = colleges_list + architecture_colleges

        for college in all_colleges:
            current_score = 0
            
            # 1. Code Match (Exact) - Highest Priority
            code = str(college['code']).lower()
            if code in message_tokens:
                 return college # Instant match for code
                
            name_lower = college['name'].lower()
            # Normalize Name too (r v -> rv)
            name_norm = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', name_lower)
            name_norm = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', name_norm)
            
            # Clean name for tokenization
            name_clean = re.sub(r'[^a-z0-9\s]', '', name_norm) 
            
            # Use the part before hyphen if avail for primary name, but check full name too
            simple_name = name_lower.split('-')[0].strip()
            
            # 2. Abbreviation Match
            # Check if any known abbreviation maps to this specific college name
            for abbr, full_name in COLLEGE_ABBREVIATIONS.items():
                # Normalize full name in abbreviation map too just in case
                if full_name.lower().replace(" ", "") in name_lower.replace(" ", "") and abbr in message_tokens:
                    current_score += 50 # Strong signal
            
            # 3. Token Overlap Score
            college_tokens = set(name_clean.split())
            relevant_college_tokens = college_tokens - stopwords
            
            if relevant_college_tokens and relevant_message_tokens:
                # Intersection of meaningful words (e.g. "acharya", "technology")
                overlap = relevant_college_tokens.intersection(relevant_message_tokens)
                
                # Weight matches
                current_score += len(overlap) * 10
                
                # Bonus: First word match (Very important for "Acharya", "BMS", "RV")
                if message_tokens and college_tokens:
                    m_parts = message_clean.split()
                    c_parts = name_clean.split()
                    
                    if m_parts and c_parts and m_parts[0] == c_parts[0]:
                         if m_parts[0] not in {"sri", "smt", "dr", "govt", "government", "university", "college", "institute"}:
                             current_score += 20

                # Bonus for sequential substring match
                # Use normalized simple name check
                simple_norm = re.sub(r'\b([a-z])\s([a-z])\b', r'\1\2', simple_name)
                simple_norm = re.sub(r'[^a-z0-9\s]', '', simple_norm)
                
                if simple_norm in message_clean:
                    current_score += 15
            
            # Update best candidate
            if current_score > best_score:
                best_score = current_score
                best_college = college
        
        # Threshold: At least one significant word match (score >= 10)
        if best_score >= 10:
            return best_college
            
        return None

    def get_college_cutoff_stats(self, college_code, year="2026"):
        """
        Fetch basic statistics for a college's cutoff to answer 'what is cutoff for X'
        """
        try:
            with engine.connect() as conn:
                # Get range of cutoffs for GM category
                query = text(f"""
                    SELECT branch, MIN(predicted_closing_rank), MAX(predicted_closing_rank)
                    FROM predictions_{year}
                    WHERE college_code = :code AND category = 'GM'
                    GROUP BY branch
                    ORDER BY MIN(predicted_closing_rank) ASC
                """)
                # Using LIMIT in python as SQLite/SQLServer syntax varies
                result = conn.execute(query, {"code": college_code})
                rows = result.fetchall()
                if not rows:
                    return "No cutoff data available for this college."
                    
                summary = "**Expected Cutoffs (GM):**\n"
                # Show top 5 branches
                for row in rows[:5]:
                    branch = row[0]
                    # Clean branch name
                    if '-' in branch: branch = branch.split('-', 1)[1]
                    summary += f"- {branch[:25]}..: ~{row[1]}\n"
                
                if len(rows) > 5:
                    summary += f"*(and {len(rows)-5} more courses)*"
                return summary
        except Exception as e:
            print(f"Error fetching cutoffs: {e}")
            return None

    def get_available_branches(self, college_code):
        """Fetch distinct branches for a college from DB with normalization"""
        try:
            with engine.connect() as conn:
                query = text("SELECT DISTINCT branch FROM predictions_2026 WHERE college_code = :code ORDER BY branch")
                result = conn.execute(query, {"code": college_code})
                raw_branches = [row[0] for row in result if row[0]]
                
                # --- Normalization Logic (Copied/Adapted from routes.py) ---
                
                # 1. Maps
                standard_branches_by_code = {b['code']: b['name'] for b in engineering_branches + architecture_branches}
                
                def simplify_text(text):
                    if not isinstance(text, str): return ""
                    return re.sub(r'[^a-z0-9]', '', text.lower())

                standard_branches_by_name = {}
                for b in engineering_branches + architecture_branches:
                    rectified_name = b['name']
                    simple = simplify_text(rectified_name)
                    if simple:
                        standard_branches_by_name[simple] = (b['code'], rectified_name)
                        
                # 2. Fixes
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

                unique_map = {}
                
                for branch in raw_branches:
                    raw_branch = branch.strip()
                    clean_branch = raw_branch.replace('- ', '-').strip()
                    final_branch = clean_branch # Default
                    found_standard = False
                    
                    # Strategy A: Code Prefix
                    parts = clean_branch.split('-', 1)
                    if len(parts) > 1:
                        code_prefix = parts[0].strip().upper()
                        if code_prefix in standard_branches_by_code:
                            final_branch = f"{code_prefix}-{standard_branches_by_code[code_prefix]}"
                            found_standard = True
                    
                    # Strategy B: Simplified Match
                    if not found_standard:
                        text_part = parts[1] if len(parts) > 1 else clean_branch
                        simple_text = simplify_text(text_part)
                        
                        if simple_text in standard_branches_by_name:
                             code, name = standard_branches_by_name[simple_text]
                             final_branch = f"{code}-{name}"
                             found_standard = True
                        else:
                             simple_full = simplify_text(clean_branch)
                             if simple_full in standard_branches_by_name:
                                  code, name = standard_branches_by_name[simple_full]
                                  final_branch = f"{code}-{name}"
                                  found_standard = True
                                  
                    # Strategy C: Regex Fixes
                    if not found_standard:
                        for pattern, replacement in common_fixes:
                           final_branch = re.sub(pattern, replacement, final_branch, flags=re.IGNORECASE)

                    # Deduplicate using simplified key to catch "AS- Aero..." vs "AS-Aero..."
                    dedupe_key = simplify_text(final_branch)
                    if dedupe_key not in unique_map:
                         unique_map[dedupe_key] = final_branch
                    else:
                         # Prefer longer/cleaner version? Or just keep first.
                         pass

                # Return sorted values
                return sorted(list(unique_map.values()))
                
        except Exception as e:
            print(f"Error fetching branches for {college_code}: {e}")
            return []

    def get_college_info(self, message):
        message = message.lower()
        
        # 1. Identify College
        found_college = self.find_college_match(message)
        
        # 2. Contextual Fallback (e.g. "what about placements there?")
        if not found_college and self.context.get('last_college'):
            context_words = ['it', 'that', 'there', 'college', 'institute', 'campus']
            if any(w in message for w in context_words):
                found_college = self.context['last_college']

        if found_college:
            # Update Context
            self.context['last_college'] = found_college
            
            response = f"**{found_college['name']}**\n"
            response += f"**Code:** {found_college['code']}\n"
            response += f"**Location:** {found_college['location']}\n"
            
            # Add About if available
            if found_college.get('about'):
                response += f"\n{found_college['about']}\n"

            # Add Website
            if found_college.get('website'):
                response += f"\n**Official Website:** {found_college['website']}\n"
            
            # Check for specific info types
            info_type = None
            if any(w in message for w in ["placement", "package", "salary", "recruiters", "job"]):
                info_type = "placement"
            elif any(w in message for w in ["hostel", "accommodation", "dorm", "mess"]):
                info_type = "hostel"
            elif any(w in message for w in ["infrastructure", "campus", "facilities", "labs"]):
                info_type = "infrastructure"
            elif any(w in message for w in ["academic", "curriculum", "faculty"]):
                info_type = "academics"
            elif any(w in message for w in ["admission", "eligibility", "process", "seat"]):
                info_type = "admissions"
            elif any(w in message for w in ["cutoff", "rank", "closing"]):
                info_type = "cutoff"
                
            code = found_college['code']
            
            # Handle Cutoff Query specifically
            if info_type == "cutoff":
                stats = self.get_college_cutoff_stats(code)
                if stats:
                    response += f"\n{stats}\n"
                else:
                    response += "\nCutoff data not currently available.\n"
            
            # Handle Web Data (Enriched)
            elif info_type and self.enriched_web_data and code in self.enriched_web_data:
                details = self.enriched_web_data[code].get('links', {}).get(info_type)
                if details:
                    if details.get('url'):
                        response += f"\nHere is the official {info_type} information: {details['url']}\n"
                    if details.get('content') and len(details['content']) > 20:
                        snippet = details['content'][:300].replace('\n', ' ') + "..."
                        response += f"Snippet: {snippet}\n"
                else:
                    response += f"\nSpecific {info_type} details are not indexed. Visit: {found_college.get('website', 'official website')}\n"
            
            # If General Inquiry (No specific info requested), Show Branches
            if not info_type:
                 branches = self.get_available_branches(code)
                 if branches:
                     response += "\n**Available Branches:**\n"
                     for i, b in enumerate(branches, 1):
                         response += f"{i}. {b}\n"
                 else:
                     response += "\n(No branch data found in predictions DB)\n"

                 response += "\n*Ask about placements, cutoffs, or hostels for more info.*"
                
            return response
            
        return None

    def get_response(self, message):
        if not self.model:
            return "I am currently under maintenance. Please try again later."
            
        if not message:
            return "Please say something."
            
        try:
            msg_lower = message.lower()
            
            # --- Hardcoded Keyword Checks ---
            
            # 1. General Cutoff/Predictor Queries (If no specific college context found later)
            general_cutoff_keywords = ["cutoff", "cutoffs", "closing rank", "rank needed", "prediction", "predict", "chances"]

            # [Rank Prediction Logic - Keeping existing code structure]
            rank_match = re.search(r'rank\s*[:is]?\s*(\d+)', msg_lower)
            if not rank_match and message.strip().isdigit():
                rank_match = re.search(r'(\d+)', message)
            
            if rank_match:
                # ... (Calling predict_colleges etc - same as before)
                rank = int(rank_match.group(1))
                # Identify requested course type
                course_type = None
                arch_keywords = ["architecture", "b.arch", "b arch", "at"]
                design_keywords = ["design", "b.des", "b des"]
                eng_keywords = ["engineering", "b.e", "b.tech", "b tech", "be", "technology"]

                if any(k in msg_lower for k in arch_keywords + design_keywords):
                    course_type = 'architecture'
                elif any(k in msg_lower for k in eng_keywords):
                    course_type = 'engineering'

                results = predict_colleges(rank, None, 'GM', course_type=course_type)
                
                # --- Categorize Results (Copied from original) ---
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

                # Format response
                response = f"Entered Rank: {rank}\n"
                if course_type:
                     response += f" (Filtered for {course_type.title()})\n\n"
                else:
                     response += "\n"

                def format_list(lst, limit=10):
                    txt = ""
                    # Sort primarily by cutoff
                    lst.sort(key=lambda x: x['cutoff']) # Ascending cutoff (Better rank first)
                    for i, r in enumerate(lst[:limit], 1):
                         txt += f"{i}. {r['college']} – {r['branch']}\n   (Cutoff: {r['cutoff']}, Round: {r['round']})\n"
                    return txt

                has_content = False
                if arch_list:
                    response += "**Architecture (B.Arch)**\n" + format_list(arch_list) + "\n"
                    has_content = True
                if design_list:
                    response += "**Design Courses**\n" + format_list(design_list) + "\n*Note: Design ranks may differ.*\n"
                    has_content = True
                if plan_list:
                    response += "**Planning Courses**\n" + format_list(plan_list) + "\n"
                    has_content = True
                if eng_list:
                    if course_type == 'engineering' or not course_type:
                        if has_content: response += "\n**Engineering Courses** (Top matches)\n"
                        else: response += "**Eligible Engineering Colleges**:\n"
                        response += format_list(eng_list, limit=10)
                        has_content = True

                if not has_content:
                    response += "Based on historical data, no colleges found for this rank."
                return response

            # 2. Check College Info
            college_response = self.get_college_info(message)
            if college_response:
                return college_response

            # --- General Category Counts (MOVED AFTER COLLEGE CHECK) ---
            is_rank_query = bool(re.search(r'\d+', msg_lower))
            if not is_rank_query:
                if "b.arch" in msg_lower or "architecture" in msg_lower:
                    return f"We have data on **{len(architecture_colleges)}** Architecture colleges. You can explore the list of all participating colleges in the **Colleges** section."

            # 3. Fallback Keyword Checks (If no specific college found)
            if any(k in msg_lower for k in general_cutoff_keywords):
                 return "Cutoffs vary by college and branch. Use our **College Predictor** tool to check detailed closing ranks and admission chances."

            # 4. Model Prediction
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
