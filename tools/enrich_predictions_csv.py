import pandas as pd
import re
from backend.colleges_data import colleges_list, architecture_colleges

# Load the original predictions CSV
input_csv = 'predictions_2026.csv'
df = pd.read_csv(input_csv)

# Combine all colleges into one list
all_colleges = colleges_list + architecture_colleges

# Build a mapping from college name (upper, stripped) to code and official name
name_map = {c['name'].strip().upper(): (c['code'], c['name']) for c in all_colleges}

# Helper: Try to extract college name from branch or other info (if possible)
def guess_college_name(row):
    # This function can be improved if you have a pattern
    # For now, returns None
    return None

# Add columns for code and name
college_codes = []
college_names = []

for idx, row in df.iterrows():
    # Try to match college name from a known field (if available)
    # If your CSV has a college name field, use it. Otherwise, try to guess.
    college_name = guess_college_name(row)
    code = None
    name = None
    if college_name:
        key = college_name.strip().upper()
        if key in name_map:
            code, name = name_map[key]
    # If not found, leave as None
    college_codes.append(code if code else 'UNKNOWN')
    college_names.append(name if name else 'Unknown College')

# Add to DataFrame
df['college_code'] = college_codes
df['college_name'] = college_names

# Save enriched CSV
output_csv = 'predictions_2026_enriched.csv'
df.to_csv(output_csv, index=False)
print(f'Enriched CSV saved as {output_csv}')
