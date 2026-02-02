# Explicit data for college links and courses
# This file is regenerated based on the official websites in colleges_data.py

college_links_data = {}

def get_college_explicit_data(college_code):
    """
    Returns manually curated data for a college if available.
    """
    if college_code in college_links_data:
        return {'links': college_links_data[college_code]}
    return {}
