import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
from googlesearch import search
import json
import os
import sys

# Setup path to import sibling modules if running as script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Predictor import removed as it is handled by the database layer now.
# try:
#     from backend.prediction_2025 import predictor_2025
# except ImportError:
#     try:
#         from prediction_2025 import predictor_2025
#     except ImportError:
#         predictor_2025 = None
#         print("Warning: Could not import predictor_2025. Make sure prediction_2025.py is in the same directory.")

class CollegeAgent:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.ignored_domains = [
            'shiksha.com', 'careers360.com', 'wikipedia.org', 'collegedunia.com', 
            'getmyuni.com', 'collegedekho.com', 'justdial.com', 'facebook.com', 
            'linkedin.com', 'instagram.com', 'youtube.com', 'twitter.com'
        ]
        self.prediction_file = os.path.join(os.path.dirname(__file__), 'prediction_2025.json')

    def is_official_domain(self, url):
        domain = urlparse(url).netloc.lower()
        for ignored in self.ignored_domains:
            if ignored in domain:
                return False
        return True

    def find_official_website(self, college_name):
        # Clean college name: remove location details after hyphen
        clean_name = college_name.split('-')[0].strip()
        query = f"{clean_name} official website"
        print(f"Searching query: {query}")
        try:
            # Increase results to find a valid one
            for url in search(query, num_results=10):
                print(f"Found URL: {url}")
                if self.is_official_domain(url):
                    return url
        except Exception as e:
            print(f"Error searching for website: {e}")
        return None

    def search_deep_link(self, base_url, keywords):
        domain = urlparse(base_url).netloc
        query = f"site:{domain} {' '.join(keywords)}"
        try:
            for url in search(query, num_results=3):
                return url
        except Exception as e:
            print(f"Error searching deep link: {e}")
        return None

    def fetch_page(self, url):
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {e}")
        return None

    def find_link(self, soup, keywords, base_url):
        if not soup:
            return None
        
        # Try exact match first
        for a in soup.find_all('a', href=True):
            text = a.get_text().strip().lower()
            href = a['href']
            for keyword in keywords:
                if keyword == text:
                    return urljoin(base_url, href)
        
        # Then partial match
        for a in soup.find_all('a', href=True):
            text = a.get_text().strip().lower()
            href = a['href']
            for keyword in keywords:
                if keyword in text:
                    return urljoin(base_url, href)
        return None

    def extract_placements(self, soup, url):
        data = {
            "highest_package": "Information not available on official website",
            "average_package": "Information not available on official website",
            "top_recruiters": "Information not available on official website",
            "year": "Information not available on official website",
            "url": url
        }
        if not soup:
            return data

        text = soup.get_text()
        
        # Improved regex for packages
        # Look for patterns like "50 LPA", "50 Lakhs", "50.5 LPA"
        lpa_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:LPA|Lakhs?|Cr|Crores?)', text, re.IGNORECASE)
        
        if lpa_matches:
            # Convert to float to find max
            try:
                values = [float(x) for x in lpa_matches]
                if values:
                    max_val = max(values)
                    data["highest_package"] = f"{max_val} LPA"
                    
                    # Heuristic: Average is usually lower than max but not too low
                    # Filter out very small numbers (likely years or other stats)
                    valid_values = [v for v in values if 2 < v < max_val]
                    if valid_values:
                        # Take the average of remaining valid values or just a likely candidate
                        # This is a rough heuristic
                        avg_val = sum(valid_values) / len(valid_values)
                        data["average_package"] = f"{avg_val:.2f} LPA"
            except ValueError:
                pass

        # Fallback to specific keywords if general search failed
        if data["highest_package"] == "Information not available on official website":
            highest_match = re.search(r'Highest\s*(?:Package|Salary)?\s*[:\-]?\s*([\d\.]+\s*(?:LPA|Lakhs?|Cr|Crores?))', text, re.IGNORECASE)
            if highest_match:
                data["highest_package"] = highest_match.group(1)

        if data["average_package"] == "Information not available on official website":
            avg_match = re.search(r'Average\s*(?:Package|Salary)?\s*[:\-]?\s*([\d\.]+\s*(?:LPA|Lakhs?))', text, re.IGNORECASE)
            if avg_match:
                data["average_package"] = avg_match.group(1)

        # Year
        year_match = re.search(r'20\d{2}', text)
        if year_match:
            data["year"] = year_match.group(0)

        return data

    def extract_hostel(self, soup, url):
        data = {
            "availability": "Yes", # Assume yes if page exists
            "separate": "Information not available on official website",
            "facilities": "Information not available on official website",
            "url": url
        }
        if not soup:
            data["availability"] = "Information not available on official website"
            return data

        text = soup.get_text().lower()
        if "boys" in text and "girls" in text:
            data["separate"] = "Yes"
        else:
             data["separate"] = "No"
        
        facilities = []
        if "wifi" in text or "wi-fi" in text:
            facilities.append("Wi-Fi")
        if "mess" in text:
            facilities.append("Mess")
        if "gym" in text:
            facilities.append("Gym")
        if "laundry" in text:
            facilities.append("Laundry")
        if "security" in text:
            facilities.append("Security")
        
        if facilities:
            data["facilities"] = ", ".join(facilities)
            
        return data

    def get_college_data(self, college_url, college_name=None):
        if not college_url and college_name:
            print(f"Searching for official website for {college_name}...")
            college_url = self.find_official_website(college_name)
        
        if not college_url:
            return {"error": "Could not find official website"}

        print(f"Fetching data from: {college_url}")
        main_soup = self.fetch_page(college_url)
        data = {}

        # 1. Placements
        placement_url = self.find_link(main_soup, ['placement', 'career', 'recruit'], college_url)
        if not placement_url:
             placement_url = self.search_deep_link(college_url, ['placements'])
        
        if placement_url:
            placement_soup = self.fetch_page(placement_url)
            data['placements'] = self.extract_placements(placement_soup, placement_url)
        else:
            data['placements'] = {
                "highest_package": "Information not available on official website",
                "average_package": "Information not available on official website",
                "top_recruiters": "Information not available on official website",
                "year": "Information not available on official website",
                "url": "Information not available on official website"
            }

        # 2. Hostel
        hostel_url = self.find_link(main_soup, ['hostel', 'accommodation', 'residence'], college_url)
        if not hostel_url:
            hostel_url = self.search_deep_link(college_url, ['hostel', 'facilities'])

        if hostel_url:
            hostel_soup = self.fetch_page(hostel_url)
            data['hostel'] = self.extract_hostel(hostel_soup, hostel_url)
        else:
            data['hostel'] = {
                "availability": "Information not available on official website",
                "separate": "Information not available on official website",
                "facilities": "Information not available on official website",
                "url": "Information not available on official website"
            }

        # 3. Infrastructure
        infra_url = self.find_link(main_soup, ['infrastructure', 'campus', 'facility'], college_url)
        if not infra_url:
            infra_url = self.search_deep_link(college_url, ['infrastructure', 'campus'])

        data['infrastructure'] = {
            "campus_area": "Information not available on official website",
            "facilities": "Information not available on official website",
            "url": infra_url if infra_url else "Information not available on official website"
        }

        # 4. Academics
        acad_url = self.find_link(main_soup, ['academic', 'department', 'program'], college_url)
        if not acad_url:
            acad_url = self.search_deep_link(college_url, ['academics', 'departments'])

        data['academics'] = {
            "courses": "Information not available on official website",
            "accreditation": "Information not available on official website",
            "faculty": "Information not available on official website",
            "url": acad_url if acad_url else "Information not available on official website"
        }

        # 5. Admissions
        admin_url = self.find_link(main_soup, ['admission'], college_url)
        if not admin_url:
            admin_url = self.search_deep_link(college_url, ['admissions'])

        data['admissions'] = {
            "comedk_accepted": "Yes",
            "process": "Information not available on official website",
            "url": admin_url if admin_url else "Information not available on official website"
        }

        # 6. Contact
        contact_url = self.find_link(main_soup, ['contact', 'reach us'], college_url)
        if not contact_url:
            contact_url = self.search_deep_link(college_url, ['contact'])

        data['contact'] = {
            "address": "Information not available on official website",
            "phone": "Information not available on official website",
            "email": "Information not available on official website",
            "url": contact_url if contact_url else "Information not available on official website"
        }

        return data

    def load_prediction_results(self):
        """Load prediction results from prediction_2025 file"""
        if not os.path.exists(self.prediction_file):
            return None
            
        try:
            with open(self.prediction_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding prediction file")
            return None

    def enrich_predictions_with_college_data(self, predictions):
        """Enrich prediction results with college details from web scraping"""
        if not predictions:
            return None
        
        enriched_results = []
        
        # Handle both list and dict formats
        colleges = []
        if isinstance(predictions, list):
            colleges = predictions
        elif isinstance(predictions, dict):
            colleges = predictions.get('colleges', [])
        
        print(f"Found {len(colleges)} colleges to process")
        
        # Limit to top 5 for demo to prevent long execution times, remove slice [:5] for full run
        for college_info in colleges[:5]: 
            # Handle different key names for college name
            college_name = college_info.get('college_name') or college_info.get('name') or college_info.get('college')
            
            if not college_name:
                continue

            print(f"\nProcessing: {college_name}")
            try:
                # First check if we already have data (simple caching logic could go here)
                college_data = self.get_college_data(None, college_name)
                
                # Merge prediction data with scraped data
                enriched_college = {
                    **college_info,
                    'details': college_data
                }
                enriched_results.append(enriched_college)
            except Exception as e:
                print(f"Failed to fetch data for {college_name}: {str(e)}")
                enriched_results.append(college_info)
        
        return enriched_results

    def process_and_save_enriched_predictions(self, rank=None, category='GM', output_file='enriched_predictions.json'):
        """Generate predictions, enrich with college data, and save results"""
        print(f"Starting prediction and enrichment process...")
        
        predictions = []
        
        # 1. Use live predictor if available (Priority)
        if predictor_2025:
            # If no rank provided, use a default rank that yields results for testing 
            search_rank = rank if rank else 10000 
            print(f"Generating live predictions for Rank: {search_rank}, Category: {category}...")
            predictions = predictor_2025.predict(search_rank, category)
            print(f"Predictor returned {len(predictions)} results.")
        
        # 2. If valid predictions weren't generated, try loading file
        if not predictions:
            print(f"No live predictions generated. Checking for file: {self.prediction_file}")
            predictions = self.load_prediction_results()

        if not predictions:
            print("Error: No predictions available to enrich.")
            return {"error": "Failed to generate or load predictions"}
        
        enriched_data = self.enrich_predictions_with_college_data(predictions)
        
        # Save enriched data
        output_path = os.path.join(os.path.dirname(__file__), output_file)
        with open(output_path, 'w') as f:
            json.dump(enriched_data, f, indent=2)
        
        print(f"\nEnriched predictions saved to: {output_path}")
        return enriched_data

college_agent = CollegeAgent()

if __name__ == "__main__":
    # Test Run: Connect prediction with enrichment
    # Using Rank 15000 GM as a test case
    college_agent.process_and_save_enriched_predictions(rank=15000, category='GM')
