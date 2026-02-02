# COMEDK College Predictor & Chatbot Assistant 🎓

A comprehensive web application to help students navigate the COMEDK UGET counseling process for 2026. It features a Machine Learning-based college predictor, an AI chatbot for instant results, and detailed information about colleges, branches, and courses.

## 🚀 Key Features

*   **🏆 Advanced College Predictor**: Uses historical data (2021-2025) and trend analysis to predict eligible engineering colleges and branches based on your 2026 rank.
*   **🤖 AI Chatbot Assistant**: An NLP-powered assistant capable of answering queries about syllabus, cutoffs (General/HK), exam patterns, and college info.
*   **📚 Course Explorer**: Detailed insights into available Engineering and Architecture courses.
*   **🏛️ College Directory**: Searchable list of top Engineering colleges in Karnataka with location and code details.
*   **📈 Historical Data Analysis**: Built on verified cutoff data from the last 5 years (2021-2025).

## 🛠️ Tech Stack

*   **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5 (Responsive Design)
*   **Backend**: Python, Flask
*   **Machine Learning / AI**: 
    *   `scikit-learn` for rank prediction (Linear Regression & Trend Analysis)
    *   NLP (Bag of Words) for the chatbot
    *   XGBoost for advanced predictions
*   **Data Processing**: Pandas, NumPy.
*   **Database**: SQLite (auto-generated from CSV), JSON/CSV file storage for rapid access.

## 📂 Project Structure

```
COMEDK_DTL/
├── backend/
│   ├── app.py                # Main Flask Application Factory
│   ├── routes.py             # URL Routing & Controller Logic
│   ├── store_predictions.py  # Prediction Engine (2026 forecasts)
│   ├── store_predictions_barch.py # B.Arch/Design Prediction Engine
│   ├── chatbot_ai.py         # NLP Chatbot Implementation
│   ├── college_agent.py      # Web scraping for official college info
│   ├── colleges_data.py      # College list & details
│   ├── branches_data.py      # Branches/courses list
│   ├── database.py           # DB connection (SQLite)
│   ├── college_details_data.py # Manual college info (optional)
│   ├── intents.json          # Chatbot Training Data
│   └── ...                   # Other backend utilities
├── data/
│   └── processed/            # Cleaned datasets (CSV)
├── frontend/
│   ├── static/               # CSS, Images, JS
│   └── templates/            # HTML Templates (Jinja2)
├── tools/
│   ├── enrich_predictions_csv.py # Data enrichment scripts
│   └── scrape_collegedunia.py    # Web scraping tools
├── COMEDK_MASTER_2021_2025.csv  # Primary Historical Dataset
├── requirements.txt          # Python dependencies
├── run.py                   # Application Entry Point
├── setup_db.py              # DB setup from CSV
├── Procfile                 # For PaaS deployment (Heroku/Railway)
├── nixpacks.toml            # For Nixpacks deployment
├── runtime.txt              # Python version for deployment
└── README.md                # Project documentation
```


## ☁️ Deployment

### Deploy on Railway (Recommended)

1.  Sign up at [railway.app](https://railway.app/).
2.  Click "New Project" → "Deploy from GitHub repo".
3.  Select your repository (e.g., `ComedK-Predictor`).
4.  Railway will detect the `Procfile` and deploy your app.
    *   *Note: The database (`comedk.db`) is rebuilt every time the server starts using `python setup_db.py`.*

### Deploy on Heroku

1.  Install the Heroku CLI and login (`heroku login`).
2.  Create a new app: `heroku create`.
3.  Deploy: `git push heroku main`.


## ⚙️ Installation & Local Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/kollipatisravanthi-a11y/ComedK-Predictor.git
    cd COMEDK_DTL
    ```

2.  **(Recommended) Create a Virtual Environment**:
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Initialize Database**:
    This step creates the `comedk.db` file from the CSV data.
    ```bash
    python setup_db.py
    ```

5.  **Run the Application**:
    ```bash
    python run.py
    ```
    The application will start at [http://127.0.0.1:5000/](http://127.0.0.1:5000/)


## 👥 Meet the Team

*   **Kollipati Lakshmi Sravanthi** - Project Lead
*   **K Manoj Kumar** - Backend Developer
*   **Nanda Kumar HR** - Frontend Developer
*   **NR Mahesh Raju** - Database Admin

---
---
*© 2025-2026 COMEDK Predictor. All Rights Reserved.*
