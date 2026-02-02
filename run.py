print("=== run.py started ===")
import os
from backend.app import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's default port if not set
    app.run(host="0.0.0.0", port=port, debug=True)
