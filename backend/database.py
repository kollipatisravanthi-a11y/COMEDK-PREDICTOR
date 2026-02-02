import os
from sqlalchemy import create_engine

# Get the base directory (root of the project)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'comedk.db')

# Access the DB engine
# We use check_same_thread=False because Flask runs in multiple threads
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={'check_same_thread': False})

def get_engine():
    return engine
