import pandas as pd
import numpy as np
from sqlalchemy import text
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.database import engine

ROUND_ORDER = ["R1", "R2", "R3", "R4"]

def generate_predictions_btech():
    # TODO: Move code from store_predictions.py's generate_predictions here
    pass

def generate_predictions_barch():
    # TODO: Move code from store_predictions_barch.py's generate_predictions_barch here
    pass
