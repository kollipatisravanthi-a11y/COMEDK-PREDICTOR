print("=== setup_db.py started ===")
import pandas as pd
import os
from backend.database import engine
from backend.store_predictions import generate_predictions

def setup_database():
    print("🚀 Starting Database Setup...")
    
    # helper for finding files
    def find_data_file():
        # Priority: confirmed working file in backend first
        check_paths = [
            'backend/final_cutoff_data.csv',
            'data/processed/COMEDK_MASTER_2021_2025.csv',
            'data/processed/comedk_master_2021_2025.csv',
            'COMEDK_MASTER_2021_2025.csv'
        ]
        for p in check_paths:
            if os.path.exists(p):
                return p
        return None

    master_csv = find_data_file()
            
    if not master_csv:
        print(f"⚠️ Warning: Master CSV not found. Checking if DB already exists.")
        if os.path.exists('comedk.db'):
            print("✅ comedk.db already exists. Skipping reconstruction.")
            return
        else:
            print(f"❌ Error: Master data file missing! Current Dir: {os.getcwd()}")
            print(f"Contents of backend/: {os.listdir('backend') if os.path.exists('backend') else 'N/A'}")
            return

    print(f"📂 Reading {master_csv}...")
    try:
        df_master = pd.read_csv(master_csv)
        
        # Normalize columns (handles various CSV formats)
        rename_map = {
            'branch_code': 'branch',
            'Closing_Rank': 'closing_rank',
            'College Name': 'college_name',
            'College Code': 'college_code',
            'Seat Category': 'category'
        }
        df_master.rename(columns=rename_map, inplace=True)
        
        # Ensure college_name exists
        if 'college_name' not in df_master.columns and 'college_code' in df_master.columns:
            print("ℹ️ Mapping college codes to names...")
            from backend.colleges_data import colleges_list, architecture_colleges
            all_cols = colleges_list + architecture_colleges
            code_to_name = {str(c['code']): c['name'] for c in all_cols}
            df_master['college_name'] = df_master['college_code'].astype(str).map(code_to_name)
            
        print(f"   Loaded {len(df_master)} rows.")

        # Load B.Arch Data if available
        barch_csv = None
        for p in ['data/processed/COMEDK_MASTER_BARCH.csv', 'data/processed/comedk_master_barch.csv']:
            if os.path.exists(p):
                barch_csv = p
                break

        if barch_csv:
            try:
                df_barch = pd.read_csv(barch_csv)
                if not df_barch.empty:
                    if 'round_no' in df_barch.columns: df_barch.rename(columns={'round_no': 'round'}, inplace=True)
                    if 'round' in df_barch.columns: df_barch['round'] = 'R' + df_barch['round'].astype(str).str.replace('R', '', regex=False)
                    for col in df_master.columns:
                        if col not in df_barch.columns: df_barch[col] = None
                    df_barch = df_barch[df_master.columns]
                    df_master = pd.concat([df_master, df_barch], ignore_index=True)
                    print(f"   Added B.Arch data.")
            except Exception as ex:
                print(f"   ⚠️ Warning reading B.Arch file: {ex}")

        # Write to SQLite
        print("💾 Saving to SQLite database (comedk.db)...")
        df_master.to_sql('COMEDK_MASTER_2021_2025', engine, if_exists='replace', index=False)
        print("✅ Master data secured.")
        
    except Exception as e:
        print(f"❌ Critical Error loading master data: {e}")
        return


    # Skipping ML-based prediction generation; using precomputed predictions only

    print("\n🎉 Setup Complete!")

if __name__ == "__main__":
    setup_database()
