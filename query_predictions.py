import pandas as pd


CSV_PATH = 'data/processed/predictions_2026.csv'

def query_predictions(rank, course_type, category=None):
    df = pd.read_csv(CSV_PATH)
    # Filter for course type
    if course_type.lower() == 'btech':
        df = df[~df['branch'].str.contains('arch|design', case=False, na=False)]
    elif course_type.lower() == 'barch':
        df = df[df['branch'].str.contains('arch|design', case=False, na=False)]
    else:
        print('Invalid course type. Use "btech" or "barch".')
        return

    # Filter by rank and category if provided
    df = df[df['predicted_closing_rank'] >= rank]
    if category and 'category' in df.columns:
        df = df[df['category'].str.lower() == category.lower()]

    if df.empty:
        print('No matching colleges/courses found.')
    else:
        print(df[['branch', 'category', 'predicted_closing_rank']])

if __name__ == '__main__':
    rank = int(input('Enter your rank: '))
    course_type = input('Enter course type (btech/barch): ')
    category = input('Enter category (optional, press Enter to skip): ')
    category = category if category.strip() else None
    query_predictions(rank, course_type, category)
