import pandas as pd
import glob

def setup_database():
    # Find the Kaggle CSV you dropped in the folder
    csv_files = [f for f in glob.glob("*.csv") if f != "medicine_dataset.csv"]
    
    if not csv_files:
        print("Please drag the Kaggle CSV into this folder first!")
        return
        
    file_path = csv_files[0]
    print(f"Reading {file_path}...")
    df = pd.read_csv(file_path)
    
    # Look for the column that contains the drug names (like 'name', 'Drug_Name', etc.)
    col_name = next((col for col in df.columns if 'name' in col.lower()), df.columns[0])
    print(f"Extracting medicines from column: '{col_name}'")
    
    # Format it exactly how ocr_engine.py expects it
    new_df = pd.DataFrame()
    new_df['medicine_name'] = df[col_name].dropna().astype(str).str.upper()
    
    new_df.drop_duplicates().to_csv('medicine_dataset.csv', index=False)
    print(f"Done! Created medicine_dataset.csv with {len(new_df.drop_duplicates())} medicines.")

if __name__ == "__main__":
    setup_database()