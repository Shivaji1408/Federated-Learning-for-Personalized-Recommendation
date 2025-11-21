# backend/create_mappings.py
import pandas as pd
import json

def create_mappings():
    # Load your processed data
    processed_ratings = pd.read_csv('path/to/your/processed_ratings_ml.csv')  # Adjust path
    
    # Create user mapping
    user_mapping = processed_ratings[['user_id', 'user']].drop_duplicates()
    user_map_dict = {int(row['user_id']): str(row['user']) for _, row in user_mapping.iterrows()}
    
    # Create item mapping  
    item_mapping = processed_ratings[['item_id', 'item']].drop_duplicates()
    item_map_dict = {int(row['item_id']): str(row['item']) for _, row in item_mapping.iterrows()}
    
    # Save mappings
    with open('backend/data/user_mapping.json', 'w') as f:
        json.dump(user_map_dict, f, indent=2)
    
    with open('backend/data/item_mapping.json', 'w') as f:
        json.dump(item_map_dict, f, indent=2)
    
    print("Mapping files created successfully!")

if __name__ == '__main__':
    create_mappings()   