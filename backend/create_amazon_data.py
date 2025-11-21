# backend/create_amazon_data.py
import pandas as pd
import json

def create_amazon_product_data():
    # Load your processed Amazon data to get product IDs
    processed_amazon = pd.read_csv('path/to/your/processed_ratings_amazon.csv')  # Adjust path
    
    # Get unique products
    unique_products = processed_amazon[['item_id', 'item']].drop_duplicates()
    
    # Create product data (you can enhance this with real product names)
    products_data = []
    for _, row in unique_products.iterrows():
        products_data.append({
            'product_id': row['item_id'],
            'asin': row['item'],  # Original Amazon ASIN
            'title': f'Electronics Product {row["item_id"]}',
            'category': 'Electronics'
        })
    
    # Create DataFrame and save
    products_df = pd.DataFrame(products_data)
    products_df.to_csv('backend/data/amazon_products.csv', index=False)
    print(f"Created Amazon product data with {len(products_df)} products")

if __name__ == '__main__':
    create_amazon_product_data()