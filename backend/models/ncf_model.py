import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import pandas as pd
import numpy as np

class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=8, mlp_layer_sizes=[16, 8, 4]):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers - using Sequential with indices to match the saved model
        mlp_layers = []
        input_size = embedding_dim * 2
        for size in mlp_layer_sizes:
            mlp_layers.extend([
                nn.Linear(input_size, size),
                nn.ReLU()
            ])
            input_size = size
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Output layer - using output_layer to match the saved model
        self.output_layer = nn.Linear(mlp_layer_sizes[-1] if mlp_layer_sizes else embedding_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        
        # Concatenate user and item embeddings
        x = torch.cat([user_embedded, item_embedded], dim=-1)
        
        # Pass through MLP
        x = self.mlp(x)
        
        # Output layer
        x = self.output_layer(x)
        x = self.sigmoid(x)
        
        # Scale to 1-5 rating range
        return x * 4 + 1

class RecommendationModel:
    def __init__(self, model_path, metadata_path, movies_path, dataset_name='movielens'):
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Set dataset-specific parameters
        self.dataset_name = dataset_name
        
        # Handle both old and new metadata formats
        if dataset_name == 'movielens':
            # Try new format first, then fall back to old format
            num_users = self.metadata.get('num_users_ml', self.metadata.get('num_users', 610))
            num_items = self.metadata.get('num_items_ml', self.metadata.get('num_items', 9724))
        else:  # amazon
            num_users = self.metadata.get('num_users_amazon', self.metadata.get('num_users', 192403))
            num_items = self.metadata.get('num_items_amazon', self.metadata.get('num_items', 63001))
        
        # Get model parameters with fallbacks
        embedding_dim = self.metadata.get('embedding_dim', 32)
        mlp_layer_sizes = self.metadata.get('mlp_layer_sizes', [64, 32, 16])
        
        print(f"Initializing model for {dataset_name}:")
        print(f"  - Users: {num_users}")
        print(f"  - Items: {num_items}")
        print(f"  - Embedding dim: {embedding_dim}")
        print(f"  - MLP layers: {mlp_layer_sizes}")
        
        # Initialize model
        self.model = NCF(
            num_users,
            num_items, 
            embedding_dim,
            mlp_layer_sizes
        )
        
        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
        # Store dataset info for easy access
        self.num_users = num_users
        self.num_items = num_items
        
        # Load item data based on dataset
        self.items_df = self.load_item_data(movies_path)
        
    def load_item_data(self, file_path):
        """Load item data based on dataset type"""
        try:
            if self.dataset_name == 'movielens':
                # Movie details
                df = pd.read_csv(file_path)
                return df
            else:
                # Amazon product details
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    expected_items = self.num_items
                    
                    # If we have fewer items than expected, create placeholders
                    if len(df) < expected_items:
                        print(f"Warning: Amazon product file has {len(df)} items but expected {expected_items}. Adding placeholders.")
                        # Create placeholders for missing items
                        missing_indices = set(range(expected_items)) - set(df['product_id'])
                        if missing_indices:
                            missing_df = pd.DataFrame({
                                'product_id': list(missing_indices),
                                'title': [f'Product {i}' for i in missing_indices],
                                'category': ['Electronics'] * len(missing_indices)
                            })
                            df = pd.concat([df, missing_df], ignore_index=True)
                    
                    # Ensure we have the correct number of items
                    df = df.sort_values('product_id').reset_index(drop=True)
                    if len(df) > expected_items:
                        df = df.iloc[:expected_items]
                        
                    return df
                else:
                    # Create a placeholder if file doesn't exist
                    print(f"Warning: Amazon product file {file_path} not found. Using placeholder data.")
                    return pd.DataFrame({
                        'product_id': range(self.num_items),
                        'title': [f'Product {i}' for i in range(self.num_items)],
                        'category': ['Electronics'] * self.num_items
                    })
        except Exception as e:
            print(f"Error loading item data: {e}")
            # Return a minimal valid DataFrame with the correct number of items
            return pd.DataFrame({
                'product_id': range(self.num_items),
                'title': [f'Product {i}' for i in range(self.num_items)],
                'category': ['Electronics'] * self.num_items
            })
    
    def get_item_details(self, item_id):
        """Get item details based on dataset type"""
        try:
            if self.dataset_name == 'movielens':
                # Movie details
                movie_row = self.items_df.iloc[item_id]
                return {
                    'title': movie_row['title'],
                    'genres': movie_row.get('genres', 'Unknown'),
                    'type': 'movie'
                }
            else:
                # Amazon product details
                if len(self.items_df) > item_id:
                    product_row = self.items_df.iloc[item_id]
                    return {
                        'title': product_row.get('title', f'Product {item_id}'),
                        'category': product_row.get('category', 'Electronics'),
                        'type': 'product'
                    }
                else:
                    return {
                        'title': f'Product {item_id}',
                        'category': 'Electronics', 
                        'type': 'product'
                    }
        except:
            # Return placeholder if details not found
            if self.dataset_name == 'movielens':
                return {
                    'title': f'Movie {item_id}',
                    'genres': 'Unknown',
                    'type': 'movie'
                }
            else:
                return {
                    'title': f'Product {item_id}',
                    'category': 'Electronics',
                    'type': 'product'
                }
    
    def predict_rating(self, user_id, item_id):
        """Predict rating for a user-item pair"""
        with torch.no_grad():
            user_tensor = torch.tensor([user_id], dtype=torch.long)
            item_tensor = torch.tensor([item_id], dtype=torch.long)
            prediction = self.model(user_tensor, item_tensor)
            return float(prediction.item())
    
    def get_top_recommendations(self, user_id, top_k=10):
        """Get top recommendations for a user"""
        try:
            print(f"\n=== Generating top {top_k} recommendations for user {user_id} on {self.dataset_name} ===")
            
            # Use the stored num_items instead of looking in metadata
            num_items = self.num_items
            num_users = self.num_users
            
            print(f"Total items in dataset: {num_items}")
            print(f"Total users in dataset: {num_users}")
            
            # Validate user_id
            if user_id < 0 or user_id >= num_users:
                raise ValueError(f"User ID {user_id} is out of range (0-{num_users-1})")
            
            # Prepare items and predictions
            all_items = list(range(num_items))
            predictions = []
            
            print(f"Starting prediction for {len(all_items)} items...")
                
            with torch.no_grad():
                user_tensor = torch.tensor([user_id] * len(all_items), dtype=torch.long)
                item_tensor = torch.tensor(all_items, dtype=torch.long)
                
                # Batch predictions to avoid memory issues
                batch_size = 1000
                for i in range(0, len(all_items), batch_size):
                    batch_start = i
                    batch_end = min(i + batch_size, len(all_items))
                    print(f"Processing batch: items {batch_start} to {batch_end-1}")

                    batch_users = user_tensor[batch_start:batch_end]
                    batch_items = item_tensor[batch_start:batch_end]

                    batch_preds = self.model(batch_users, batch_items)
                    
                    # Convert tensor to list of floats
                    if hasattr(batch_preds, 'detach'):
                        batch_preds = batch_preds.detach()
                    if hasattr(batch_preds, 'numpy'):
                        batch_preds = batch_preds.numpy()
                    
                    # Handle different output types
                    if isinstance(batch_preds, np.ndarray):
                        # If it's a 2D array with shape [batch_size, 1], squeeze it
                        if len(batch_preds.shape) > 1 and batch_preds.shape[1] == 1:
                            batch_preds = batch_preds.squeeze(axis=1)
                        batch_scores = [float(x) for x in batch_preds]
                    elif isinstance(batch_preds, list):
                        batch_scores = [float(x) for x in batch_preds]
                    else:
                        # Try to iterate through tensor
                        try:
                            batch_scores = [float(x) for x in batch_preds]
                        except:
                            # Last resort: try direct conversion
                            batch_scores = [float(batch_preds)]
                    
                    predictions.extend(batch_scores)

                print(f"Completed predictions for all {len(predictions)} items")

            if not predictions:
                raise ValueError("No predictions were generated")
                
            print(f"Converting predictions to numpy array...")
            predictions = np.array(predictions)
            
            if len(predictions) != num_items:
                print(f"Warning: Expected {num_items} predictions, got {len(predictions)}")
            
            print("Finding top recommendations...")
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            top_recommendations = []
            
            print("Processing top recommendations...")
            for rank, idx in enumerate(top_indices, 1):
                try:
                    idx_int = int(idx)
                    item_details = self.get_item_details(idx_int)
                    
                    recommendation = {
                        'rank': rank,
                        'item_id': idx_int,
                        'title': str(item_details.get('title', f'Item {idx_int}')),
                        'category': str(item_details.get('genres') or item_details.get('category', 'Unknown')),
                        'type': str(item_details.get('type', 'product')),
                        'predicted_rating': float(round(predictions[idx], 4))
                    }
                    top_recommendations.append(recommendation)
                    
                except Exception as item_error:
                    print(f"Error processing item {idx}: {str(item_error)}")
                    continue
            
            print(f"Successfully generated {len(top_recommendations)} recommendations")
            return top_recommendations
            
        except Exception as e:
            error_msg = f"Error in get_top_recommendations: {str(e)}"
            print(f"\n=== ERROR ===\n{error_msg}")
            import traceback
            print("\nStack trace:")
            print(traceback.format_exc())
            print("============\n")
            raise

    def get_dataset_info(self):
        """Get information about the dataset for API responses"""
        return {
            "dataset": self.dataset_name,
            "total_users": self.num_users,
            "total_items": self.num_items,
            "model_type": "Federated NCF"
        }