from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from models.ncf_model import RecommendationModel
import pandas as pd

app = Flask(__name__)
CORS(app)

# Initialize both models
print("Loading recommendation models...")
model_ml = RecommendationModel(
    model_path='saved_models/global_model_ml_fedavg.pth',
    metadata_path='saved_models/model_metadata.json', 
    movies_path='data/movies_ml.csv',
    dataset_name='movielens'
)

model_amazon = RecommendationModel(
    model_path='saved_models/global_model_amazon_fedavg.pth',
    metadata_path='saved_models/model_metadata.json',
    movies_path='data/amazon_products.csv',  # You'll need to create this
    dataset_name='amazon'
)
print("Both models loaded successfully!")

@app.route('/')
def home():
    return jsonify({
        "message": "Federated Learning Recommendation API", 
        "status": "active",
        "datasets": ["movielens", "amazon"]
    })

@app.route('/api/<dataset>/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(dataset, user_id):
    """Get top recommendations for a user from specific dataset"""
    try:
        print(f"\n=== New Recommendation Request ===")
        print(f"Dataset: {dataset}, User ID: {user_id}")
        
        # Select the right model
        if dataset == 'movielens':
            print("Using MovieLens model")
            model = model_ml
        elif dataset == 'amazon':
            print("Using Amazon model")
            model = model_amazon
        else:
            error_msg = f"Dataset '{dataset}' not found. Use 'movielens' or 'amazon'"
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        top_k = request.args.get('top_k', 10, type=int)
        print(f"Top K: {top_k}")
        
        # Check user ID range
        max_users = model.metadata[f'num_users_{dataset}'] if dataset == 'amazon' else model.metadata['num_users_ml']
        print(f"Max users for {dataset}: {max_users}")
        
        if user_id >= max_users:
            error_msg = f"User ID {user_id} out of range. Max: {max_users-1}"
            print(f"Error: {error_msg}")
            return jsonify({"error": error_msg}), 400
        
        print("Generating recommendations...")
        recommendations = model.get_top_recommendations(user_id, top_k)
        print(f"Generated {len(recommendations)} recommendations")
        
        return jsonify({
            "dataset": dataset,
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations)
        })
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print("\n=== ERROR DETAILS ===")
        print(error_trace)
        print("===================\n")
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "trace": error_trace if app.debug else None
        }), 500

@app.route('/api/<dataset>/predict', methods=['POST'])
def predict_rating(dataset):
    """Predict rating for a specific user-item pair"""
    try:
        if dataset not in ['movielens', 'amazon']:
            return jsonify({"error": "Dataset not found. Use 'movielens' or 'amazon'"}), 400
        
        # Select model
        model = model_ml if dataset == 'movielens' else model_amazon
        
        data = request.get_json()
        user_id = data.get('user_id')
        item_id = data.get('item_id')
        
        if user_id is None or item_id is None:
            return jsonify({"error": "user_id and item_id are required"}), 400
        
        predicted_rating = model.predict_rating(user_id, item_id)
        
        return jsonify({
            "dataset": dataset,
            "user_id": user_id,
            "item_id": item_id,
            "predicted_rating": round(predicted_rating, 2),
            "model_used": "federated_learning"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/<dataset>/info', methods=['GET'])
def get_dataset_info(dataset):
    """Get information about a specific dataset"""
    try:
        if dataset == 'movielens':
            model = model_ml
            users_key = 'num_users_ml'
            items_key = 'num_items_ml'
        elif dataset == 'amazon':
            model = model_amazon
            users_key = 'num_users_amazon'
            items_key = 'num_items_amazon'
        else:
            return jsonify({"error": "Dataset not found"}), 400
        
        return jsonify({
            "dataset": dataset,
            "total_users": model.metadata[users_key],
            "total_items": model.metadata[items_key],
            "model_type": "Federated NCF"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)