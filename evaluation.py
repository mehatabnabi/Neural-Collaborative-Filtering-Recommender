

import numpy as np
import pandas as pd
from keras.models import load_model

# Load the saved model
model = load_model('ncf_model.h5')

# Load the test data
test_data = pd.read_csv('test_data.csv')
num_items = test_data['item_id'].nunique()

# Function to calculate Hit Ratio@10
def hit_ratio_at_k(model, test_data, top_k=10):
    hits = 0
    total_users = test_data['user_id'].nunique()
    
    for user in test_data['user_id'].unique():
        user_items = test_data[test_data['user_id'] == user]['item_id'].values
        item_ids = np.arange(num_items)
        predictions = model.predict([np.array([user] * num_items), item_ids])
        
        # Get the top-k items
        top_k_items = np.argsort(predictions[:, 0])[-top_k:]
        
        # Check if any relevant item is in the top-k predictions
        if any(item in top_k_items for item in user_items):
            hits += 1
    
    hr_at_k = hits / total_users
    return hr_at_k

# Calculate Hit Ratio@10
hr_at_10 = hit_ratio_at_k(model, test_data, top_k=10)
print(f"Hit Ratio@10: {hr_at_10:.2f}")
