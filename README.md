# Neural Collaborative Filtering Recommender System

## Key Features
- **Model Type:** Neural Collaborative Filtering with implicit feedback.
- **Evaluation Metric:** Hit Ratio@10 (HR@10).

## Data Preprocessing
- **Handle Missing Values:** Any rows with missing values are removed to ensure data consistency.
- **Remove Duplicates:** Duplicates in user-item interactions are removed to avoid bias.
- **Implicit Feedback Conversion:** Ratings are converted into implicit feedback, with positive interactions labeled as `1` and others as `0`.
- **Re-indexing:** User and item IDs are re-indexed to start from 0 for embedding purposes.
- **Train-Test Split:** Data is split into 80% training and 20% testing using PySpark.
- **Data Format:** Parquet files are used for efficient storage, with conversion to Pandas for further processing if req.

## Project Structure
- **data_preprocessing.py:** Handles data cleaning, feature engineering, and train-test splitting using PySpark.
- **ncf_model.py:** Builds and trains the Neural Collaborative Filtering model using TensorFlow/Keras.
- **evaluation.py:** Evaluates the model's performance using Hit Ratio@10.

## How to Run
1. Install the required dependencies: `pip install -r requirements.txt`
2. Place your dataset in **Parquet** format in the project directory and update `data_preprocessing.py` to reference it.
3. Run `data_preprocessing.py` to preprocess the dataset.
4. Run `ncf_model.py` to train and save the NCF model.
5. Run `evaluation.py` to calculate the Hit Ratio@10 metric.

## Dataset
- The dataset must contain columns for `user_id`, `item_id`, and `rating` (implicit feedback).
