from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql import functions as F

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("Data Preprocessing for NCF") \
    .getOrCreate()

# Load the dataset
# Replace the input path with the actual file path
data = spark.read.parquet('ratings_data.parquet')

# Data Exploration
# Checking for missing values and duplicates
data.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in data.columns]).show()
print("Number of duplicate entries: ", data.dropDuplicates(['user_id', 'item_id']).count())

# Handle Missing Values
# Dropping rows with any missing values
data_clean = data.dropna()

# Remove Duplicates
# Removing duplicate user-item interactions
data_clean = data_clean.dropDuplicates(['user_id', 'item_id'])

# Feature Engineering - Implicit Feedback
# Converting ratings into implicit feedback (0 or 1)
data_clean = data_clean.withColumn('implicit_feedback', when(col('rating') > 0, 1).otherwise(0))

# Re-indexing User and Item IDs
# Re-index user_id and item_id to start from 0
from pyspark.ml.feature import StringIndexer

user_indexer = StringIndexer(inputCol='user_id', outputCol='user_id_index')
item_indexer = StringIndexer(inputCol='item_id', outputCol='item_id_index')

data_clean = user_indexer.fit(data_clean).transform(data_clean)
data_clean = item_indexer.fit(data_clean).transform(data_clean)

# Train-Test Split (80/20 split)
train_data, test_data = data_clean.randomSplit([0.8, 0.2], seed=42)

# Save processed datasets as Parquet files (optional)
train_data.write.parquet('train_data.parquet', mode='overwrite')
test_data.write.parquet('test_data.parquet', mode='overwrite')

# Convert to Pandas DataFrame (if needed)
# Switching to Pandas if further processing requires it (e.g., for TensorFlow/Keras input)
train_pandas = train_data.toPandas()
test_pandas = test_data.toPandas()

# Save
train_pandas.to_csv('train_data.csv', index=False)
test_pandas.to_csv('test_data.csv', index=False)

print("Data preprocessing complete!")
