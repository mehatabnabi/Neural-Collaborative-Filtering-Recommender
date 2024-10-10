# ncf_model.py

import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

# Load the datasets
train_data = pd.read_csv('train_data.csv')

# Model hyperparameters
embedding_size = 50
num_users = train_data['user_id'].nunique()
num_items = train_data['item_id'].nunique()

# Define Inputs
user_input = Input(shape=(1,), name='user_input')
item_input = Input(shape=(1,), name='item_input')

# Embedding layers
user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
item_embedding = Embedding(input_dim=num_items, output_dim=embedding_size, name='item_embedding')(item_input)

# Flatten embeddings
user_vector = Flatten()(user_embedding)
item_vector = Flatten()(item_embedding)

# Concatenate user and item embeddings
concatenated = Concatenate()([user_vector, item_vector])

# Dense layers for the Neural Collaborative Filtering
dense_1 = Dense(128, activation='relu')(concatenated)
dense_2 = Dense(64, activation='relu')(dense_1)

# Output layer
output = Dense(1, activation='sigmoid')(dense_2)

# Define model
ncf_model = Model(inputs=[user_input, item_input], outputs=output)

# Compile the model
ncf_model.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)

# Prepare input data
train_user_input = train_data['user_id'].values
train_item_input = train_data['item_id'].values
train_labels = train_data['rating'].values

# Train the model
ncf_model.fit([train_user_input, train_item_input], train_labels, epochs=10, batch_size=256, validation_split=0.1)

# Save the trained model
ncf_model.save('ncf_model.h5')
