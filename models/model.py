import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split



df = pd.read_csv("dataset/ratings_subset.csv")
movies_df = pd.read_csv("dataset/full-dataset/movies.csv")

#index to userId
df["user_idx"] = df["userId"].astype("category").cat.codes
#index to movieID
df["movie_idx"] = df["movieId"].astype("category").cat.codes

idx_to_movie_id = dict(enumerate(df["movieId"].astype("category").cat.categories))
movie_id_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

def model_train():
    """
    trains a model from given dataset
    """
    #input features
    X = df[["user_idx","movie_idx"]].values
    y = df["rating"].values
    #split into train and train
    X_train,X_test,y_train,y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )

    num_users = df["user_idx"].nunique()
    num_movies = df["movie_idx"].nunique()
    embedding_size = 32  # can be 16–64 depending on dataset size

    # Inputs
    user_input = tf.keras.Input(shape=(1,))
    movie_input = tf.keras.Input(shape=(1,))

    # Embeddings
    user_embedding = tf.keras.layers.Embedding(num_users, embedding_size)(user_input)
    movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_size)(movie_input)

    # Flatten
    user_vec = tf.keras.layers.Flatten()(user_embedding)
    movie_vec = tf.keras.layers.Flatten()(movie_embedding)

    # Combine user + movie vectors
    x = tf.keras.layers.Concatenate()([user_vec, movie_vec])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(1)(x)  # Output = predicted rating

    # Build model
    model = tf.keras.Model(inputs=[user_input, movie_input], outputs=output)

    # Compile
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(
        [X_train[:, 0], X_train[:, 1]],  # user_idx, movie_idx
        y_train,                         # actual ratings
        epochs=5,                        # try 5 to start, can increase later
        batch_size=64,                  # how many samples per training step
        validation_split=0.1,           # 10% of training data used for validation
        verbose=1                        # shows training progress
    )

    model.save("models/test_model.keras")


def predict(model_path,user_id,n):
    """
    Returns the top n predicted ratings for movies the user has never seen
    Args:
        model_path: model path
        user_id: user id
        n: number of movies to return
    """
    model = tf.keras.models.load_model(model_path)
    rated_movies = df[df["user_idx"] == user_id]["movie_idx"].values
    all_movies = df["movie_idx"].unique()
    unseen_movies = [m for m in all_movies if m not in rated_movies]
    user_input = np.full(len(unseen_movies), user_id)
    movie_input = np.array(unseen_movies)
    predictions = model.predict([user_input,movie_input], verbose = 0)
    results = list(zip(unseen_movies, predictions.flatten()))
    # Sort by predicted rating, descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:n] 

result = predict("models/test_model.keras",3,5)
for count,i in enumerate(result):
    movie_id = idx_to_movie_id[i[0]]
    print(f"movie {count+1}: {movie_id_to_title[movie_id]}, predicted rating: {i[1]:.2f}")