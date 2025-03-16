from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import pandas as pd

# Sample data
def get_data(path, seed):
    # Load data
    df = pd.read_csv(path, encoding='utf-8')

    # Shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split into train and validation set
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    x_train, y_train = train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values

    return x_train, y_train, x_val, y_val

# Load data
X_train, y_train, X_test, y_test = get_data(
    path=os.path.join('data', 'train.csv'), seed=24323)
print("Data Loaded")

# Create a pipeline that combines Count vectorization and MNB classification
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")