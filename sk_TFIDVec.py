from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Print the first 5 rows of X_train
print("First 5 rows of X_train:")
for i, row in enumerate(X_train[:5]):
    print(f"Row {i+1}: {row}")

# Print the first 5 rows of y_train
print("First 5 rows of y_train:")
for i, row in enumerate(y_train[:5]):
    print(f"Row {i+1}: {row}")

# Print the first 5 rows of X_test
print("First 5 rows of X_val:")
for i, row in enumerate(X_test[:5]):
    print(f"Row {i+1}: {row}")

# Print the first 5 rows of y_test
print("First 5 rows of y_val:")
for i, row in enumerate(y_test[:5]):
    print(f"Row {i+1}: {row}")

# Create a pipeline that combines Count vectorization and MNB classification
pipeline = make_pipeline(CountVectorizer(), MultinomialNB())

# Define the parameter grid
param_grid = {
    'countvectorizer__max_features': [1000, 5000, 10000],
    'multinomialnb__alpha': [0.1, 0.5, 1.0]
}

# Perform grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters and best score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_}")

# Predict on the test set using the best estimator
y_pred = grid_search.best_estimator_.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy}")