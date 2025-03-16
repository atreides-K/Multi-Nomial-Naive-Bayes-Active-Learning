import os

from utils import *
from model import *


def main(args: argparse.Namespace):
    # set seed for reproducibility
    set_seed(args.sr_no)
    print(f"sr_no: {args.sr_no}")
    print(f"data_path: {args.data_path}")
    print(f"train_file: {args.train_file}")
    print(f"intermediate: {args.intermediate}")
    print(f"max_vocab_len: {args.max_vocab_len}")
    print(f"smoothing: {args.smoothing}")
    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")
    
    # Print the first 5 rows of X_train
    print("First 5 rows of X_train:")
    for i, row in enumerate(X_train[:5]):
        print(f"Row {i+1}: {row}")

    # Print the first 5 rows of y_train
    print("First 5 rows of y_train:")
    for i, row in enumerate(y_train[:5]):
        print(f"Row {i+1}: {row}")

    # Print the first 5 rows of X_val
    print("First 5 rows of X_val:")
    for i, row in enumerate(X_val[:5]):
        print(f"Row {i+1}: {row}")

    # Print the first 5 rows of y_val
    print("First 5 rows of y_val:")
    for i, row in enumerate(y_val[:5]):
        print(f"Row {i+1}: {row}")
    # Preprocess the data
    vectorizer = Vectorizer(max_vocab_len=args.max_vocab_len,test=True)
    vectorizer.fit(X_train)
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        print("Preprocessed Data Loaded")
    else:
        X_train_vec = vectorizer.transform(X=X_train)
        pickle.dump(
            X_train_vec,
            open(f"{args.data_path}/X_train{args.intermediate}", "wb"))
        pickle.dump(
            y_train, open(f"{args.data_path}/y_train{args.intermediate}", "wb"))
        X_val_vec = vectorizer.transform(X=X_val)
        pickle.dump(
            X_val_vec,
            open(f"{args.data_path}/X_val{args.intermediate}", "wb"))
        pickle.dump(
            y_val, open(f"{args.data_path}/y_val{args.intermediate}", "wb"))
        print("Data Preprocessed")


    # Train the model
    # Hyperparameter tuning
    alphas = [70,70.5,70.75,71,71.25,71.5,72,73]
    best_alpha,grid_results=grid_search_alpha(X_train_vec,y_train,X_val_vec,y_val,alphas)

    # Display grid search results
    print("\nGrid Search Results:")
    for alpha,val_acc in grid_results:
        print(f"Alpha: {alpha} | Validation Accuracy: {val_acc}")

    # Final training with best alpha
    print("\nTraining final model with optimized alpha...")
    final_model=MultinomialNaiveBayes(alpha=best_alpha)
    final_model.fit(X_train_vec, y_train)

    # Detailed evaluation
    train_acc,val_acc=evaluate_model(final_model, X_train_vec, y_train, X_val_vec, y_val)
    print(f"\nFinal Model Performance:")
    print(f"Train Accuracy: {train_acc}")
    print(f"Validation Accuracy: {val_acc}")

    # Load the test data
    if os.path.exists(f"{args.data_path}/X_test{args.intermediate}"):
        X_test_vec = pickle.load(open(
            f"{args.data_path}/X_test{args.intermediate}", "rb"))
        print("Preprocessed Test Data Loaded")
    else:
        X_test = pd.read_csv(
            f"{args.data_path}/X_test_{args.sr_no}.csv", header=None
        ).values.squeeze()
        print("Test Data Loaded")
        X_test_vec = vectorizer.transform(X=X_test)
        pickle.dump(
            X_test_vec,
            open(f"{args.data_path}/X_test{args.intermediate}", "wb"))
        print("Test Data Prepro"
        "cessed")
    preds = final_model.predict(X_test_vec)
    with open(f"predictions.csv", "w") as f:
        for pred in preds:
            f.write(f"{pred}\n")
    print("Predictions Saved to predictions.csv")
    print("You may upload the file at http://10.192.30.174:8000/submit")



# Add these new functions before main()
def evaluate_model(model, X_train, y_train, X_val, y_val):
    """Evaluate model performance on train and validation sets"""
    y_pred_train = model.predict(X_train)
    train_acc = np.mean(y_pred_train == y_train)
    
    y_pred_val = model.predict(X_val)
    val_acc = np.mean(y_pred_val == y_val)
    
    return train_acc, val_acc

def grid_search_alpha(X_train, y_train, X_val, y_val, alphas):
    """Perform grid search over alpha values"""
    best_alpha = alphas[0]
    best_val_acc = 0
    results = []

    print("\nAlpha Grid Search Results:")
    for alpha in alphas:
        model = MultinomialNaiveBayes(alpha=alpha)
        model.fit(X_train, y_train)
        _, val_acc = evaluate_model(model, X_train, y_train, X_val, y_val)
        
        results.append((alpha, val_acc))
        if val_acc > best_val_acc:
            best_alpha = alpha
            best_val_acc = val_acc
        
        print(f"Alpha: {alpha} | Val Acc: {val_acc}")
    
    print(f"\nBest Alpha: {best_alpha} with Val Acc: {best_val_acc}")
    return best_alpha, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--train_file", type=str, default="train.csv")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=100_000)
    parser.add_argument("--smoothing", type=float, default=70.5)
    main(parser.parse_args())
