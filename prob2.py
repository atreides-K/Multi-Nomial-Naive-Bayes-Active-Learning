import os

from utils import *
from model import *
import time

def main(args: argparse.Namespace):
    # set seed for reproducibility
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no+args.run_id)

    # Load the preprocessed data
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        idxs =\
            np.random.RandomState(args.run_id).permutation(X_train_vec.shape[0])
        X_train_vec = X_train_vec[idxs]
        y_train = y_train[idxs]
        print("Preprocessed Data Loaded")
    else:
        raise Exception("Preprocessed Data not found")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    accs=[]
    total_items = 10_000
    idxs = np.arange(10_000)
    remaining_idxs = np.setdiff1d(np.arange(X_train_vec.shape[0]), idxs)
    # add time fr report
    train_times = []
    
    # Train the model
    for i in range(1, 60):
        start_time = time.time()
        X_train_batch=X_train_vec[idxs]
        y_train_batch=y_train[idxs]
        if i==1:
            model.fit(X_train_batch, y_train_batch)
        else:
            model.fit(X_train_batch, y_train_batch, update=True)
        # for the plot thingy
        train_time = time.time() - start_time
        train_times.append(train_time)
        

        y_preds=model.predict(X_val_vec)
        val_acc=np.mean(y_preds == y_val)
        print(f"{total_items} items - Val acc: {val_acc}")
        accs.append(val_acc)

        if args.is_active:
            # Active learning: select the most uncertain samples
            y_probs = model.predict_prob(X_train_vec[remaining_idxs])
            
            # Calculate entropy-based uncertainty
            y_log_probs=np.log(y_probs + 1e-10)  # Preventin log(0)
            uncertainty=-np.sum(y_probs*y_log_probs,axis=1)  #Entropy calc

            
            # Selecting the top 5% most uncertain samples
            uncertain_idxs=np.argpartition(uncertainty,-5000)[-5000:]
            
            # Ensure we don't select duplicates
            selected=np.unique(remaining_idxs[uncertain_idxs],return_index=True)[1]
            uncertain_idxs=uncertain_idxs[selected][:5000]
            
            # Update indices
            idxs=np.concatenate([idxs,remaining_idxs[uncertain_idxs]])
            remaining_idxs=np.delete(remaining_idxs,uncertain_idxs)
            # raise NotImplementedError
        else:
            idxs=np.concatenate([idxs,remaining_idxs[:5_000]])
            remaining_idxs=remaining_idxs[5_000:]
        total_items+=5_000
    # Saving timings
    np.save(f"{args.logs_path}/times_{args.run_id}_{args.is_active}.npy", train_times)
    accs = np.array(accs)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy",accs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=100_000)
    parser.add_argument("--smoothing", type=float, default=70.5)
    main(parser.parse_args())
