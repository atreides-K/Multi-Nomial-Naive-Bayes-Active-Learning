import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"
    # for i in [True, False]:
    #     for j in range(1, 6):
    #         assert os.path.exists(os.path.join(args.logs_path, f"run_{j}_{i}.npy")),\
    #             f"File run_{j}_{i}.npy not found in {args.logs_path}"
    # the accuracies for the two settings (active and random strategies)
    
    # Initialize storage for results for both active and randomized
    strategies = {'Active': [], 'Random': []}
    
    # TODO: Load data and plot the standard means and standard deviations of
    for strategy in [True, False]:
        all_runs = []
        for run_id in range(1, 6):
            file_path = os.path.join(args.logs_path, f"run_{run_id}_{strategy}.npy")
            data = np.load(file_path)
            all_runs.append(data)
        
        
        # TODO: also ensure that the files have the same length
        
        min_length = min(len(run) for run in all_runs)
        trimmed_runs = [run[:min_length] for run in all_runs]
        strategies['Active' if strategy else 'Random'] = np.array(trimmed_runs)



    # Calculate statistics
    stats = {}
    for name, data in strategies.items():
        stats[name] = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'x': np.arange(1, data.shape[1]+1)*5000 + 10000  # Starting from 10k
        }




    # Plot configuration
    plt.figure(figsize=(10, 6))
    colors = {'Active':'#4C72B0', 'Random':'#DD8452'}
    
    for name in ['Active', 'Random']:
        plt.plot(stats[name]['x'], stats[name]['mean'],label=name,color=colors[name])
        plt.fill_between(stats[name]['x'],stats[name]['mean']-stats[name]['std'],stats[name]['mean']+stats[name]['std'],color=colors[name],alpha=0.2)

    # Adding supervised baseline
    plt.axhline(y=args.supervised_accuracy,color='#55A868',linestyle='--',label='Supervised Baseline' )
    




    # Formatting
    plt.title('Active Learning vs Random Sampling Performnce')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f'plot_{args.sr_no}.png')
    print(f"Plot saved as plot_{args.sr_no}.png")

    # Time comparison plot
    plt.figure(figsize=(10, 6))
    
    for strategy in [True, False]:
        all_times = []
        for run_id in range(1, 6):
            time_file = os.path.join(args.logs_path, f"times_{run_id}_{strategy}.npy")
            times = np.load(time_file)
            all_times.append(times[:min_length])  # Match accuracy data length
            
        name = 'Active' if strategy else 'Random'
        mean_time = np.mean(all_times, axis=0)
        std_time = np.std(all_times, axis=0)
        
        # Plot cumulative time
        plt.plot(stats[name]['x'], np.cumsum(mean_time), 
                 label=name, color=colors[name])
        plt.fill_between(stats[name]['x'],
                         np.cumsum(mean_time - std_time),
                         np.cumsum(mean_time + std_time),
                         color=colors[name], alpha=0.2)

    plt.title('Training Time Comparison: Update vs Retrain')
    plt.xlabel('Number of Labeled Samples')
    plt.ylabel('Cumulative Training Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'time_comparison_{args.sr_no}.png')
    print(f"Time plot saved as time_comparison_{args.sr_no}.png")

    # raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())
