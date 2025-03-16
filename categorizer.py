import os
from collections import defaultdict

def read_file_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.readlines()

def compare_files(file_list):
    if not file_list:
        print("No matching files found.")
        return
    
    clusters = []
    file_to_cluster = {}
    differences = {}
    
    for file in file_list:
        lines = read_file_lines(file)
        matched_cluster = None
        
        for cluster in clusters:
            cluster_rep_lines = read_file_lines(cluster[0])
            if lines == cluster_rep_lines:
                cluster.append(file)
                file_to_cluster[file] = cluster[0]
                matched_cluster = cluster
                break
        
        if not matched_cluster:
            clusters.append([file])
            file_to_cluster[file] = file
    
    for cluster in clusters:
        if len(cluster) > 1:
            continue
        file = cluster[0]
        differences[file] = []
        for other_file in file_list:
            if other_file == file:
                continue
            other_lines = read_file_lines(other_file)
            diff_lines = [
                (i + 1, lines[i].strip(), other_lines[i].strip())
                for i in range(min(len(lines), len(other_lines)))
                if lines[i] != other_lines[i]
            ]
            if diff_lines:
                differences[file].append((other_file, diff_lines[0]))
    
    print("Identical file clusters:")
    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {', '.join(os.path.basename(f) for f in cluster)}")
    
    if differences:
        print("\nDifferences found:")
        for file, diffs in differences.items():
            for other_file, (line_no, ref_line, diff_line) in diffs:
                print(f"Between {os.path.basename(file)} and {os.path.basename(other_file)} at line {line_no}:")
                print(f" {os.path.basename(file)} : {ref_line}")
                print(f" {os.path.basename(other_file)} : {diff_line}\n")

if __name__ == "__main__":
    directory = os.getcwd()
    files = [f for f in os.listdir(directory) if f.startswith("predictions") and f.endswith(".csv")]
    files = [os.path.join(directory, f) for f in files]
    compare_files(files)