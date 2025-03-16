import os
from multiprocessing import Pool

def run_active(run_id):
    os.system(f"python prob2.py --sr_no 24323 --run_id {run_id} --is_active")

def run_random(run_id):
    os.system(f"python prob2.py --sr_no 24323 --run_id {run_id}")

if __name__ == "__main__":
    with Pool(5) as p:
        p.map(run_active, range(1,6))  # Active runs
        p.map(run_random, range(1,6))  # Random runs
