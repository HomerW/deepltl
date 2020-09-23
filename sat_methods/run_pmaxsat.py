import os
import pickle
import numpy as np
from pmaxsat import create_pmaxsat
from utils import print_formula, err
import time
import argparse

MAX_TIME = 300

"""
Runs PMAX-SAT on a single formula. Since multiple formulas can be found,
we need to record all of them. Thus can't run a separate thread for timeout
purposes as in SAT. Instead we check the amount of time left after each run of
the solver.
"""
def single_formula_run_pmaxsat(positive_traces,negative_traces,trace_length,lits,formula_length):
    start_time = time.time()
    res = []
    time_left = MAX_TIME
    k = 1
    while time_left > 0:
        optimizer = create_pmaxsat(k,trace_length,lits,positive_traces,negative_traces)
        r = optimizer.check()
        model = optimizer.model()
        formula_print = print_formula(model)
        error = err(model,len(positive_traces),len(negative_traces))
        total_time = (time.time()-start_time)
        time_left = MAX_TIME - total_time
        if time_left > 0:
            res.append((formula_length, total_time, formula_print, 1.0-error))
        k += 1
    return res

def run_pmaxsat(data_file, output_file, num_formulas, trace_length, lits):
    for n in range(1, 16):
        count = 0
        for j in range(100):
            # some formulas may be skipped so read in as many files
            # as it takes to get to num_formulas (up to 100)
            if count >= num_formulas:
                break
            if os.path.exists(f"{data_file}/{n}/train-{j}"):
                count += 1
                traces,labels = pickle.load(open(f"{data_file}/{n}/train-{j}", "rb"))
                positive_traces = []
                negative_traces = []
                for trace, label in zip(traces,labels):
                    if label == 1.0:
                        positive_traces.append(trace.numpy().astype(np.bool_).tolist())
                    else:
                        negative_traces.append(trace.numpy().astype(np.bool_).tolist())
                res = single_formula_run_pmaxsat(positive_traces,negative_traces,trace_length,lits,n)
                print(f"Size {n} number {count}: {res}")
                with open(output_file,"ab+") as file:
                    pickle.dump(res,file)

if __name__ == "__main__":
    trace_length = 15
    lits = ["a","b","c"]
    num_formulas = 50
    parser = argparse.ArgumentParser(description="Inputs for LTL test")
    parser.add_argument('--data_path',required=True,type=str,help="Path to traces for testing")
    parser.add_argument('--output_file',required=True,type=str,help="File to write outputs to using pickle")
    args = parser.parse_args()
    run_pmaxsat(args.data_path, args.output_file, num_formulas, trace_length, lits)
