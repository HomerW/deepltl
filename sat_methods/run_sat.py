import timeout_decorator
import os
import pickle
import numpy as np
from pmaxsat import create_pmaxsat
from sat import create_sat
from utils import print_formula, err
import time
import argparse


MAX_TIME = 300
"""
Runs SAT on a single formula for MAX_TIME seconds
"""
@timeout_decorator.timeout(MAX_TIME,use_signals=False)
def single_formula_run_sat(positive_traces,negative_traces,trace_length,lits,formula_length):
    start_time = time.time()
    k = 1
    while True:
        optimizer = create_sat(k, trace_length, lits, positive_traces, negative_traces)
        r = optimizer.check()
        k += 1
        if r.r == -1:
            continue
        else:
            break

    model = optimizer.model()
    formula_print = print_formula(model)
    error = err(model,len(positive_traces),len(negative_traces))
    total_time = (time.time()-start_time)
    return formula_length, total_time, formula_print, 1.0-error

def run_sat(data_file,output_file,num_formulas,trace_length,lits):
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
                try:
                    res = single_formula_run_sat(positive_traces,negative_traces,trace_length,lits,n)
                except timeout_decorator.timeout_decorator.TimeoutError:
                    res = (n,MAX_TIME,"TIMED OUT", 0.0)
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
    run_sat(args.data_path,args.output_file,num_formulas,trace_length,lits)
