import pickle
import argparse

"""
Prints the results of a SAT or PMAX-SAT test
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Results to Read")
    parser.add_argument('--results_file',required=True,type=str,help="Path to results file")
    parser.add_argument('--sat', default=False, action='store_true')
    parser.add_argument('--pmaxsat', dest='sat', action='store_false')

    args = parser.parse_args()
    with open(args.results_file, "rb") as file:
        results = []
        while True:
            try:
                results.append(pickle.load(file))
            except EOFError:
                break

    for n in range(1, 16):
        if args.sat:
            accs = [res[-1] for res in results if res[0] == n]
            print(f"Size {n}")
            print(f"Total formulas: {len(accs)}")
            print(f"Average accuracy: {sum(accs)/len(accs)}")
            print(f"Percent zero-loss formulas: {len([x for x in accs if x == 1])/len(accs)}")
            print("-----------------------------")
        else:
            # take best accuracy of formulas found in under 300 seconds
            accs = [max([x[-1] for x in res if x[1] < 300]) for res in results if res[0][0] == n]
            print(f"Size {n}")
            print(f"Total formulas: {len(accs)}")
            print(f"Average accuracy: {sum(accs)/len(accs)}")
            print(f"Percent zero-loss formulas: {len([x for x in accs if x == 1])/len(accs)}")
            print("-----------------------------")
