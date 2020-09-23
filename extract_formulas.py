import argparse
from translation import translate
from models import get_model_zero, get_model_one, get_model_two
import spot
import os
import pickle

batch_size = 100

"""
Takes a string representation of a formula in Spot and strips whitespace and
parentheses to calculate length
"""
def len_form(f):
    return len(f.replace(" ", "").replace("(", "").replace(")", ""))

"""
Given model checkpoints, extracts the best LTL formulas
"""
def extract_formulas(models, data_path, train_path, output_path):
    lits = [spot.formula("a"), spot.formula("b"), spot.formula("c")]
    formulas = []
    for name, get_model in enumerate(models):
        model = get_model()
        model.compile(loss='binary_crossentropy', metrics=['accuracy'])
        formulas_m = []
        for n in range(1, 16):
            formulas_n = []
            count = 0
            # some formulas may be skipped so read in as many files
            # as it takes to get to num_formulas (up to 100)
            for i in range(100):
                if count >= 50:
                    break
                checkpoint_path = f"{train_path}/{name}/{n}/cp-{i}.ckpt"
                if not os.path.exists(f"{train_path}/{name}/{n}/cp-{i}.ckpt.index"):
                    continue
                with open(f"{data_path}/{n}/train-{i}", "rb") as file:
                    train_traces, train_labels = pickle.loads(file.read())
                count += 1
                model.load_weights(checkpoint_path).expect_partial()
                acc = model.evaluate(train_traces, train_labels, batch_size=batch_size)[1]
                layer_weights = [l.get_weights() for l in model.layers[:-2]]
                f = translate(layer_weights, lits, metric=False)
                formulas_n.append((acc, f))
                print(f"Translated formula size {n} number {count}")
            formulas_m.append(formulas_n)
        formulas.append(formulas_m)

    for n in range(15):
        final_formulas = []
        for forms in zip(formulas[0][n], formulas[1][n], formulas[2][n]):
            max_acc = max(forms, key=lambda x: x[0])[0]
            max_forms = [f for acc, f in forms if acc == max_acc]
            form_lens = [len_form(f) for f in max_forms]
            best_idx = min(enumerate(form_lens), key=lambda x: x[1])[0]
            final_formulas.append(max_forms[best_idx])
        with open(f"{output_path}/{n+1}.txt", "w") as file:
            for f in final_formulas:
                file.write(f"{f}\n")

if __name__ == "__main__":
    models = [get_model_zero, get_model_one, get_model_two]
    parser = argparse.ArgumentParser(description="Inputs for DeepLTL formula extraction")
    parser.add_argument('--data_path',required=True,type=str,help="Path to traces")
    parser.add_argument('--train_path',required=True,type=str,help="Path to read model checkpoints from")
    parser.add_argument('--output_path',required=True,type=str,help="Path to write formulas to")
    args = parser.parse_args()
    extract_formulas(models, args.data_path, args.train_path, args.output_path)
