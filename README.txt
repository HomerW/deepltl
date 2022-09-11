Code for NeuralLTLf - https://arxiv.org/abs/2111.04147
--------------------------------------------
Python 3.6+ is required along with the pip packages listed in requirements.txt
and the Spot LTL library.

The required versions of the pip packages can be installed with
pip install -r requirements.txt

Instructions for installing the Spot LTL library and the associated python
bindings can be found at:
https://spot.lrde.epita.fr/install.html
This code was tested with version 2.9.3.

The data directory contains the original and noisy datasets described in the
paper. The formulas directory contains the associated formulas.

Executable Code Files
--------------------------------------------
train_deepltl.py runs NeuralLTLf on a specified dataset using the hyperparameters
described in the paper and saves the model checkpoints to a specified directory.
For example:

python train_deepltl.py --data_path=data/original --train_path=training

After NeuralLTLf has been fully trained. extract_formulas.py extracts the formulas
and writes them to a directory using the model checkpoints. For example:

python extract_formulas.py --data_path=data/test --train_path=training --output_path=formula_out

The code for the SAT methods can be found in the sat_methods directory.
run_pmaxsat.py and run_sat.py run the SAT and PMAX-SAT methods on the specified
data. For example:

python run_sat.py --data_path=../data/original --output_file=sat.pkl

print_results.py prints the results of a run of a SAT-based method. For example

python print_results.py --results_file=sat.pkl --sat

Other Relevant Files
--------------------------------------------------
LTLOperator.py contains the implementation of the custom neural operator NeuralLTLf
uses.

translation.py contains the logic that converts network weights into an LTL
formula.

data.py contains the algorithm for generating a characteristic sample of a
formula and the function used to randomly sample traces to mix with the
characteristic sample.
