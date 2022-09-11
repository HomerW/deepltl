import numpy as np
from itertools import product
from pyeda.inter import *
from pyeda.boolalg.expr import Variable, OrOp, AndOp, Complement, _Zero, _One
import spot

def len_form(f):
    return spot.length(spot.formula(f)) - f.count("!")

def translate_layer(layer, metric=False):
    w_prop = layer[0]
    if metric:
        w_metric = layer[1]
        init_metric = layer[2]
        w_qual = np.clip(layer[3], 0, None)
        bias = layer[4]
        init_run = layer[5]
    else:
        w_qual = np.clip(layer[1], 0, None)
        bias = layer[2]
        init_run = layer[3]

    num_var = w_prop.shape[0]
    num_filters = w_prop.shape[1]

    if metric:
        w_augment = np.concatenate([w_prop, w_metric, np.expand_dims(w_qual, 0)], axis=0)
        ttable_temp_var = ttvars('x', 2*num_var)
        ttable_nontemp_var = ttvars('x', 2*num_var)
        ttable = np.flip(np.array(list(product([1, 0], repeat=2*num_var+1))), axis=0)
    else:
        w_augment = np.concatenate([w_prop, np.expand_dims(w_qual, 0)], axis=0)
        ttable_temp_var = ttvars('x', num_var)
        ttable_nontemp_var = ttvars('x', num_var)
        ttable = np.flip(np.array(list(product([1, 0], repeat=num_var+1))), axis=0)

    formulas = []
    output = np.matmul(ttable, w_augment) + bias

    for f in range(num_filters):
        ttable_temp = truthtable(ttable_temp_var,
                                 np.transpose(output[np.nonzero(ttable[:, -1]), f]) > 0)
        ttable_nontemp = truthtable(ttable_nontemp_var,
                                    np.transpose(output[np.nonzero(-1 * (ttable[:, -1] - 1)), f]) > 0)
        f_temp = espresso_tts(ttable_temp)
        f_nontemp = espresso_tts(ttable_nontemp)
        if metric:
            formulas.append((f_temp[0], f_nontemp[0], num_var, init_metric[0], init_run[0][f]))
        else:
            formulas.append((f_temp[0], f_nontemp[0], num_var, init_run[0][f]))
    return formulas

"""
Takes the weights from a trained DeepLTL model and returns the corresponding
LTL formula as a string
"""
def translate(layers, lits, metric=True):
    layer_formulas = [translate_layer(l, metric) for l in layers]

    simpopt = spot.tl_simplifier_options()
    simpopt.reduce_basics = True
    simpopt.reduce_size_strictly = True
    simpopt.synt_impl = True
    simpopt.event_univ = True
    simpopt.containment_checks = True
    simpopt.containment_checks_stronger = True
    simpopt.favor_event_univ = True
    simpopt.nenoform_stop_on_boolean = False
    simpopt.boolean_to_isop = True

    def compile_formula(layer, lits):
        lits = list(reversed(lits))
        if metric:
            temp, non_temp, num_var, init_metric, init_run = layer
        else:
            temp, non_temp, num_var, init_run = layer

        def pyeda_to_spot_string(form):
            if isinstance(form, Variable) or isinstance(form, Complement):
                group = str(form)
                group = group.replace("~", "!")
                for i in range(num_var):
                    if metric:
                        group = group.replace(f"x[{i+num_var}]", f"({lits[i]})")
                        if init_metric[i] > 0:
                            group = group.replace(f"x[{i}]", f"X ({lits[i]})")
                        else:
                            # since spot doesn't have a weak next op. we introduce
                            # a 'last' proposition denoting the end of the trace
                            # UPDATE THIS: NEW VERSION OF SPOT INTRODUCES WEAK NEXT
                            group = group.replace(f"x[{i}]", f"(X ({lits[i]}) | X(last))")
                    else:
                        group = group.replace(f"x[{i}]", f"({lits[i]})")
                return f"({group})"
            elif isinstance(form, OrOp):
                groups = list(map(pyeda_to_spot_string, form.xs))
                formula = "false"
                for group in groups:
                    formula = f"({formula}) | ({group})"
                return formula
            elif isinstance(form, AndOp):
                groups = list(map(pyeda_to_spot_string, form.xs))
                formula = "true"
                for group in groups:
                    formula = f"({formula}) & ({group})"
                return formula
            elif isinstance(form, _Zero):
                return "0"
            elif isinstance(form, _One):
                return "1"
            else:
                assert False, "unknown pyeda formula"

        temp_formula = spot.formula(pyeda_to_spot_string(temp))
        non_temp_formula = spot.formula(pyeda_to_spot_string(non_temp))
        temp_formula = spot.tl_simplifier(simpopt).simplify(temp_formula)
        non_temp_formula = spot.tl_simplifier(simpopt).simplify(non_temp_formula)

        if init_run > 0:
            formula = spot.formula.W(temp_formula, non_temp_formula)
        else:
            formula = spot.formula.U(temp_formula, non_temp_formula)

        formula = spot.tl_simplifier(simpopt).simplify(formula)
        return formula

    for layer in layer_formulas:
        forms = [compile_formula(filter, lits) for filter in layer]
        lits = forms

    # only one formula in forms since all networks should end with 1 filter
    return_formula = forms[0]

    # remove W and M operators and put in NNF to match input and SAT
    return_formula = spot.unabbreviate(return_formula, "WM").negative_normal_form()

    return str(return_formula)
