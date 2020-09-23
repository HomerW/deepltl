import re

"""
Formula printing functions
"""
def print_formula(model):
  formula = ""
  operator_mapping = {
      r'AND\(\d+\)': "^",
      r'OR\(\d+\)': "v",
      r'NEXT\(\d+\)': "X",
      r'WNEXT\(\d+\)': "N",
      r'UNTIL\(\d+\)': "U",
      r'RELEASE\(\d+\)': "R",
      r'EVENTUALLY\(\d+\)': "F",
      r'ALWAYS\(\d+\)': "G",
      r'LIT\(\d+\)': "TERM"
  }
  binary_operators = [r'AND\(\d+\)',r'OR\(\d+\)',r'UNTIL\(\d+\)',r'RELEASE\(\d+\)']
  return print_formula_recurse(1,model,operator_mapping,binary_operators)

def find_subformulae(s,model,is_binary):
  a_index = None
  b_index = None
  for statement in model:
      if model[statement]:
        if re.match(r'A\(\d+, \d+\)',str(statement)):
          numbers = re.findall(r'\d+',str(statement))
          if int(numbers[0]) == s:
            a_index = int(numbers[1])
        if is_binary:
          if re.match(r'B\(\d+, \d+\)',str(statement)):
            numbers = re.findall(r'\d+',str(statement))
            if int(numbers[0]) == s:
              b_index = int(numbers[1])
  if is_binary:
    return a_index, b_index
  else:
    return a_index

def find_literal(s,model):
  for statement in model:
    if model[statement]:
      if re.match(r'L\(\d+, -?[a-zA-Z]+\)',str(statement)):
        number = int(re.search(r'\d+',str(statement)).group())
        if number == s:
          literal = re.findall(r'-?[a-zA-Z]+',str(statement))[1]
          return literal
  return None

def print_formula_recurse(s,model,operator_mapping,binary_operators):
  formula = ""
  for statement in model:
      if model[statement]:
        for operator in operator_mapping.keys():
          if re.match(operator,str(statement)):
            if int(re.search(r'\d+',str(statement)).group()) == s:
              # Literal case
              if operator == r'LIT\(\d+\)':
                formula = find_literal(s,model)
                return formula
              formula = formula + operator_mapping[operator]
              # Binary operator case
              if operator in binary_operators:
                a_index, b_index = find_subformulae(s,model,True)
                left_formula = print_formula_recurse(a_index,model,operator_mapping,binary_operators)
                right_formula = print_formula_recurse(b_index,model,operator_mapping,binary_operators)
                formula = "(" + left_formula + ")" + formula + "(" + right_formula + ")"
                return formula
              # Unary operator case
              else:
                a_index = find_subformulae(s,model,False)
                right_formula = print_formula_recurse(a_index,model,operator_mapping,binary_operators)
                formula = formula + "(" + right_formula + ")"
                return formula
  return formula

def err(model, num_pos, num_neg):
  regex_p = re.compile('RUN\(([0-9])+, 1, 1\)')
  regex_n = re.compile('RUN_d\(([0-9])+, 1, 1\)')
  pos = 0
  neg = 0
  for v in model:
    if model[v]:
      if regex_p.match(str(v)):
        pos = pos + 1
      if regex_n.match(str(v)):
        neg = neg + 1
  return 1 - (pos+neg)/(num_pos+num_neg)
