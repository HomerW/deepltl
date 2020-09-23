from z3 import *
from itertools import product, combinations, permutations

"""
N - length of formula
k - length of traces
n - number of variables
"""

def create_pmaxsat(N, k, lits, pos_traces, neg_traces, metric=False):
  optimizer = Optimize()
  n = len(lits)

  def one_of(v):
    # either strategy works here
    # return And(Or(v), And([Not(And(i, j)) for i, j in combinations(v, 2)]))
    return PbEq([(elem, 1) for elem in v], 1)

  # skeleton type variables
  if metric:
    skel_names = ["AND", "OR", "NEXT", "WNEXT", "UNTIL", "RELEASE", "EVENTUALLY", "ALWAYS", "LIT"]
  else:
    skel_names = ["AND", "OR", "UNTIL", "EVENTUALLY","RELEASE", "ALWAYS", "LIT"]

  for s in range(1, N+1):
    skel_vars = [Bool(f"{name}({s})") for name in skel_names]
    # add and restrict to one skeleton per node
    optimizer.add(one_of(skel_vars))

  # trace variables - L(s, v) and L(s, -v)
  for s in range(1, N+1):
    trace_vars = []
    for i in range(n):
      pos = Bool(f"L({s}, {lits[i]})")
      neg = Bool(f"L({s}, -{lits[i]})")
      trace_vars.append(pos)
      trace_vars.append(neg)
    # add and restrict to one trace var per lit
    optimizer.add(one_of(trace_vars))

  # subformula variables - A(s, s') and B(s, s'')
  for s in range(1, N+1):
    subform_vars_a = []
    subform_vars_b = []
    for s_p in range(1, N+1):
      if s + 1 <= s_p:
        a = Bool(f"A({s}, {s_p})")
        subform_vars_a.append(a)
      if s + 1 < s_p:
        b = Bool(f"B({s}, {s_p})")
        subform_vars_b.append(b)
    # each skeleton should have associated A if unary, an A and B if binary, and neither A or B if Lit
    if (not subform_vars_a == []):
      optimizer.add(Or(one_of(subform_vars_a), Bool(f"LIT({s})")))
    if (not subform_vars_b == []):
        # With or without metric operators
      if metric:
        optimizer.add(Or(one_of(subform_vars_b), Bool(f"LIT({s})"), Bool(f"NEXT({s})"), Bool(f"WNEXT({s})"), Bool(f"EVENTUALLY({s})"), Bool(f"ALWAYS({s})")))
      else:
        optimizer.add(Or(one_of(subform_vars_b), Bool(f"LIT({s})"), Bool(f"EVENTUALLY({s})"), Bool(f"ALWAYS({s})")))

  # enforce formula size, prevent reuse of skeletons
  for s, s_p, s_pp in product(range(N+1), repeat=3):
    if (not s == s_pp):
      optimizer.add(Not(And(Bool(f"A({s}, {s_p})"), Bool(f"A({s_pp}, {s_p})"))))
      optimizer.add(Not(And(Bool(f"B({s}, {s_p})"), Bool(f"B({s_pp}, {s_p})"))))
    optimizer.add(Not(And(Bool(f"A({s}, {s_p})"), Bool(f"B({s_pp}, {s_p})"))))

  # enforce s, s_p, s_pp relationships
  for s, s_p in product(range(1, N+1), repeat=2):
      if not s + 1 <= s_p:
        optimizer.add(Not(Bool(f"A({s}, {s_p})")))
      if not s + 1 < s_p:
        optimizer.add(Not(Bool(f"B({s}, {s_p})")))

  # no unary operators at final index
  if metric:
      optimizer.add(Not(Bool(f"NEXT({N})")))
      optimizer.add(Not(Bool(f"WNEXT({N})")))
  optimizer.add(Not(Bool(f"EVENTUALLY({N})")))
  optimizer.add(Not(Bool(f"ALWAYS({N})")))
  optimizer.add(Not(Bool(f"AND({N})")))
  optimizer.add(Not(Bool(f"OR({N})")))
  optimizer.add(Not(Bool(f"UNTIL({N})")))
  optimizer.add(Not(Bool(f"RELEASE({N})")))

  # no binary operators at second to final index
  optimizer.add(Not(Bool(f"AND({N-1})")))
  optimizer.add(Not(Bool(f"OR({N-1})")))
  optimizer.add(Not(Bool(f"UNTIL({N-1})")))
  optimizer.add(Not(Bool(f"RELEASE({N-1})")))

  skel_triple = [(s, s_p, s_pp) for (s, s_p, s_pp) in permutations(range(1, N+1), 3) if s_p > s and s_pp > s_p]
  skel_double = [(s, s_p) for (s, s_p) in permutations(range(1, N+1), 2) if s_p > s]
  skel_single = list(range(1, N+1))

  # no out of order subformula variables
  for (s, s_p, s_pp) in skel_triple:
    optimizer.add(Not(And(Bool(f"A({s}, {s_pp})"), Bool(f"B({s}, {s_p})"))))


  """
  ACCEPTANCE OF POSITVE EXAMPLES
  """

  # add RUN(e, 1, 1) for each example
  # soft constraint weight can be any value less than 1, we choose 0.5
  for e in range(len(pos_traces)):
    optimizer.add_soft(Bool(f"RUN({e}, 1, 1)"), 0.5)

  # AND
  for s, s_p, s_pp in skel_triple:
    for e in range(len(pos_traces)):
      for t in range(1, k+1):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"AND({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {t}, {s_p})")))
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"AND({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN({e}, {t}, {s_pp})")))

  # OR
  for s, s_p, s_pp in skel_triple:
    for e in range(len(pos_traces)):
      for t in range(1, k+1):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"OR({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN({e}, {t}, {s_p})"), Bool(f"RUN({e}, {t}, {s_pp})"))))

  if metric:
      # NEXT
      for s, s_p in skel_double:
        for e in range(len(pos_traces)):
          for t in range(1, k):
            optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"NEXT({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {t+1}, {s_p})")))
          optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"NEXT({s})"), Bool(f"A({s}, {s_p})")), False))

      # WNEXT
      for s, s_p in skel_double:
        for e in range(len(pos_traces)):
          for t in range(1, k):
            optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"WNEXT({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {t+1}, {s_p})")))
          optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"WNEXT({s})"), Bool(f"A({s}, {s_p})")), True)) # vacuously true

  # UNTIL
  for s, s_p, s_pp in skel_triple:
    for e in range(len(pos_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"UNTIL({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN({e}, {t}, {s_pp})"), And(Bool(f"RUN({e}, {t+1}, {s})"), Bool(f"RUN({e}, {t}, {s_p})")))))
      optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"UNTIL({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN({e}, {k}, {s_pp})")))

  # RELEASE
  for s, s_p, s_pp in skel_triple:
    for e in range(len(pos_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"RELEASE({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN({e}, {t}, {s_p})"), Bool(f"RUN({e}, {t+1}, {s})"))))
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"RELEASE({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN({e}, {t}, {s_pp})")))
      optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"RELEASE({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN({e}, {k}, {s_pp})")))

  # EVENTUALLY
  for s, s_p in skel_double:
    for e in range(len(pos_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"EVENTUALLY({s})"), Bool(f"A({s}, {s_p})")), Or(Bool(f"RUN({e}, {t}, {s_p})"), Bool(f"RUN({e}, {t+1}, {s})"))))
      optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"EVENTUALLY({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {k}, {s_p})")))

  # ALWAYS
  for s, s_p in skel_double:
    for e in range(len(pos_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"ALWAYS({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {t+1}, {s})")))
        optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"ALWAYS({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {t}, {s_p})")))
      optimizer.add(Implies(And(Bool(f"RUN({e}, {k}, {s})"), Bool(f"ALWAYS({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN({e}, {k}, {s_p})")))

  # LIT
  for s in skel_single:
    for e in range(len(pos_traces)):
      for t in range(1, k+1):
        for i in range(n):
          if pos_traces[e][t-1][i]:
            optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"LIT({s})"), Bool(f"L({s}, -{lits[i]})")), False))
          else:
            optimizer.add(Implies(And(Bool(f"RUN({e}, {t}, {s})"), Bool(f"LIT({s})"), Bool(f"L({s}, {lits[i]})")), False))

  """
  REJECTION OF NEGATIVE EXAMPLES
  """

  # add RUN_d(e, 1, 1) for each example
  # soft constraint weight can be any value less than 1, we choose 0.5
  for e in range(len(neg_traces)):
    optimizer.add_soft(Bool(f"RUN_d({e}, 1, 1)"), 0.5)

  # AND
  for s, s_p, s_pp in skel_triple:
    for e in range(len(neg_traces)):
      for t in range(1, k+1):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"AND({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN_d({e}, {t}, {s_p})"), Bool(f"RUN_d({e}, {t}, {s_pp})"))))

  # OR
  for s, s_p, s_pp in skel_triple:
    for e in range(len(neg_traces)):
      for t in range(1, k+1):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"OR({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {t}, {s_p})")))
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"OR({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN_d({e}, {t}, {s_pp})")))

  if metric:
      # NEXT
      for s, s_p in skel_double:
        for e in range(len(neg_traces)):
          for t in range(1, k):
            optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"NEXT({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {t+1}, {s_p})")))
          optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"NEXT({s})"), Bool(f"A({s}, {s_p})")), True)) # vacuously true

      # WNEXT
      for s, s_p in skel_double:
        for e in range(len(neg_traces)):
          for t in range(1, k):
            optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"WNEXT({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {t+1}, {s_p})")))
          optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"WNEXT({s})"), Bool(f"A({s}, {s_p})")), False))

  # UNTIL
  for s, s_p, s_pp in skel_triple:
    for e in range(len(neg_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"UNTIL({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN_d({e}, {t}, {s_p})"), Bool(f"RUN_d({e}, {t+1}, {s})"))))
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"UNTIL({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN_d({e}, {t}, {s_pp})")))
      optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"UNTIL({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN_d({e}, {k}, {s_pp})")))

  # RELEASE
  for s, s_p, s_pp in skel_triple:
    for e in range(len(neg_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"RELEASE({s})"), Bool(f"A({s}, {s_p})"), Bool(f"B({s}, {s_pp})")), Or(Bool(f"RUN_d({e}, {t}, {s_pp})"), And(Bool(f"RUN_d({e}, {t+1}, {s})"), Bool(f"RUN_d({e}, {t}, {s_p})")))))
      optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"RELEASE({s})"), Bool(f"B({s}, {s_pp})")), Bool(f"RUN_d({e}, {k}, {s_pp})")))

  # EVENTUALLY
  for s, s_p in skel_double:
    for e in range(len(neg_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"EVENTUALLY({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {t+1}, {s})")))
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"EVENTUALLY({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {t}, {s_p})")))
      optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"EVENTUALLY({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {k}, {s_p})")))

  # ALWAYS
  for s, s_p in skel_double:
    for e in range(len(neg_traces)):
      for t in range(1, k):
        optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"ALWAYS({s})"), Bool(f"A({s}, {s_p})")), Or(Bool(f"RUN_d({e}, {t}, {s_p})"), Bool(f"RUN_d({e}, {t+1}, {s})"))))
      optimizer.add(Implies(And(Bool(f"RUN_d({e}, {k}, {s})"), Bool(f"ALWAYS({s})"), Bool(f"A({s}, {s_p})")), Bool(f"RUN_d({e}, {k}, {s_p})")))

  # LIT
  for s in skel_single:
    for e in range(len(neg_traces)):
      for t in range(1, k+1):
        for i in range(n):
          # same as positive but the '-' is switched
          if neg_traces[e][t-1][i]:
            optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"LIT({s})"), Bool(f"L({s}, {lits[i]})")), False))
          else:
            optimizer.add(Implies(And(Bool(f"RUN_d({e}, {t}, {s})"), Bool(f"LIT({s})"), Bool(f"L({s}, -{lits[i]})")), False))

  return optimizer
