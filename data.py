from flloat.parser.ltlf import LTLfParser
from itertools import product
from collections import deque
import math
import random
from copy import deepcopy

def lexographic(x):
  value = 0
  for e in x:
    if x[e]:
        value += ord(e)
  return value

def generate_edges(lits):
    prod = product([True, False], repeat=len(lits))
    edges = [{l: p for l, p in zip(lits, pr)} for pr in prod]
    return edges

def generate_cs(formula_string, lits, trace_length):
  """
  algorithm from https://arxiv.org/pdf/1605.07805.pdf (section 4) or
  https://faculty.ist.psu.edu/vhonavar/Papers/parekh-dfa.pdf
  """
  parser = LTLfParser()
  formula = parser(formula_string)
  dfa = formula.to_automaton().minimize().complete()
  pos = []
  neg = []
  edges = generate_edges(lits)
  edges.sort(key=lexographic)

  sink_states = set()
  for state in dfa.states:
      is_sink = True
      for edge in edges:
        if dfa.get_successor(state, edge) != state:
            is_sink = False
            break
      if is_sink:
          sink_states.add(state)

  # 1) bfs to find shortest path to each node, S_p(L)
  parent = dict()
  visited = set()
  queue = deque([dfa.initial_state])
  while len(queue) is not 0:
    state = queue.popleft()
    for edge in edges:
      if dfa.get_successor(state, edge) not in visited:
        visited.add(dfa.get_successor(state, edge))
        queue.append(dfa.get_successor(state, edge))
        parent[dfa.get_successor(state, edge)] = (state, edge)
  shortest_paths = []
  for state in dfa.states:
    if state in parent and state != dfa.initial_state:
      path = []
      par, inc_edge = parent[state]
      path.append(inc_edge)
      while par is not dfa.initial_state:
        par, inc_edge = parent[par]
        path.append(inc_edge)
      path.reverse()
      shortest_paths.append((state, path))

  # add the empty path to S_p(L) since it is the shortest path to initial state
  shortest_paths.append((dfa.initial_state, []))

  # 2) extend each shortest path with every possible edge, N(L)
  extended_paths = []
  for state, short_path in shortest_paths:
    for edge in edges:
      extended_paths.append((dfa.get_successor(state, edge), short_path + [edge]))

  # add the empty path to N(L)
  extended_paths.append((dfa.initial_state, []))

  # 3) finish the extended paths using bfs to get to a sink, add these to the corresponding set
  for ext_state, ext_path in extended_paths:
    if ext_state in dfa.accepting_states:
      pos.append(ext_path)
    elif ext_state in sink_states:
      neg.append(ext_path)
    else:
      parent = dict()
      visited = set()
      queue = deque([ext_state])
      acc_state = None
      while len(queue) is not 0:
        state = queue.popleft()
        for edge in edges:
          if dfa.get_successor(state, edge) not in visited:
            visited.add(dfa.get_successor(state, edge))
            queue.append(dfa.get_successor(state, edge))
            parent[dfa.get_successor(state, edge)] = (state, edge)
          if dfa.get_successor(state, edge) in dfa.accepting_states:
            acc_state = dfa.get_successor(state, edge)
            break
      path = []
      par, inc_edge = parent[acc_state]
      path.append(inc_edge)
      while par is not ext_state:
        par, inc_edge = parent[par]
        path.append(inc_edge)
      path.reverse()
      pos.append(ext_path + path)

  # 4) find shortest distinguishing suffix for each pair of strings, one from S_p(q) and one from N(L)
  for short_state, short_path in shortest_paths:
    for ext_state, ext_path in extended_paths:
      if short_state is not ext_state:
        done = False
        # first try appending the empty string
        if short_state in dfa.accepting_states and ext_state not in dfa.accepting_states:
            pos.append(short_path)
            neg.append(ext_path)
            done = True
        elif ext_state in dfa.accepting_states and short_state not in dfa.accepting_states:
            pos.append(ext_path)
            neg.append(short_path)
            done = True

        length = 1
        while not done:
          suffixes = [list(prod) for prod in product(edges, repeat=length)]
          len_suff = len(suffixes)
          for k, s in enumerate(suffixes):
            trace_short = short_path + s
            trace_ext = ext_path + s
            if formula.truth(trace_short, 0) != formula.truth(trace_ext, 0):
              if formula.truth(trace_short, 0):
                pos.append(short_path + s)
                neg.append(ext_path + s)
                done = True
                break
              if formula.truth(trace_ext, 0):
                pos.append(ext_path + s)
                neg.append(short_path + s)
                done = True
                break
          length += 1

  # remove duplicates
  pos = set(list(map(lambda x: tuple([tuple([t[l] for l in lits]) for t in x]), pos)))
  neg = set(list(map(lambda x: tuple([tuple([t[l] for l in lits]) for t in x]), neg)))
  pos = [list(map(list, example)) for example in pos]
  neg = [list(map(list, example)) for example in neg]

  # remove empty path
  if [] in pos:
      pos.remove([])
  if [] in neg:
      neg.remove([])

  # failed to generate cs (can happen if formula reduces to true/false)
  if len(pos) == 0 or len(neg) == 0:
      return None

  new_pos = []
  new_neg = []
  for e in pos:
    # pad end of trace
    pad_amount = trace_length - len(e)
    padded = e + [e[-1]]*pad_amount
    new_pos.append(padded)
  for e in neg:
    # pad end of trace
    pad_amount = trace_length - len(e)
    padded = e + [e[-1]]*pad_amount
    new_neg.append(padded)

  return new_pos, new_neg

def rejection_sample(formula_string, lits, trace_length, num_pos, num_neg):
    parser = LTLfParser()
    formula = parser(formula_string)
    pos = []
    neg = []

    def perturb_trace(trace):
        perturbed_trace = deepcopy(trace)
        step = random.choice(range(trace_length))
        l = random.choice(lits)
        perturbed_trace[step][l] = not perturbed_trace[step][l]
        return perturbed_trace

    counter = 0
    while (len(pos) < num_pos or len(neg) < num_neg):
      trace = []
      for i in range(trace_length):
        timestep = {}
        for l in lits:
          if random.getrandbits(1):
            timestep[l] = True
          else:
            timestep[l] = False
        trace.append(timestep)
      if formula.truth(trace, 0):
        if len(pos) < num_pos:
          pos.append(trace)
      else:
        if len(neg) < num_neg:
          neg.append(trace)
      counter += 1
      if counter > 250000:
          sub_counter = 0
          print("Trying perturbed traces")
          if len(pos) == 0 or len(neg) == 0:
              print("Giving up - empty trace set")
              return None
          while (len(pos) < num_pos or len(neg) < num_neg):
              pos_copy = deepcopy(pos)
              neg_copy = deepcopy(neg)
              if len(pos) < num_pos:
                  perturbed_trace = perturb_trace(random.choice(pos_copy))
                  if formula.truth(perturbed_trace, 0):
                    if len(pos) < num_pos:
                      pos.append(perturbed_trace)
                  else:
                    if len(neg) < num_neg:
                      neg.append(perturbed_trace)
              if len(neg) < num_neg:
                  perturbed_trace = perturb_trace(random.choice(neg_copy))
                  if formula.truth(perturbed_trace, 0):
                    if len(pos) < num_pos:
                      pos.append(perturbed_trace)
                  else:
                    if len(neg) < num_neg:
                      neg.append(perturbed_trace)
              sub_counter += 1
              if sub_counter > 250000:
                  print("Giving up - perturb failed")
                  return None
          break

    pos = [[[t[l] for l in lits] for t in trace] for trace in pos]
    neg = [[[t[l] for l in lits] for t in trace] for trace in neg]

    return pos, neg
