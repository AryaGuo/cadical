import random
import sys
from pathlib import Path

curPath = Path(__file__).resolve()
sys.path.append(str(curPath.parents[1]))

from monkeys.typing import params, rtype, priority, constant, free
from monkeys.config import weight


@params(float, float)
@rtype(float)
@priority(weight['arithmetic'])
def addition_float(a, b):
    return '({} + {})'.format(a, b)


@params(int, int)
@rtype(int)
@priority(weight['arithmetic'])
def addition_int(a, b):
    return '({} + {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype('nonzero')
@priority(weight['arithmetic'])
def addition_nonzero(a, b):
    return '({} + {})'.format(a, b)


@params(float, float)
@rtype(float)
@priority(weight['arithmetic'])
def subtraction_float(a, b):
    return '({} - {})'.format(a, b)


@params(int, int)
@rtype(int)
@priority(weight['arithmetic'])
def subtraction_int(a, b):
    return '({} - {})'.format(a, b)


@params(float, float)
@rtype(float)
@priority(weight['arithmetic'])
def multiplication_float(a, b):
    return '({} * {})'.format(a, b)


@params(int, int)
@rtype(int)
@priority(weight['arithmetic'])
def multiplication_int(a, b):
    return '({} * {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype('nonzero')
@priority(weight['arithmetic'])
def multiplication_nonzero(a, b):
    return '({} * {})'.format(a, b)


@params(int, 'nonzero')
@rtype(int)
@priority(weight['arithmetic'])
def division_int(a, b):
    return '({} / {})'.format(a, b)


@params(float, 'nonzero')
@rtype(float)
@priority(weight['arithmetic'])
def division_float(a, b):
    return '({} / {})'.format(a, b)


@params(int, 'nonzero')
@rtype(int)
@priority(weight['arithmetic'])
def modulo(a, b):
    return '({} % {})'.format(a, b)


@params(int)
@rtype(float)
@priority(weight['conversion'])
def int_to_float(a):
    return 'double({})'.format(a)


@params('nonzero')
@rtype(float)
@priority(weight['conversion'])
def nonzero_to_float(a):
    return 'double({})'.format(a)


@params(int, int)
@rtype(bool)
@priority(weight['comparison'])
def equal_int(a, b):
    return '({} == {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype(bool)
@priority(weight['comparison'])
def equal_nonzero(a, b):
    return '({} == {})'.format(a, b)


@params(float, float)
@rtype(bool)
@priority(weight['comparison'])
def equal_float(a, b):
    return '(abs({} - {}) <= 0)'.format(a, b)
    # todo


@params(int, int)
@rtype(bool)
@priority(weight['comparison'])
def geq_int(a, b):
    return '({} >= {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype(bool)
@priority(weight['comparison'])
def geq_nonzero(a, b):
    return '({} >= {})'.format(a, b)


@params(float, float)
@rtype(bool)
@priority(weight['comparison'])
def geq_float(a, b):
    return '({} - {} >= 1e-10)'.format(a, b)
    # todo


@params(bool, float, float)
@rtype(float)
@priority(weight['condition'])
def condition_float(a, b, c):
    return '({} ? {} : {})'.format(a, b, c)


@params(bool, int, int)
@rtype(int)
@priority(weight['condition'])
def condition_int(a, b, c):
    return '({} ? {} : {})'.format(a, b, c)


@params(bool, 'nonzero', 'nonzero')
@rtype('nonzero')
@priority(weight['condition'])
def condition_nonzero(a, b, c):
    return '({} ? {} : {})'.format(a, b, c)


@params(float)
@rtype('assign_unbumped')
@priority(weight['special'])
def assign_unbumped(rhs):
    return """for (auto var : vars) {{
    double lhs = stab[var];
    stab[var] = {};
}}""".format(rhs)


@params(float)
@rtype('assign_new_score')
@priority(weight['special'])
def assign_new_score(rhs):
    return """double lhs = score(idx);
new_score = {};
""".format(rhs)


@params(float)
@rtype('assign_score_inc')
@priority(weight['special'])
def assign_score_inc(rhs):
    return """double lhs = score_inc;
new_score_inc = {};""".format(rhs)


@params('assign_unbumped', 'assign_new_score', 'assign_score_inc')
@rtype('heuristic')
@priority(weight['special'])
def heuristic(a, b, c):
    return [a, b, c]


constant(int, str(0), weight['constant'])
for i in range(1, 10):
    constant('nonzero', str(i), weight['constant'])
for i in range(1, 11):
    constant('nonzero', str(2 ** i), weight['constant'])
for i in range(1, 10):
    constant(float, str(i / 10), weight['constant'])
constant('nonzero', 'stats.conflicts', weight['i'])
constant(float, 'lhs', weight['lhs'])
constant(float, 'score_inc', weight['inc'])

free(int, 'nonzero', weight['conversion'])


def selection_strategy(parent, children):
    ret = []
    for child_list in children:
        weights = [child.priority for child in child_list]
        ret.append(random.choices(child_list, weights)[0])
    return ret
