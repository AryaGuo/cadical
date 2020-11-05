import sys
from pathlib import Path

curPath = Path(__file__).resolve()
sys.path.append(str(curPath.parents[1]))

from monkeys.typing import params, rtype, constant, free


@params(float, float)
@rtype(float)
def addition_float(a, b):
    return '({} + {})'.format(a, b)


@params(int, int)
@rtype(int)
def addition_int(a, b):
    return '({} + {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype('nonzero')
def addition_nonzero(a, b):
    return '({} + {})'.format(a, b)


@params(float, float)
@rtype(float)
def subtraction_float(a, b):
    return '({} - {})'.format(a, b)


@params(int, int)
@rtype(int)
def subtraction_int(a, b):
    return '({} - {})'.format(a, b)


@params(float, float)
@rtype(float)
def multiplication_float(a, b):
    return '({} * {})'.format(a, b)


@params(int, int)
@rtype(int)
def multiplication_int(a, b):
    return '({} * {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype('nonzero')
def multiplication_nonzero(a, b):
    return '({} * {})'.format(a, b)


@params(int, 'nonzero')
@rtype(int)
def division_int(a, b):
    return '({} / {})'.format(a, b)


@params(float, 'nonzero')
@rtype(float)
def division_float(a, b):
    return '({} / {})'.format(a, b)


@params(int, 'nonzero')
@rtype(int)
def modulo(a, b):
    return '({} % {})'.format(a, b)


@params(int)
@rtype(float)
def int_to_float(a):
    return 'double({})'.format(a)


@params('nonzero')
@rtype(float)
def nonzero_to_float(a):
    return 'double({})'.format(a)


@params(int, int)
@rtype(bool)
def equal_int(a, b):
    return '({} == {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype(bool)
def equal_nonzero(a, b):
    return '({} == {})'.format(a, b)


@params(float, float)
@rtype(bool)
def equal_float(a, b):
    return '(abs({} - {}) <= 0)'.format(a, b)
    # todo


@params(int, int)
@rtype(bool)
def geq_int(a, b):
    return '({} >= {})'.format(a, b)


@params('nonzero', 'nonzero')
@rtype(bool)
def geq_nonzero(a, b):
    return '({} >= {})'.format(a, b)


@params(float, float)
@rtype(bool)
def geq_float(a, b):
    return '({} - {} >= 1e-10)'.format(a, b)
    # todo


@params(bool, float, float)
@rtype(float)
def condition_float(a, b, c):
    return '{} ? {} : {}'.format(a, b, c)


@params(bool, int, int)
@rtype(int)
def condition_int(a, b, c):
    return '{} ? {} : {}'.format(a, b, c)


@params(bool, 'nonzero', 'nonzero')
@rtype('nonzero')
def condition_nonzero(a, b, c):
    return '{} ? {} : {}'.format(a, b, c)


@params(float)
@rtype('assign_unbumped')
def assign_unbumped(rhs):
    return """for (auto var : vars) {{
    double lhs = stab[var];
    stab[var] = {};
}}""".format(rhs)


@params(float)
@rtype('assign_new_score')
def assign_new_score(rhs):
    return """double lhs = score(idx);
new_score = {};
""".format(rhs)


# @params()
# @rtype('assign_new_score')
# def assign_new_score(rhs):
#     return """double lhs = score(idx);
# new_score = score_inc;
# """.format(rhs)


@params(float)
@rtype('assign_score_inc')
def assign_score_inc(rhs):
    return """double lhs = score_inc;
score_inc = {};""".format(rhs)


@params('assign_unbumped', 'assign_new_score', 'assign_score_inc')
@rtype('heuristic')
def heuristic(a, b, c):
    return [a, b, c]


constant(int, str(0))
for i in range(1, 10):
    constant('nonzero', str(i))
for i in range(1, 11):
    constant('nonzero', str(2 ** i))
for i in range(1, 10):
    constant(float, str(i / 10))
constant('nonzero', 'stats.conflicts')
constant(float, 'lhs')
constant(float, 'score_inc')

free(int, 'nonzero')
