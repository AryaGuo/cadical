import shutil
import subprocess
from random import randint, random
from pathlib import Path
from lark import Lark, Tree, Token


def gen_random_dsl(parse_tree, rule_dict: dict, token_dict: dict, depth: int):
    """
    Generate a random instance from the specified grammar rule.
    :param parse_tree: Grammar rule in lark.Tree or lark.Token format.
    :param rule_dict: Dictionary of rules (rule's name->parse tree)
    :param token_dict: Dictionary of tokens (token's name->parse tree)
    :param depth: Recursive depth limits for subtrees.
    :return: A random string satisfying the grammar rule.
    """
    # print('[gen_random @ {}]'.format(depth), parse_tree)

    if depth < 0:
        return None

    if type(parse_tree) == Token:
        if parse_tree.type == 'TOKEN':
            return gen_random_dsl(token_dict[parse_tree], rule_dict, token_dict, depth)
        elif parse_tree.type == 'STRING':
            return parse_tree.strip('"')
        elif parse_tree.type == 'RULE':
            return gen_random_dsl(rule_dict[parse_tree], rule_dict, token_dict, depth)
        else:
            return parse_tree
            # raise Exception('Error: Unknown token type', parse_tree.type)

    assert type(parse_tree) == Tree, type(parse_tree)

    sample = ''

    if parse_tree.data == 'rule' or parse_tree.data == 'token':
        return gen_random_dsl(parse_tree.children[-1], rule_dict, token_dict, depth)
    elif parse_tree.data == 'expansion':
        for c in parse_tree.children:
            substr = gen_random_dsl(c, rule_dict, token_dict, depth - 1)
            if substr is None:
                return None
            sample += substr
        return sample
    elif parse_tree.data == 'expansions':
        expansions = parse_tree.children
        for i in range(100):
            n = randint(0, len(expansions) - 1)
            substr = gen_random_dsl(expansions[n], rule_dict, token_dict, depth - 1)
            if substr is not None:
                return substr
        return None
    elif parse_tree.data == 'expr':
        OP = parse_tree.children[-1]
        assert OP.type == 'OP'
        if '?' in OP:
            coin = randint(0, 1)
            if coin:
                return gen_random_dsl(parse_tree.children[0], rule_dict, token_dict, depth - 1)
            else:
                return ''
    else:
        return gen_random_dsl(parse_tree.children[0], rule_dict, token_dict, depth)


def gen_random_tree_(parse_tree, rule_dict: dict, token_dict: dict, depth: int):
    """
    Generate a random instance from the specified grammar rule.
    :param parse_tree: Grammar rule in lark.Tree or lark.Token format.
    :param rule_dict: Dictionary of rules (rule's name->parse tree)
    :param token_dict: Dictionary of tokens (token's name->parse tree)
    :param depth: Recursive depth limits for subtrees.
    :return: A random string satisfying the grammar rule.
    """
    # print('[gen_random @ {}]'.format(depth), parse_tree)

    if depth < 0:
        return None

    if type(parse_tree) == Token:
        if parse_tree.type == 'TOKEN':
            return gen_random_tree_(token_dict[parse_tree], rule_dict, token_dict, depth)
        elif parse_tree.type == 'STRING':
            return Token('STRING', parse_tree.strip('"'))
        elif parse_tree.type == 'RULE':
            return gen_random_tree_(rule_dict[parse_tree], rule_dict, token_dict, depth)
        elif parse_tree.type in token_dict:
            return gen_random_tree_(token_dict[parse_tree.type], rule_dict, token_dict, depth)
        else:
            return parse_tree

    assert type(parse_tree) == Tree, type(parse_tree)
    # assert parse_tree.data == 'rule' or parse_tree.data == 'token', parse_tree.data

    if parse_tree.data == 'rule' or parse_tree.data == 'token':
        return gen_random_tree_(parse_tree.children[-1], rule_dict, token_dict, depth)
    elif parse_tree.data == 'expansion':
        ret = Tree(parse_tree.data, [])
        for c in parse_tree.children:
            substr = gen_random_tree_(c, rule_dict, token_dict, depth - 1)
            if substr is None:
                return None
            ret.children.append(substr)
        return ret
    elif parse_tree.data == 'expansions':
        expansions = parse_tree.children
        for i in range(100):
            n = randint(0, len(expansions) - 1)
            substr = gen_random_tree_(expansions[n], rule_dict, token_dict, depth - 1)
            if substr is not None:
                return substr
        return None
    elif parse_tree.data == 'expr':
        OP = parse_tree.children[-1]
        assert OP.type == 'OP'
        if '?' in OP:
            coin = randint(0, 1)
            if coin:
                return gen_random_tree_(parse_tree.children[0], rule_dict, token_dict, depth - 1)
            else:
                return None
    else:
        return gen_random_tree_(parse_tree.children[0], rule_dict, token_dict, depth)


def gen_random_tree(rule, rule_dict, token_dict, depth, is_token=False):
    dsl_str = gen_random_dsl(rule, rule_dict, token_dict, depth)

    with open("expr.bnf") as grammar:
        parser = Lark(grammar)
        if is_token:
            ret = parser.parse(dsl_str).children[0]
            # print('GEN', 'TOKEN', ret)
        else:
            name = rule.children[0].lstrip('!?')
            ret = parser.parse(dsl_str, name)
            # print('GEN', name, ret)
        return ret


def dsl_to_cpp_(parse_tree, term_stabs: dict):
    if type(parse_tree) == Tree:
        ret = ''
        first = True
        if_body = False
        if 'pow' in parse_tree.data:
            ret += 'pow('
        for c in parse_tree.children:
            if c == 'new_score' and parse_tree.data == 'assign_unbumped':
                ret += 'for (auto var : vars) {\n\t'
                first = True
            if if_body and first:
                ret += '\t'
            if not first:
                ret += ' '
            first = False
            ret += dsl_to_cpp_(c, term_stabs)
            if type(c) == Tree and c.data == 'condition':
                if_body = True
                ret += ' {\n\t'
        if 'pow' in parse_tree.data:
            ret += ')'
        if parse_tree.data == 'assign_unbumped':
            ret += '\n\t}' if if_body else '\n}'
        if if_body:
            ret += '\n}'
        return ret
    elif type(parse_tree) == Token:
        # print('type', parse_tree.type, parse_tree)
        if parse_tree.type in term_stabs:
            return term_stabs[parse_tree.type]
        else:
            return parse_tree


def dsl_to_cpp(parse_tree: Tree):
    assert type(parse_tree) == Tree, type(parse_tree)
    assert parse_tree.data == 'start', parse_tree.data

    parse_tree = parse_tree.children[0]  # todo

    code = ''
    first = True
    term_subs = {'CONFLICT_INDEX': 'stats.conflicts', 'CIRCUMFLEX': ','}
    for c in parse_tree.children:
        if not first:
            code += '\n'
        first = False
        if c.data == 'assign_unbumped':
            # code += 'for (auto var : vars) {\n\t'
            term_subs['SCORE'] = 'stab[var]'
            term_subs['NEW_SCORE'] = 'stab[var]'
            code += dsl_to_cpp_(c, term_subs)
            # code += '\n}'
        elif c.data == 'assign_new_score':
            term_subs['SCORE'] = 'old_score'
            term_subs['NEW_SCORE'] = 'new_score'
            code += 'old_score = score(idx);\n'
            code += dsl_to_cpp_(c, term_subs)
        elif c.data == 'assign_score_inc':
            code += dsl_to_cpp_(c, term_subs)
    return code


def embed_cadical(code_snippet):
    code_snippet = code_snippet.splitlines()
    for i in range(len(code_snippet)):
        code_snippet[i] = '\t\t' + code_snippet[i]
    code_snippet = '\n'.join(code_snippet)

    SPLIT = 92
    with open('analyze_blank.cpp', 'r') as cpp_file:
        code = cpp_file.readlines()
    code.insert(SPLIT, code_snippet)
    with open('analyze.cpp', 'w') as cpp_file:
        cpp_file.writelines(code)


def compile_cadical():
    shutil.move('analyze.cpp', '../src/analyze.cpp')
    subprocess.run('cd .. ; make', shell=True, check=True)  # ./configure && make


def eval_fitness(parse_tree):
    code = dsl_to_cpp(parse_tree)
    embed_cadical(code)
    compile_cadical()

    subprocess.run('cd .. ; sh python/cadical.sh', shell=True, check=True)
    process = subprocess.run('cd .. ; BACKUPDIR=$(ls -td output/*/ | head -1); DIRNAME=$(basename $BACKUPDIR);'
                             'python python/gen_csv.py -S "cadical" -D ../Main-18/ -I $BACKUPDIR -O result/ -N $DIRNAME',
                             shell=True, check=True, capture_output=True)
    out = process.stdout.decode()
    print(out)
    out = out.split()
    solved, rtime = int(out[0]), float(out[-1][:-1])
    return solved ** 2 + 100 / rtime


def mutation(parse_tree, rule_dict, token_dict):
    if type(parse_tree) == Token:
        if parse_tree.type in token_dict and token_dict[parse_tree.type].data != 'literal':
            return gen_random_tree(token_dict[parse_tree.type], rule_dict, token_dict, 10, True)
        else:
            return parse_tree

    assert type(parse_tree) == Tree
    p_stop = 0.4  # todo

    if random() < p_stop and parse_tree.data != 'start':
        return gen_random_tree(rule_dict[parse_tree.data.lstrip('?!')], rule_dict, token_dict, 10)

    rand_child_index = randint(0, len(parse_tree.children) - 1)
    parse_tree.children[rand_child_index] = mutation(parse_tree.children[rand_child_index], rule_dict, token_dict)
    return parse_tree


def match_children(tree_a, tree_b):
    if len(tree_a.children) != len(tree_b.children):
        return False
    for i in range(len(tree_a.children)):
        if type(tree_a.children[i]) == Tree:
            if type(tree_a.children[i]) != Tree or tree_a.children[i].data != tree_b.children[i].data:
                return False
        else:
            if type(tree_a.children[i]) != Token or type(tree_b.children[i]) != Token:
                return False
    return True


def crossover(tree_a, tree_b, rule_dict, depth):
    assert type(tree_a) == Tree, type(tree_a)
    assert type(tree_b) == Tree, type(tree_b)
    print(tree_a.data, tree_b.data)

    p_stop = depth / (depth + 3)  # depth ~ p
    if random() < p_stop and tree_a.data != 'start':
        return tree_b, tree_a

    rule = rule_dict[tree_a.data]
    subrule = rule.children[-1]
    if type(subrule) == Token:
        return tree_b, tree_a
    elif subrule.data == 'expansions' and not match_children(tree_a, tree_b):
        return tree_b, tree_a
    else:
        # print(subrule.data)
        assert match_children(tree_a, tree_b)
        n = randint(0, len(tree_a.children) - 1)
        tree_a.children[n], tree_b.children[n] = crossover(tree_a.children[n], tree_b.children[n], rule_dict, depth + 1)
        return tree_a, tree_b


def init_dicts(tree):
    rule_dict = dict()
    token_dict = dict()

    for rule in tree.children:
        if rule.data == 'rule':
            key = rule.children[0].lstrip('?!')
            rule_dict[key] = rule
        elif rule.data == 'token':
            key = rule.children[0]
            # print(key, rule.children[-1])
            token_dict[key] = rule.children[-1]
    for rule in tree.children:
        if rule.data == 'rule' and type(rule.children[2]) == Tree and rule.children[2].data == 'expansions':
            for c in rule.children[2].children:
                if c.data == 'alias':
                    # key = c.children[-1]
                    # val = c.children[0]
                    # print('add_dict', key, val)
                    rule_dict[c.children[-1]] = c.children[0]

    return rule_dict, token_dict


if __name__ == '__main__':
    # from synthesis.schemes import vsids
    #
    # with open("expr.bnf") as grammar:
    #     real_parser = Lark(grammar)
    #     vtree = real_parser.parse(vsids)
    # print(vtree)

    with open("bnf.bnf") as meta_grammar:
        meta_parser = Lark(meta_grammar)
    with open("expr.bnf") as grammar:
        grammar_tree = meta_parser.parse(grammar.read())
        rule_dict, token_dict = init_dicts(grammar_tree)

        print('--- start ---')
        gen_str = gen_random_dsl(rule_dict['heuristic'], rule_dict, token_dict, 20)
        print(gen_str)
        print('--- end ---')

    with open("expr.bnf") as grammar:
        real_parser = Lark(grammar)
        tree = real_parser.parse(gen_str)
        # print(tree)
        code = dsl_to_cpp(tree)
        print('\n--- cpp code ---')
        print(code)
        print('--- end ---')
        # eval_fitness(tree)
        # exit(0)

        mut = mutation(tree, rule_dict, token_dict)
        # print('\n--- mut ---')
        # print(mut)
        # print('--- end ---')

        ca, cb = crossover(tree, mut, rule_dict, 1)
        print('\n--- crossed cpp code ---')
        ca_code = dsl_to_cpp(ca)
        print(ca_code)
        print('--- end ---')
        exit(0)

        code = dsl_to_cpp(tree)
        print('\n--- mutated cpp code ---')
        print(code)
        print('--- end ---')
        exit(0)
