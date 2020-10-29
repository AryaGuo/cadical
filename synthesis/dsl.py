import shutil
import subprocess
import random
import sys
from copy import deepcopy
from pathlib import Path

from lark import Lark, Tree, Token

curPath = Path(__file__).resolve()
sys.path.append(str(curPath.parents[1]))

from synthesis.config import Config


class Scheme:
    """
    Describes a specific heuristic scheme.
    Can be described by dsl(str), a parse tree(lark.Tree) or a c++ code snippet.
    """

    def __init__(self, rule_dict, token_dict, dsl, tree):
        self.rule_dict = rule_dict
        self.token_dict = token_dict
        self.dsl = dsl
        self.tree = tree
        self.code = self.dsl_to_cpp(self.tree)
        self.solved, self.rtime, self.fitness = self.eval_fitness()

    def dsl_to_cpp_(self, parse_tree, term_stabs: dict):
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
                ret += self.dsl_to_cpp_(c, term_stabs)
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

    def dsl_to_cpp(self, parse_tree: Tree):
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
                code += self.dsl_to_cpp_(c, term_subs)
                # code += '\n}'
            elif c.data == 'assign_new_score':
                term_subs['SCORE'] = 'old_score'
                term_subs['NEW_SCORE'] = 'new_score'
                code += 'old_score = score(idx);\n'
                code += self.dsl_to_cpp_(c, term_subs)
            elif c.data == 'assign_score_inc':
                code += self.dsl_to_cpp_(c, term_subs)
        return code

    @staticmethod
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

    @staticmethod
    def compile_cadical():
        # todo
        shutil.move('analyze.cpp', '../src/analyze.cpp')
        subprocess.run('cd .. ; make', shell=True, check=True)  # ./configure && make

    def eval_fitness(self):
        # return 40, 1.0, 100
        self.embed_cadical(self.code)
        self.compile_cadical()

        subprocess.run('cd .. ; sh python/cadical.sh', shell=True, check=True, capture_output=True)
        process = subprocess.run('cd .. ; BACKUPDIR=$(ls -td output/*/ | head -1); DIRNAME=$(basename $BACKUPDIR);'
                                 'python python/gen_csv.py -S "cadical" -D ../Main-18/ -I $BACKUPDIR -O result/ -N $DIRNAME',
                                 shell=True, check=True, capture_output=True)
        out = process.stdout.decode()
        print(out)
        out = out.split()
        solved, rtime = int(out[0]), float(out[-1][:-1])
        return solved, rtime, solved ** 2 + 100 / rtime  # todo

    def update(self, new_tree):
        self.tree = new_tree
        self.code = self.dsl_to_cpp(self.tree)
        self.solved, self.rtime, self.fitness = self.eval_fitness()


class DSL:
    def __init__(self, meta_parser, grammar_file):
        with open(grammar_file) as grammar:
            self.parser = Lark(grammar)
        with open(grammar_file) as grammar:
            self.grammar_tree = meta_parser.parse(grammar.read())
            self.rule_dict, self.token_dict = self.__init_dicts(self.grammar_tree)
            # self.rule_dict: Dictionary of rules (rule's name->parse tree)
            # self.token_dict: Dictionary of tokens (token's name->parse tree)

    def gen_random_scheme(self, rule, depth):
        return self.get_scheme_from_dsl(self.__gen_random_dsl(self.rule_dict[rule], depth))

    def get_scheme_from_tree(self, tree):
        scheme = Scheme(self.rule_dict, self.token_dict, None, tree)
        return scheme

    def get_scheme_from_dsl(self, dsl):
        scheme = Scheme(self.rule_dict, self.token_dict, dsl, self.parser.parse(dsl))
        return scheme

    def __gen_random_dsl(self, parse_tree, depth: int):
        """
        Generate a random instance from the specified grammar rule.
        :param parse_tree: Grammar rule in lark.Tree or lark.Token format.
        :param depth: Recursive depth limits for subtrees.
        :return: A random string satisfying the grammar rule.
        """
        # print('[gen_random @ {}]'.format(depth), parse_tree)

        if depth < 0:
            return None

        if type(parse_tree) == Token:
            if parse_tree.type == 'TOKEN':
                return self.__gen_random_dsl(self.token_dict[parse_tree], depth)
            elif parse_tree.type == 'STRING':
                return parse_tree.strip('"')
            elif parse_tree.type == 'RULE':
                return self.__gen_random_dsl(self.rule_dict[parse_tree], depth)
            else:
                return parse_tree
                # raise Exception('Error: Unknown token type', parse_tree.type)

        assert type(parse_tree) == Tree, type(parse_tree)

        sample = ''

        if parse_tree.data == 'rule' or parse_tree.data == 'token':
            return self.__gen_random_dsl(parse_tree.children[-1], depth)
        elif parse_tree.data == 'expansion':
            for c in parse_tree.children:
                substr = self.__gen_random_dsl(c, depth - 1)
                if substr is None:
                    return None
                sample += substr
            return sample
        elif parse_tree.data == 'expansions':
            expansions = parse_tree.children
            for i in range(100):
                n = random.randrange(len(expansions))
                substr = self.__gen_random_dsl(expansions[n], depth - 1)
                if substr is not None:
                    return substr
            return None
        elif parse_tree.data == 'expr':
            OP = parse_tree.children[-1]
            assert OP.type == 'OP'
            if '?' in OP:
                coin = random.randint(0, 1)
                if coin:
                    return self.__gen_random_dsl(parse_tree.children[0], depth - 1)
                else:
                    return ''
        else:
            return self.__gen_random_dsl(parse_tree.children[0], depth)

    def __gen_random_tree(self, rule, depth, is_token=False):
        dsl_str = self.__gen_random_dsl(rule, depth)

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

    def mutate(self, parse_tree, rate):
        if random.random() > rate:
            return parse_tree
        if type(parse_tree) == Token:
            if parse_tree.type in self.token_dict and self.token_dict[parse_tree.type].data != 'literal':
                return self.__gen_random_tree(self.token_dict[parse_tree.type], 10, True)
            else:
                return parse_tree

        assert type(parse_tree) == Tree

        p_stop = 0.3  # todo
        if random.random() < p_stop and parse_tree.data != 'start':
            return self.__gen_random_tree(self.rule_dict[parse_tree.data.lstrip('?!')], 10)

        rand_child_index = random.randrange(len(parse_tree.children))
        parse_tree.children[rand_child_index] = self.mutate(parse_tree.children[rand_child_index], 1)
        return parse_tree

    @staticmethod
    def __match_expansions(tree_a, tree_b):
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

    @staticmethod
    def __match(tree, rule, is_token):
        # print('__match', tree, rule, is_token)
        if is_token:
            return type(tree) == Token and tree.type == rule
        else:
            return type(tree) == Tree and tree.data == rule

    @staticmethod
    def __match_node(tree_a, tree_b):
        if type(tree_a) == Token:
            return type(tree_b) == Token and tree_a.type == tree_b.type
        else:
            return type(tree_b) == Tree and tree_a.data == tree_b.data

    def crossover(self, tree_a, tree_b, depth):
        """
        Randomly substitute a branch of tree_a with a branch of tree_b.
        :param tree_a:
        :param tree_b:
        :param depth: Current depth in the tree. Used to calculate probability.
        :return: Tree_a' with new branch.
        """
        # print('crossover @ depth', depth, tree_a, '\n', tree_b)
        if type(tree_a) == Token:
            assert type(tree_b) == Token
            # p_stop = depth / (depth + 3)  # todo depth ~ p
            # if random.random() < p_stop:
            #     return tree_b
            # else:
            #     return tree_a
            return tree_b

        assert type(tree_a) == Tree, type(tree_a)
        assert type(tree_b) == Tree, type(tree_b)
        assert tree_a.data == tree_b.data, (tree_a.data, tree_b.data)

        p_stop = depth / (depth + 3)  # todo depth ~ p
        if random.random() < p_stop and tree_a.data != 'start':
            return tree_b

        rule = self.rule_dict[tree_a.data]
        subrule = rule.children[-1]
        # print('[subrule]', subrule)
        if subrule.data == 'expansions':
            # if not self.__match_expansions(tree_a, tree_b):
            #     return tree_b
            # matched = []
            # for i in range(len(tree_a.children)):
            #     subtree = tree_a.children[i]
            #     if type(subtree) == Token and subtree.type not in self.token_dict:
            #         continue
            #     matched.append((i, i))
            matched = []
            for i in range(len(tree_a.children)):
                for j in range(len(tree_b.children)):
                    if self.__match_node(tree_a.children[i], tree_b.children[j]):
                        matched.append((i, j))
            if len(matched) > 0:
                n = random.randrange(len(matched))
                tree_a.children[matched[n][0]] = self.crossover(tree_a.children[matched[n][0]],
                                                                tree_b.children[matched[n][1]], depth + 1)
                return tree_a
            else:
                return tree_b
        elif subrule.data == 'expansion':
            j, k = 0, 0  # tree_a, tree_b
            matched = []
            for i in range(len(subrule.children)):
                expr = subrule.children[i]
                if expr.data == 'expr':
                    # print(i, ' expr', expr)
                    OP = expr.children[1]
                    assert OP.type == 'OP'
                    name = expr.children[0]
                    is_token = name.children[0].type == 'TOKEN'
                    if '?' in OP:
                        ji = self.__match(tree_a.children[j], name.children[0], is_token)
                        ki = self.__match(tree_b.children[k], name.children[0], is_token)
                        if ji and ki:
                            matched.append((j, k))
                        j += ji
                        k += ki
                    else:
                        assert self.__match(tree_a.children[j], name.children[0], is_token)
                        assert self.__match(tree_b.children[k], name.children[0], is_token)
                        matched.append((j, k))
                        j, k = j + 1, k + 1
                else:
                    if expr.data != 'name':
                        j, k = j + 1, k + 1
                        continue
                    # print(i, 'name', expr)
                    is_token = expr.children[0].type == 'TOKEN'
                    assert self.__match(tree_a.children[j], expr.children[0], is_token)
                    assert self.__match(tree_b.children[k], expr.children[0], is_token)
                    matched.append((j, k))
                    j, k = j + 1, k + 1
                if j >= len(tree_a.children) or k >= len(tree_b.children):
                    break
            if len(matched) > 0:
                n = random.randrange(len(matched))
                tree_a.children[matched[n][0]] = self.crossover(tree_a.children[matched[n][0]],
                                                                tree_b.children[matched[n][1]], depth + 1)
                return tree_a
            else:
                return tree_b
        else:
            assert subrule.data == 'name'
            is_token = subrule.children[0].type == 'TOKEN'
            assert self.__match(tree_a.children[0], subrule.children[0], is_token)
            assert self.__match(tree_b.children[0], subrule.children[0], is_token)
            tree_a.children[0] = self.crossover(tree_a.children[0], tree_b.children[0], depth + 1)
            return tree_a

    @staticmethod
    def __init_dicts(tree):
        rule_dict = dict()
        token_dict = dict()

        for rule in tree.children:
            if rule.data == 'rule':
                key = rule.children[0].lstrip('?!')
                rule_dict[key] = rule
            elif rule.data == 'token':
                key = rule.children[0]
                token_dict[key] = rule.children[-1]
        for rule in tree.children:
            if rule.data == 'rule' and type(rule.children[2]) == Tree and rule.children[2].data == 'expansions':
                for c in rule.children[2].children:
                    if c.data == 'alias':
                        rule_dict[c.children[-1]] = c.children[0]
        return rule_dict, token_dict


class GP:
    def __init__(self, cfg):
        with open(cfg.meta_file) as meta_grammar:
            meta_parser = Lark(meta_grammar)
        self.dsl = DSL(meta_parser, cfg.grammar_file)
        self.population = []
        self.generation = 0
        self.pop_size = cfg.pop_size
        self.elitism = cfg.elitism
        self.depth_lim = cfg.depth_lim
        self.tournament_size = cfg.tournament_size
        self.mutation_rate = cfg.mutation_rate
        if self.tournament_size > self.pop_size:
            raise Exception('tournament_size larger than pop_size')

    def init_population(self):  # todo: depth ~ distribution / depth range
        assert self.generation == 0, self.generation
        self.generation = 1
        for i in range(self.pop_size):
            self.population.append(self.dsl.gen_random_scheme('heuristic', self.depth_lim))

    def evolve(self):
        new_population = []

        start_index = 0
        if self.elitism:
            start_index = 1
            # finding the most elite member of last generation and sending it directly to new generation
            new_population.append(deepcopy(max(self.population, key=lambda x: x.fitness)))

        for i in range(start_index, self.pop_size):
            parent_0, parent_1 = self.__tournament_selection()
            new_population.append(self.dsl.get_scheme_from_tree(self.__crossover(parent_0, parent_1)))
            self.__mutate(new_population[-1])

        self.population = new_population
        self.generation += 1

    def report(self, num):
        num = min(num, self.pop_size)
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        print('Generation', self.generation, ': Top', num, 'schemes\' score:',
              [self.population[i].fitness for i in range(num)])

    def get_winner(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return self.population[0]

    def __tournament_selection(self):
        rand_indices = random.sample(range(self.pop_size), self.tournament_size)
        temp_tournament = [self.population[rand_indices[i]] for i in range(self.tournament_size)]
        temp_tournament = sorted(temp_tournament, key=lambda x: x.fitness, reverse=True)
        return deepcopy(temp_tournament[0]), deepcopy(temp_tournament[1])

    def __mutate(self, scheme):
        scheme.update(self.dsl.mutate(scheme.tree), self.mutation_rate)

    def __crossover(self, scheme_0, scheme_1):
        return self.dsl.crossover(scheme_0.tree, scheme_1.tree, 1)

    def test_cross(self):
        ta = self.dsl.gen_random_scheme('heuristic', 15)
        tb = self.dsl.gen_random_scheme('heuristic', 15)
        self.__crossover(ta, tb)


def run_gp():
    gp = GP(Config)
    gp.init_population()
    for i in range(Config.epoch):
        gp.report(5)
        gp.evolve()

    winner = gp.get_winner()
    print(winner.code)
    print(winner.solved, winner.rtime, winner.fitness)


if __name__ == '__main__':
    random.seed(Config.seed)
    run_gp()
