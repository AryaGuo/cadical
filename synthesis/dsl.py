import argparse
import functools
import logging
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from lark import Lark, Tree, Token

curPath = Path(__file__).resolve()
sys.path.append(str(curPath.parents[1]))

from synthesis.config import Config

from monkeys import optimize, tournament_select, next_generation
from monkeys.typing import params
from monkeys.search import require
from monkey import grammar


class Node:
    def __init__(self, type, data=None, parent=None, children=None):
        self.type = type  # name of TERM/RULE
        self.data = data  # terminal string/None for rule
        self.is_term = self.data is not None
        self.parent = parent
        self.depth = 1
        self.size = 1
        if children:
            self.children = children
        else:
            self.children = []
        for c in self.children:
            self.depth = max(self.depth, c.depth + 1)
            self.size += c.size


class Scheme:
    """
    Describes a specific heuristic scheme.
    Can be described by dsl(str), a parse tree(lark.Tree) or a c++ code snippet.
    """

    def __init__(self, rule_dict, token_dict, dsl, tree, name=None):
        self.rule_dict = rule_dict
        self.token_dict = token_dict
        self.dsl = dsl
        self.tree = tree
        self.code = self.dsl_to_cpp(self.tree)
        self.solved, self.rtime, self.fitness, self.file = self.eval_fitness
        if name is not None and self.file is not None:
            dst = self.file.parent / (name + '.csv')
            shutil.move(self.file, dst)
            self.file = dst

    def display(self):
        logging.info('-----')
        for code in self.code:
            logging.info(code)
        logging.info('-----')

    def dsl_to_cpp_(self, parse_tree, term_stabs: dict):
        if type(parse_tree) == Tree:
            ret = ''
            first = True
            if_body = False
            for c in parse_tree.children:
                if c == 'unbumped' and parse_tree.data == 'assign_unbumped':
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
        assert type(parse_tree) == Tree, parse_tree
        assert parse_tree.data == 'start', parse_tree

        parse_tree = parse_tree.children[0]  # todo

        codes = ['', '', '']
        term_subs = {'CONFLICT_INDEX': 'stats.conflicts', 'SCORE_INC': 'score_inc', 'UNBUMPED': 'stab[var]',
                     'BUMPED': 'new_score', 'NEW_SCORE_INC': 'new_score_inc'}
        for c in parse_tree.children:
            code = ''
            if c.data == 'assign_unbumped':
                term_subs['LHS'] = 'stab[var]'
                code += self.dsl_to_cpp_(c, term_subs)
                codes[0] = code
            elif c.data == 'assign_new_score':
                term_subs['LHS'] = 'old_score'
                code += self.dsl_to_cpp_(c, term_subs)
                codes[1] = code
            elif c.data == 'assign_score_inc':
                term_subs['LHS'] = 'score_inc'
                code += self.dsl_to_cpp_(c, term_subs)
                codes[2] = code
        return codes

    @staticmethod
    def embed_cadical(codes):
        SPLIT = [[182], [92, 99], [124, 131]]
        with open('analyze_blank.cpp', 'r') as cpp_file:
            code = cpp_file.readlines()
        for j in range(3):
            code_snippet = codes[j].splitlines()
            for i in range(len(code_snippet)):
                code_snippet[i] = '\t\t' + code_snippet[i]
            code_snippet = '\n'.join(code_snippet)
            for line in SPLIT[j]:
                code.insert(line, code_snippet)
        with open('analyze.cpp', 'w') as cpp_file:
            cpp_file.writelines(code)

    @staticmethod
    def compile_cadical():
        shutil.move('analyze.cpp', '../src/analyze.cpp')
        try:
            subprocess.run('cd .. ; make', shell=True, check=True, capture_output=True)  # ./configure && make
        except subprocess.CalledProcessError:
            return False
        return True

    @property
    def eval_fitness(self):
        # return 40, 1.0, 100, None
        try:
            self.embed_cadical(self.code)
            if not self.compile_cadical():
                return 0, 0, 0, None  # Compilation error
            subprocess.run('cd .. ; sh python/cadical.sh', shell=True, check=True, capture_output=True)
            get_name = subprocess.run('basename $(ls -td ../output/*/ | head -1)', shell=True, check=True,
                                      capture_output=True)
            basename = get_name.stdout.decode().strip()
            process = subprocess.run('BACKUPDIR=$(ls -td ../output/*/ | head -1); DIRNAME=$(basename $BACKUPDIR);'
                                     'python ../python/gen_csv.py -S "cadical" -D ~/Main-18/ -I $BACKUPDIR -O {} -N '
                                     '$DIRNAME'.format(output_dir),
                                     shell=True, check=True, capture_output=True)
            out = process.stdout.decode()
            logging.info(out)
            out = out.split()
            solved, rtime = int(out[0]), float(out[-1][:-1])
            csvfile = output_dir / (basename + '.csv')
            self.display()
            return solved, rtime, 30 * solved ** 2 + 60 / rtime, csvfile  # todo
        except subprocess.CalledProcessError as err:
            logging.error(err)
            return 0, 0, -sys.maxsize, None

    def update(self, new_tree):
        self.tree = new_tree
        self.code = self.dsl_to_cpp(self.tree)
        self.solved, self.rtime, self.fitness, self.file = self.eval_fitness


class DSL:
    def __init__(self, meta_parser, grammar_file, wt, OP_prob, gen_restart):
        self.grammar_file = grammar_file
        self.wt = wt
        self.OP_prob = OP_prob
        self.gen_restart = gen_restart
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
        scheme = Scheme(self.rule_dict, self.token_dict, None, tree, None)
        return scheme

    def get_scheme_from_dsl(self, dsl, name=None):
        scheme = Scheme(self.rule_dict, self.token_dict, dsl, self.parser.parse(dsl), name)
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

        assert type(parse_tree) == Tree, parse_tree

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
            wt_lst = []
            for c in expansions:
                cur = 1
                for token in c.children:
                    if type(token) == Token and token in self.wt:
                        cur += self.wt[token]
                wt_lst.append(cur)
            for i in range(self.gen_restart):
                # n = random.randrange(len(expansions))
                n = random.choices(range(len(expansions)), weights=wt_lst)[0]
                substr = self.__gen_random_dsl(expansions[n], depth - 1)
                if substr is not None:
                    return substr
            return None
        elif parse_tree.data == 'expr':
            OP = parse_tree.children[-1]
            assert OP.type == 'OP', OP
            if '?' in OP:
                coin = random.random()
                if coin < self.OP_prob:
                    return self.__gen_random_dsl(parse_tree.children[0], depth - 1)
                else:
                    return ''
        else:
            return self.__gen_random_dsl(parse_tree.children[0], depth)

    def __gen_random_tree(self, rule, depth, is_token=False):
        dsl_str = self.__gen_random_dsl(rule, depth)

        with open(self.grammar_file) as grammar:
            parser = Lark(grammar)
            if is_token:
                ret = parser.parse(dsl_str).children[0]
                # print('GEN', 'TOKEN', ret)
            else:
                name = rule.children[0].lstrip('!?')
                ret = parser.parse(dsl_str, name)
                # print('GEN', name, ret)
            return ret

    def mutate(self, parse_tree, rate, depth):
        if random.random() > rate:
            return parse_tree
        if type(parse_tree) == Token:
            if parse_tree.type in self.token_dict and self.token_dict[parse_tree.type].data != 'literal':
                return self.__gen_random_tree(self.token_dict[parse_tree.type], 10, True)
            else:
                return parse_tree

        assert type(parse_tree) == Tree, parse_tree

        p_stop = depth / (depth + 5)  # todo: prob to mutate current node
        if random.random() < p_stop and parse_tree.data != 'start':
            return self.__gen_random_tree(self.rule_dict[parse_tree.data.lstrip('?!')], 10)

        rand_child_index = random.randrange(len(parse_tree.children))
        parse_tree.children[rand_child_index] = self.mutate(parse_tree.children[rand_child_index], 1, depth + 1)
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
            assert type(tree_b) == Token, tree_b
            # p_stop = depth / (depth + 3)  # todo depth ~ p
            # if random.random() < p_stop:
            #     return tree_b
            # else:
            #     return tree_a
            return tree_b

        assert type(tree_a) == Tree, tree_a
        assert type(tree_b) == Tree, tree_b
        assert tree_a.data == tree_b.data, (tree_a, tree_b)

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
                    assert OP.type == 'OP', OP
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
                        assert self.__match(tree_a.children[j], name.children[0], is_token), (j, tree_a, name, is_token)
                        assert self.__match(tree_b.children[k], name.children[0], is_token), (k, tree_b, name, is_token)
                        matched.append((j, k))
                        j, k = j + 1, k + 1
                else:
                    if expr.data != 'name':
                        j, k = j + 1, k + 1
                        continue
                    # print(i, 'name', expr)
                    is_token = expr.children[0].type == 'TOKEN'
                    assert self.__match(tree_a.children[j], expr.children[0], is_token), (tree_a, expr, is_token)
                    assert self.__match(tree_b.children[k], expr.children[0], is_token), (tree_b, expr, is_token)
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
            assert subrule.data == 'name', subrule
            is_token = subrule.children[0].type == 'TOKEN'
            assert self.__match(tree_a.children[0], subrule.children[0], is_token), (tree_a, subrule, is_token)
            assert self.__match(tree_b.children[0], subrule.children[0], is_token), (tree_b, subrule, is_token)
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
        self.dsl = DSL(meta_parser, cfg.grammar_file, cfg.wt, cfg.OP_prob, cfg.gen_restart)
        self.population = []
        self.generation = 0
        self.pop_size = cfg.pop_size
        self.elitism = cfg.elitism
        self.depth_lim = cfg.depth_lim
        self.tournament_size = cfg.tournament_size
        self.mutation_rate = cfg.mutation_rate
        if self.tournament_size > self.pop_size:
            raise Exception('tournament_size larger than pop_size')

    def init_population(self, scheme_list=None, eval_mode=False):  # todo: depth ~ distribution / depth range
        assert self.generation == 0, self.generation
        self.generation = 1

        if eval_mode:
            assert scheme_list is not None
            dsl_list = self.__load_schemes(scheme_list)
            for i in range(len(dsl_list)):
                logging.info('Evaluating {}'.format(scheme_list[i]))
                self.population.append(self.dsl.get_scheme_from_dsl(dsl_list[i], scheme_list[i]))
            return

        start = 0
        if scheme_list is not None:
            scheme_list = scheme_list[:self.pop_size]
            start = len(scheme_list)
            dsl_list = self.__load_schemes(scheme_list)
            for i in range(len(dsl_list)):
                self.population.append(self.dsl.get_scheme_from_dsl(dsl_list[i], scheme_list[i]))
        for i in range(start, self.pop_size):
            self.population.append(self.dsl.gen_random_scheme('heuristic', self.depth_lim))

    @staticmethod
    def __load_schemes(scheme_list):
        from synthesis import schemes
        ret = []
        for name in scheme_list:
            try:
                scheme = schemes.__getattribute__(name)
                ret.append(scheme)
            except AttributeError:
                raise Exception('Unknown scheme "{}"'.format(name))
        return ret

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
        tops = [(self.population[i].solved, self.population[i].fitness) for i in range(num)]
        logging.info('Top {} in generation {}: {}'.format(num, self.generation, tops))

    def get_winner(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return self.population[0]

    def save(self, name):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        top = self.population[0]
        shutil.copy(top.file, output_dir / (name + '.csv'))

    def __tournament_selection(self):
        rand_indices = random.sample(range(self.pop_size), self.tournament_size)
        temp_tournament = [self.population[rand_indices[i]] for i in range(self.tournament_size)]
        temp_tournament = sorted(temp_tournament, key=lambda x: x.fitness, reverse=True)
        return deepcopy(temp_tournament[0]), deepcopy(temp_tournament[1])

    def __mutate(self, scheme):
        scheme.update(self.dsl.mutate(scheme.tree, self.mutation_rate, 0))

    def __crossover(self, scheme_0, scheme_1):
        return self.dsl.crossover(scheme_0.tree, scheme_1.tree, 1)

    def test_cross(self):
        ta = self.dsl.gen_random_scheme('heuristic', 15)
        tb = self.dsl.gen_random_scheme('heuristic', 15)
        self.__crossover(ta, tb)

    def test_gen_random(self):
        s = self.dsl.gen_random_scheme('heuristic', self.depth_lim)
        s.display()


def run_gp():
    gp = GP(Config)
    logging.info('--- Initializing population ---')
    gp.init_population(scheme_list=args.load)
    for i in range(Config.epoch):
        logging.info('--- Epoch {} starts ---'.format(i))
        gp.report(Config.report)
        gp.evolve()
        if i % Config.save == 0:
            gp.save('epoch_{}'.format(i))

    winner = gp.get_winner()
    shutil.copy(winner.file, output_dir / 'winner.csv')
    logging.info('Winner: {}'.format(winner.file))
    winner.display()
    logging.info('{} solved, avg_time = {}, fitness = {}'.format(winner.solved, winner.rtime, winner.fitness))


def get_seed():
    if Config.seed is not None:
        return Config.seed
    else:
        return random.randrange(sys.maxsize)


def main():
    if args.eval is not None:
        schemes = args.eval
        gp = GP(Config)
        gp.init_population(schemes, True)
        gp.report(len(schemes))
        return

    temp = vars(Config)
    cfg_str = ' --- Config ---\n'
    for item in temp:
        cfg_str += '\t' + item + ' = ' + str(temp[item]) + '\n'
    cfg_str += '--- End of config ---\n'
    logging.info(cfg_str)
    seed = get_seed()
    random.seed(seed)
    logging.info('Random seed: {}'.format(seed))
    try:
        run_gp()
    except AssertionError as err:
        logging.exception('Assertion failed :(')
        raise err


def monkey():
    def compile_cadical():
        shutil.move('analyze.cpp', '../src/analyze.cpp')
        try:
            subprocess.run('cd .. ; make', shell=True, check=True, capture_output=True)  # ./configure && make
        except subprocess.CalledProcessError:
            return False
        return True

    def embed_cadical(codes):
        SPLIT = [[182], [92, 99], [124, 131]]
        with open('analyze_blank.cpp', 'r') as cpp_file:
            code = cpp_file.readlines()
        for j in range(3):
            code_snippet = codes.evaluate()[j].splitlines()
            for i in range(len(code_snippet)):
                code_snippet[i] = '\t\t' + code_snippet[i]
            code_snippet = '\n'.join(code_snippet)
            for line in SPLIT[j]:
                code.insert(line, code_snippet)
        with open('analyze.cpp', 'w') as cpp_file:
            cpp_file.writelines(code)

    @require()
    @params('heuristic')
    def score(heuristic):
        try:
            embed_cadical(heuristic)
            if not compile_cadical():
                return -sys.maxsize
            subprocess.run('cd .. ; sh python/cadical.sh', shell=True, check=True, capture_output=True)
            process = subprocess.run('BACKUPDIR=$(ls -td ../output/*/ | head -1); DIRNAME=$(basename $BACKUPDIR);'
                                     'python ../python/gen_csv.py -S "cadical" -D ~/Main-18/ -I $BACKUPDIR -O {} -N '
                                     '$DIRNAME'.format(output_dir),
                                     shell=True, check=True, capture_output=True)
            out = process.stdout.decode()
            logging.info(out)
            out = out.split()
            solved, rtime = int(out[0]), float(out[-1][:-1])
            return 30 * solved ** 2 + 60 / rtime
        except subprocess.CalledProcessError as err:
            logging.error(err)
            return -sys.maxsize

    select_fn = functools.partial(tournament_select, selection_size=Config.tournament_size)
    winner = optimize(score, iterations=Config.epoch, population_size=Config.pop_size,
                      next_generation=functools.partial(next_generation, select_fn=select_fn,
                                                        mutation_rate=Config.mutation_rate))
    codes = winner.evaluate()
    print('--- Winner\'s codes ---')
    for code in codes:
        print(code)
    print('--- End of codes ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-O', '--output_root', required=True, type=str)
    parser.add_argument('-E', '--eval', nargs='+', default=None, help='Evaluation mode: run eval for given schemes.')
    parser.add_argument('-L', '--load', nargs='+', default=None, help='Initialize from given schemes.')
    parser.add_argument('-M', '--monkey', action='store_true')
    args = parser.parse_args()

    cur_time = time.strftime('%m%d-%H%M%S')
    output_dir = Path(args.output_root) / cur_time
    Path.mkdir(Path(args.output_root), exist_ok=True)
    Path.mkdir(output_dir, exist_ok=True)

    logging.basicConfig(format='%(levelname)s: %(message)s', filename=str(output_dir / 'log.txt'), level=logging.INFO)
    stdoutLogger = logging.StreamHandler(sys.stdout)
    stdoutLogger.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(stdoutLogger)
    if args.monkey:
        monkey()
    else:
        main()
