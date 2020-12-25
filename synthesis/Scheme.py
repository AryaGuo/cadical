import logging
import shutil
import subprocess
import sys

from lark import Token

from synthesis.Node import Node
from synthesis.config import cfg


class Scheme:
    """
    Describes a specific heuristic scheme.
    Can be described by dsl(str), a parse tree(Node) or a c++ code snippet.
    """

    def __init__(self, dsl, tree, test_mode, name=None):
        self.dsl = dsl
        self.tree = Node.convert_tree(tree, None, None)
        self.code = self.__dsl_to_cpp(self.tree)
        self.eval(test_mode)
        self.rename(name)

    def rename(self, name):
        if name is not None and self.file is not None:
            dst = self.file.parent / (name + '.csv')
            shutil.move(self.file, dst)
            self.file = dst

    def display(self):
        logging.info('-----')
        for code in self.code:
            logging.info(code)
        logging.info('-----\n')

    def __dsl_to_cpp_(self, tree, term_stabs: dict):
        def is_token(t):
            if type(t) == Node:
                return t.is_token
            return type(t) == Token

        if not is_token(tree):
            ret = ''
            first = True
            if_body = False
            for c in tree.children:
                if is_token(c) and c.value == 'unbumped' and tree.data == 'assign_unbumped':
                    ret += 'for (auto var : vars) {\n\t'
                    first = True
                if if_body and first:
                    ret += '\t'
                if not first:
                    ret += ' '
                first = False
                ret += self.__dsl_to_cpp_(c, term_stabs)
                if not is_token(c) and c.data == 'condition':
                    if_body = True
                    ret += ' {\n\t'
            if tree.data == 'assign_unbumped':
                ret += '\n\t}' if if_body else '\n}'
            if if_body:
                ret += '\n}'
            return ret
        elif is_token(tree):
            if tree.type in term_stabs:
                return term_stabs[tree.type]
            else:
                return tree.value

    def __dsl_to_cpp(self, tree):
        assert tree.data == 'start', tree

        tree = tree.children[0]

        codes = [''] * 3
        term_subs = {'CONFLICT_INDEX': 'stats.conflicts', 'SCORE_INC': 'score_inc', 'UNBUMPED': 'stab[var]',
                     'BUMPED': 'new_score', 'NEW_SCORE_INC': 'new_score_inc'}
        for c in tree.children:
            if c.data == 'assign_unbumped':
                term_subs['LHS'] = 'stab[var]'
                codes[0] = self.__dsl_to_cpp_(c, term_subs)
            elif c.data == 'assign_new_score':
                term_subs['LHS'] = 'old_score'
                codes[1] = self.__dsl_to_cpp_(c, term_subs)
            elif c.data == 'assign_score_inc':
                term_subs['LHS'] = 'score_inc'
                codes[2] = self.__dsl_to_cpp_(c, term_subs)
        return codes

    @staticmethod
    def embed_cadical(codes):
        SPLIT = [[182], [92, 99], [124, 131]]
        with open('analyze_blank.cpp', 'r') as cpp_file:
            code = cpp_file.readlines()
        for j in range(3):
            if codes[j] is None:
                continue
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

    def eval(self, test_mode=False):
        self.solved, self.rtime, self.fitness, self.file = self.__eval_fitness(
            cfg.test_time if test_mode else cfg.eval_time)

    def __eval_fitness(self, time_lim):
        def score(scheme):
            if scheme == 'par-2':
                return -(rtime * solved + (cnt - solved) * 2 * time_lim) / cnt
            if scheme == 'ratio':
                return -ratio
            return 30 * solved ** 2 + 60 / rtime

        try:
            self.embed_cadical(self.code)
            if not self.compile_cadical():
                return 0, 0, 0, None  # Compilation error
            subprocess.run('cd .. ; sh python/cadical.sh ' + str(time_lim) + ' ' + str(cfg.config), shell=True,
                           check=True, capture_output=True)
            get_name = subprocess.run('basename $(ls -td ../output/*/ | head -1)', shell=True, check=True,
                                      capture_output=True)
            basename = get_name.stdout.decode().strip()
            process = subprocess.run('sh ../python/statistics.sh ' + str(cfg.output_dir) + ' ' + str(time_lim),
                                     shell=True, check=True, capture_output=True)
            out = process.stdout.decode().strip()
            logging.info('[{}] {}'.format(basename, out))
            out = out.splitlines()
            solved, cnt, rtime, ratio = int(out[0].split()[0]), int(out[0].split()[3]), float(
                out[-1].split()[3][:-1]), float(out[-1].split()[-1][:-1])
            csvfile = cfg.output_dir / (basename + '.csv')
            fitness = score(cfg.score)
            logging.info('Fitness = {}'.format(fitness))
            self.display()
            return solved, rtime, fitness, csvfile
        except subprocess.CalledProcessError as err:
            logging.error(err)
            return 0, 0, -sys.maxsize, None

    def update(self, new_tree):
        self.tree = Node.convert_tree(new_tree, None, None)
        self.code = self.__dsl_to_cpp(self.tree)
        self.eval()
