import logging
import random
import shutil
import json
from copy import deepcopy
from pathlib import Path

from lark import Lark

from synthesis.DSL import DSL
from synthesis.Scheme import Scheme
from synthesis.config import cfg


class GP:
    def __init__(self):
        with open(cfg.meta_file) as meta_grammar:
            meta_parser = Lark(meta_grammar)
        self.dsl = DSL(meta_parser, cfg.grammar_file)
        self.population = []
        self.generation = 0
        self.pop_size = cfg.pop_size
        self.elitism = cfg.elitism
        self.depth_lim = cfg.depth_lim - 1
        self.tournament_size = cfg.tournament_size
        if self.tournament_size > self.pop_size:
            raise Exception('tournament_size larger than pop_size')

    def init_population(self, scheme_list=None, eval_mode=False):
        assert self.generation == 0, self.generation
        self.generation = 1
        load_file = Path(cfg.output_dir+'/'+cfg.popfilename)
        
        if load_file.is_file():
            logging.info('Loading population from file: {} '.format(cfg.output_dir+'/'+cfg.popfilename))
            with open(load_file, "r") as read_file:
                self.population = json.loads(read_file)
            return
        
        if eval_mode:
            assert scheme_list is not None
            dsl_list = self.__load_schemes(scheme_list)
            for i in range(len(dsl_list)):
                logging.info('Evaluating {}'.format(scheme_list[i]))
                self.population.append(self.dsl.get_scheme_from_dsl(dsl_list[i], True, scheme_list[i]))
            return

        start = 0
        if scheme_list is not None:
            scheme_list = scheme_list[:self.pop_size]
            start = len(scheme_list)
            dsl_list = self.__load_schemes(scheme_list)
            for i in range(len(dsl_list)):
                self.population.append(self.dsl.get_scheme_from_dsl(dsl_list[i], scheme_list[i], False))
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
            new_population.append(deepcopy(max(self.population, key=lambda x: x.fitness)))

        for i in range(start_index, self.pop_size):
            parent_0, parent_1 = self.__tournament_selection()
            new_population.append(Scheme(None, self.__crossover(parent_0, parent_1), False, None))
            self.__mutate(new_population[-1])

        self.population = new_population
        self.generation += 1

    def report(self, num):
        num = min(num, self.pop_size)
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        tops = [(self.population[i].solved, self.population[i].fitness) for i in range(num)]
        avg = sum(self.population[i].fitness for i in range(self.pop_size)) / self.pop_size
        logging.info('Top {} in generation {}: {}; Avg = {}'.format(num, self.generation, tops, avg))
        return tops, avg

    def get_winner(self):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return self.population[0]

    def save(self, name):
        self.population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        top = self.population[0]
        shutil.copy(top.file, cfg.output_dir / (name + '.csv'))
        
    def dump(self, name):
        with open(cfg.output_dir+'/'+cfg.popfilename, 'w+') as f:
            json.dumps(self.population, f)
        

    def __tournament_selection(self):
        rand_indices = random.sample(range(self.pop_size), self.tournament_size)
        temp_tournament = [self.population[rand_indices[i]] for i in range(self.tournament_size)]
        temp_tournament = sorted(temp_tournament, key=lambda x: x.fitness, reverse=True)
        return deepcopy(temp_tournament[0]), deepcopy(temp_tournament[1])

    def __mutate(self, scheme):
        if random.random() < cfg.mutation_rate:
            if cfg.STGP:
                scheme.update(self.dsl.mutate_(scheme.tree))
            else:
                scheme.update(self.dsl.mutate(scheme.tree, 0))
        return scheme.tree

    def __crossover(self, scheme_0, scheme_1):
        if random.random() < cfg.crossover_rate:
            if cfg.STGP:
                return self.dsl.crossover_(scheme_0.tree, scheme_1.tree)
            else:
                return self.dsl.crossover(scheme_0.tree, scheme_1.tree, 1)
        return scheme_0.tree
