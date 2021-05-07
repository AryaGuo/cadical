import argparse
import functools
import importlib
import logging
import random
import shutil
import subprocess
import sys
import time
from distutils.util import strtobool
from pathlib import Path

import numpy as np
from tensorboardX import SummaryWriter

curPath = Path(__file__).resolve()
sys.path.append(str(curPath.parents[1]))

from synthesis.GP import GP
from synthesis.config import cfg
from synthesis.Scheme import Scheme

from monkeys import optimize, tournament_select, next_generation, build_tree
from monkeys.typing import params
from monkeys.search import require, pre_evaluate


def get_seed():
    return cfg.seed if cfg.seed is not None else random.randrange(sys.maxsize)


def main():
    writer = SummaryWriter('../runs/' + cur_time)

    if cfg.eval is not None:
        init_dataset(cfg.test_threshold)
        schemes = cfg.eval
        gp = GP()
        gp.init_population(schemes, True)
        gp.report(len(schemes))
        return

    try:
        init_dataset(cfg.eval_threshold)
        gp1 = GP()
        logging.info('--- Initializing population ---')
        gp1.init_population(scheme_list=cfg.load)
        for i in range(cfg.epoch):
            logging.info('--- Epoch {} starts ---'.format(i))
            tops, avg = gp1.report(cfg.report)
            gp1.evolve()
            if i % cfg.save == 0:
                gp1.save('epoch_{}'.format(i))
            if cfg.period is not 0:
                if (cfg.epoch+1) % cfg.period == 0:
                    gp1.dump()
            writer.add_scalar('best_fitness', np.array(tops[0][1]), i)
            writer.add_scalar('best_solved', np.array(tops[0][0]), i)
            writer.add_scalar('avg_fitness', avg, i)

        winner = gp1.get_winner()
        shutil.copy(winner.file, output_dir / 'winner.csv')
        logging.info('Winner: {}'.format(winner.file))
        winner.display()
        init_dataset(cfg.test_threshold)
        winner.eval(True)
        winner.rename('test')
        logging.info('{} solved, avg_time = {}, fitness = {}'.format(winner.solved, winner.rtime, winner.fitness))
    except AssertionError as err:
        logging.exception('Assertion failed :(')
        raise err


def monkeys():
    init_dataset(cfg.eval_threshold)
    grammar = importlib.import_module(cfg.monkeys)

    def display(codes):
        logging.info('-----')
        for code in codes:
            logging.info(code)
        logging.info('-----\n')

    @require()
    @params('heuristic')
    @pre_evaluate
    def score(heuristic):
        try:
            Scheme.embed_cadical(heuristic)
            if not Scheme.compile_cadical():
                return -sys.maxsize
            subprocess.run('cd .. ; sh python/cadical.sh ' + str(cfg.eval_time), shell=True, check=True,
                           capture_output=True)
            process = subprocess.run('sh ../python/statistics.sh ' + str(output_dir) + ' ' + str(cfg.eval_time),
                                     shell=True, check=True, capture_output=True)
            out = process.stdout.decode().strip()
            logging.info(out)
            out = out.split()
            solved, rtime = int(out[0]), float(out[-1][:-1])
            display(heuristic)
            return 30 * solved ** 2 + 60 / rtime
        except subprocess.CalledProcessError as err:
            logging.error(err)
            return -sys.maxsize

    build_tree_ = functools.partial(build_tree, selection_strategy=grammar.selection_strategy)
    select_fn = functools.partial(tournament_select, selection_size=cfg.tournament_size)
    winner = optimize(score, iterations=cfg.epoch, population_size=cfg.pop_size,
                      next_generation=functools.partial(next_generation, select_fn=select_fn, build_tree=build_tree_,
                                                        crossover_rate=cfg.crossover_rate,
                                                        mutation_rate=cfg.mutation_rate))
    display(winner.evaluate())


def init_dataset(time_lim):
    logging.info('\nFiltering datasets for evaluation...')
    filtering = subprocess.run('cd ..; python python/filter.py -T ' + str(time_lim), shell=True, check=True,
                               capture_output=True)
    out = filtering.stdout.decode().strip()
    logging.info(out + ' problems in total\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-F', '--grammar_file', type=str)

    parser.add_argument('-O', '--output_root', type=str)
    parser.add_argument('-N', '--pop_size', type=int)
    parser.add_argument('-D', '--depth_lim', type=int)
    parser.add_argument('-S', '--tournament_size', type=int)
    parser.add_argument('-e', '--epoch', type=int)

    parser.add_argument('-t', '--eval_time', type=int)
    parser.add_argument('-r', '--eval_threshold', type=int)
    parser.add_argument('-T', '--test_time', type=int)
    parser.add_argument('-H', '--test_threshold', type=int)
    parser.add_argument('-E', '--eval', nargs='+', default=None, help='Evaluation mode: run eval for given schemes.')

    parser.add_argument('-R', '--score', type=str)
    parser.add_argument('-s', '--STGP', type=lambda x: bool(strtobool(x)))
    parser.add_argument('-M', '--monkeys', type=str)
    parser.add_argument('-L', '--load', nargs='+', default=None, help='Initialize from given schemes.')
    args = parser.parse_args()

    for k, v in vars(args).items():
        if v is not None:
            cfg.__setattr__(k, v)


if __name__ == '__main__':
    parse_args()

    cur_time = time.strftime('%m%d-%H%M%S')
    output_dir = Path(cfg.output_root) / cur_time
    cfg.output_dir = output_dir
    Path.mkdir(Path(cfg.output_root), exist_ok=True)
    Path.mkdir(output_dir, exist_ok=True)

    logging.basicConfig(format='%(levelname)s: %(message)s', filename=str(output_dir / 'log.txt'), level=logging.INFO)
    stdoutLogger = logging.StreamHandler(sys.stdout)
    stdoutLogger.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(stdoutLogger)

    cfg_str = ' --- config ---\n'
    for k, v in vars(cfg).items():
        cfg_str += '\t' + k + ' = ' + str(v) + '\n'
    cfg_str += '--- End of config ---\n'
    logging.info(cfg_str)
    seed = get_seed()
    random.seed(seed)
    logging.info('Random seed: {}'.format(seed))

    if cfg.monkeys:
        monkeys()
    else:
        main()
