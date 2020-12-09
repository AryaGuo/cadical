import argparse
import multiprocessing
from pathlib import Path

from pysat.solvers import Cadical
from pysat.formula import CNF


def solve_instance(cnf):
    with Cadical(CNF(from_file=cnf), use_timer=True) as solver:
        ans = solver.solve()
        return cnf.stem, ans, solver.status, solver.get_model(), solver.get_core(), solver.nof_vars(), solver.time()


def callback(res):
    name, status, ans, sol, core, nv, time = res
    global sat_count, time_sum, unsat_count
    time_sum += time

    fout = Path(args.output_folder) / (name + '.txt')
    with open(fout, 'w') as output:
        if ans:
            sat_count += 1
            if sol is not None:
                output.write(sol)
        else:
            unsat_count += 1
            if core is not None:
                output.write(core)


if __name__ == '__main__':
    # c = Cadical(CNF(from_file='../data_sim/Johnson_sted5_0x24204-50.cnf'), use_timer=True)
    # # c.add_clause([-1, 2])
    # # c.add_clause([-2, 3])
    # # c.add_clause([-3, 4])
    # print(c.nof_vars(), c.nof_clauses())
    # print(c.solve())
    # print(c.time())
    # print(c.get_core())
    # c.delete()
    # exit()

    parser = argparse.ArgumentParser()
    parser.add_argument("-I", "--input_folder", required=True, type=str)
    parser.add_argument("-O", "--output_folder", required=True, type=str)
    parser.add_argument("-T", "--timeout", default=5, type=int)
    args = parser.parse_args()

    # p = multiprocessing.Process(target=solve_instance,
    #                             args=['../../Main-18-bz2/Heusser/satcoin-genesis-SAT-512.cnf.bz2'])
    # p.start()
    # p.join(TIMEOUT)
    # if p.is_alive():
    #     print('alive')
    #     p.terminate()
    #     p.join()

    global sat_count, unsat_count, time_sum
    total_count = 0
    sat_count = 0
    unsat_count = 0
    time_sum = 0

    Path.mkdir(Path(args.output_folder), exist_ok=True)

    it = 0
    with multiprocessing.Pool() as pool:
        for fin in Path(args.input_folder).rglob('*.cnf*'):
            total_count += 1
            res = pool.apply_async(solve_instance, (fin,), callback=callback)
            try:
                out = res.get(args.timeout)
                print(total_count, fin, out[0], out[-2], out[-1])
            except multiprocessing.context.TimeoutError:
                print(total_count, fin, 'timeout')
            if total_count > 10:
                break
        pool.close()
        pool.terminate()
        pool.join()

    total_solved = sat_count + unsat_count
    avg_time = -1 if total_solved == 0 else time_sum / total_solved
    print()
    print('{} out of {} solved, {} sat, {} unsat, {} timeout'.format(total_solved, total_count, sat_count, unsat_count,
                                                                     total_count - total_solved))
    print('Average CPU time: {}'.format(avg_time))
