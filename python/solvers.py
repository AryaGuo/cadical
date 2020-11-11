import argparse
import csv
import logging
import multiprocessing
import subprocess
import sys
import time
from multiprocessing import set_start_method
from pathlib import Path
from subprocess import CalledProcessError


def worker(i, fin, fout, args):
    #	print("worker i:{} pid:{} ".format(i, os.getpid()))
    try:
        start = time.time()
        com = [args.solver]
        if args.arguments is not None:
            com = com + args.arguments.split(' ')
        com.append(fin)
        subprocess.run(com, timeout=args.timeout, stdout=open(fout, 'w'))
        end = time.time()
        run_time = end - start
        logger.info('Task {} done. Run time: {}'.format(i, run_time))
    except subprocess.TimeoutExpired as e:
        logger.error('Task {}: Time out'.format(i))
        run_time = -1
    except CalledProcessError as e:
        # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
        logger.error("error code {}, error output: {}".format(e.returncode, e.output))
        run_time = -1
    except Exception as e:
        logger.error('Exception: {}\n{}'.format(type(e), e))
        run_time = -1
    return run_time, fout


def cal_result(res):
    run_time, fout = res
    global sat_count, time_sum, unsat_count
    with open(fout, 'r') as _:
        ff = _.readlines()
        for line in ff:
            line_lst = line.split()
            if len(line_lst) > 0 and line_lst[0] == 's':
                sat = line_lst[1]
                if sat == 'SATISFIABLE':
                    sat_count += 1
                    time_sum += run_time
                elif sat == 'UNSATISFIABLE':
                    unsat_count += 1
                    time_sum += run_time
                break


def create_logger(log_file):
    global logger
    logger = multiprocessing.get_logger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s| %(levelname)s| %(processName)s] %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    # this bit will make sure you won't have 
    # duplicated messages in the output
    if not len(logger.handlers):
        logger.addHandler(handler)

    stdoutLogger = logging.StreamHandler(sys.stdout)
    stdoutLogger.setFormatter(formatter)
    logger.addHandler(stdoutLogger)

    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", "--solver", required=True, type=str)
    parser.add_argument("-I", "--input_folder", required=True, type=str)
    parser.add_argument("-O", "--output_folder", required=True, type=str)
    parser.add_argument("-N", "--num_processes", default=16, type=int)
    parser.add_argument("-T", "--timeout", default=3600, type=float)
    parser.add_argument("-W", "--wait_timeout", default=10, type=int)
    parser.add_argument("-P", "--problems", required=True, default=None, type=str)
    parser.add_argument("-p", "--predict_folder", default="", type=str)
    parser.add_argument('-s', "--random_seed", default=1997, type=int)
    parser.add_argument('-X', "--arguments", type=str)
    args = parser.parse_args()

    cur_time = time.strftime('%m%d-%H%M%S')
    output_root = Path(args.output_folder) / cur_time
    Path.mkdir(Path(args.output_folder), exist_ok=True)
    Path.mkdir(output_root, exist_ok=True)

    #    logging.basicConfig(format='%(levelname)s: %(message)s', filename=str(output_root / 'log.txt'), level=logging.INFO)
    #    stdoutLogger = logging.StreamHandler(sys.stdout)
    #    stdoutLogger.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    #    logging.getLogger().addHandler(stdoutLogger)
    logger = create_logger(str(output_root / 'log.txt'))
    logger.info('Starting pooling')

    logger.info('{}\n'.format(sys.argv))

    global sat_count, unsat_count, time_sum
    total_count = 0
    sat_count = 0
    unsat_count = 0
    time_sum = 0
    pool = multiprocessing.Pool(args.num_processes)

    try:
        if args.problems:
            with open(args.problems) as problem_list:
                reader = csv.DictReader(problem_list)
                for row in reader:
                    fname = row['data_point']
                    for fin in Path(args.input_folder).rglob(fname):
                        total_count += 1
                        fout = str(output_root) + '/' + '{}.txt'.format(fin.stem)
                        pool.apply_async(worker, args=(total_count, fin, fout, args), callback=cal_result)
        else:
            for fin in Path(args.input_folder).rglob('*.cnf*'):
                total_count += 1
                # fout = output_root / '{}.txt'.format(fin.stem)
                fout = str(output_root) + '/' + '{}.txt'.format(fin.stem)
                pool.apply_async(worker, args=(total_count, fin, fout, args), callback=cal_result)
    except Exception as e:
        logger.error(e)
    pool.close()
    pool.join()

    total_solved = sat_count + unsat_count
    avg_time = -1 if total_solved == 0 else time_sum / total_solved
    logger.info(
        '{} out of {} solved, {} sat, {} unsat, {} timeout'.format(total_solved, total_count, sat_count, unsat_count,
                                                                   total_count - total_solved))
    logger.info('Average CPU time: {}'.format(avg_time))


#	logger.info("# of tasks: {}".format(it))

#	csvfile = args.output_folder.rstrip("/") + ".csv"
#	outputfile = open(csvfile, "wt")
#	stat_result.stat_result(args.output_folder, outputfile)

if __name__ == "__main__":
    #    set_start_method("spawn")
    main()
