import argparse
import csv

from pathlib import Path


def parse_output(solver, fout, baseline, timeout):
    rtime = None
    ans = 'UNKNOWN'
    score = None
    if solver == 'cadical':
        ff = fout.readlines()
        for line in ff:
            line_lst = line.split()
            if len(line_lst) > 0 and line_lst[0] == 's':
                ans = line_lst[1]
                rtime = float(ff[-7].split()[-2])
                score = rtime / baseline
                break
        if rtime is None:
            score = timeout * 2 / baseline
    elif solver == 'minisat':
        ff = fout.readlines()
        if len(ff) > 0:
            sat = ff[-1].split()[-1]
            if sat == 'SATISFIABLE' or sat == 'UNSATISFIABLE':
                ans = sat
                rtime = float(ff[-3].split()[-2])
    elif solver == 'bmm':
        ff = fout.readlines()
        for (it, _) in enumerate(ff):
            line = _.split()
            if len(line) > 0 and line[0] == 's':
                ans = line[1]
                rtime = float(ff[it - 2].split()[-2])
                break
    else:
        raise Exception('[parse_output] unknown solver')
    return rtime, ans, score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--solver', choices=['minisat', 'bmm', 'cadical'], required=True)
    parser.add_argument("-D", "--data_folder", required=True, type=str, default='~/Main-18')
    parser.add_argument("-I", "--input_folder", required=True, type=str)
    parser.add_argument("-O", "--output_folder", required=True, type=str, default='~/result')
    parser.add_argument("-N", "--name", required=True, type=str)
    parser.add_argument("-P", "--problems", required=True, default=None, type=str)
    parser.add_argument("-T", "--timeout", default=3600, type=float)
    args = parser.parse_args()

    total_count = 0
    sat_count = 0
    unsat_count = 0
    time_sum = 0
    sum_score = 0

    csvfile = Path(args.output_folder.rstrip('/')) / (args.name + '.csv')
    with open(csvfile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['data_point', 'verdict', 'time', 'score'])
        writer.writeheader()

        with open(args.problems) as problem_list:
            reader = csv.DictReader(problem_list)
            for row in reader:
                data_name = row['data_point']
                baseline = float(row['time'])
                fout = Path(args.input_folder) / '{}.txt'.format(Path(data_name).stem)
                if Path.exists(fout):
                    with open(fout) as output:
                        rtime, ans, score = parse_output(args.solver, output, baseline, args.timeout)
                    writer.writerow({'data_point': data_name, 'verdict': ans, 'time': rtime, 'score': score})
                    total_count += 1
                    sum_score += score
                    if ans == 'SATISFIABLE':
                        sat_count += 1
                    elif ans == 'UNSATISFIABLE':
                        unsat_count += 1
                    if rtime is not None:
                        time_sum += rtime

    total_solved = sat_count + unsat_count
    avg_time = -1 if total_solved == 0 else time_sum / total_solved
    avg_score = 1000 if total_count == 0 else sum_score / total_count
    print('{} out of {} solved, {} sat, {} unsat, {} timeout'.format(total_solved, total_count, sat_count, unsat_count,
                                                                     total_count - total_solved))
    print('Average CPU time: {}. Average score = {}.'.format(avg_time, avg_score))


if __name__ == "__main__":
    main()
