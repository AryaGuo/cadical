import argparse
import csv

from pathlib import Path


def parse_output(solver, fout):
    rtime = None
    ans = 'UNKNOWN'
    if solver == 'cadical':
        ff = fout.readlines()
        for line in ff:
            line_lst = line.split()
            if len(line_lst) > 0 and line_lst[0] == 's':
                ans = line_lst[1]
                rtime = float(ff[-7].split()[-2])
                break
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
    return rtime, ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--solver', choices=['minisat', 'bmm', 'cadical'], required=True)
    parser.add_argument("-D", "--data_folder", required=True, type=str, default='~/Main-18')
    parser.add_argument("-I", "--input_folder", required=True, type=str)
    parser.add_argument("-O", "--output_folder", required=True, type=str, default='~/result')
    parser.add_argument("-N", "--name", required=True, type=str)
    args = parser.parse_args()

    total_count = 0
    sat_count = 0
    unsat_count = 0
    time_sum = 0

    data_root = Path(args.data_folder)
    csvfile = Path(args.output_folder.rstrip('/')) / (args.name + '.csv')
    with open(csvfile, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['data_point', 'verdict', 'time'])
        writer.writeheader()
        for fin in sorted(data_root.rglob('*.cnf')):
            data_name = Path(fin).relative_to(data_root)
            fout = Path(args.input_folder) / '{}.txt'.format(fin.stem)
            if Path.exists(fout):
                with open(fout) as output:
                    rtime, ans = parse_output(args.solver, output)
                writer.writerow({'data_point': data_name, 'verdict': ans, 'time': rtime})
                total_count += 1
                if ans == 'SATISFIABLE':
                    sat_count += 1
                elif ans == 'UNSATISFIABLE':
                    unsat_count += 1
                if rtime is not None:
                    time_sum += rtime

    total_solved = sat_count + unsat_count
    avg_time = -1 if total_solved == 0 else time_sum / total_solved
    print('{} out of {} solved, {} sat, {} unsat, {} time out'.format(total_solved, total_count, sat_count, unsat_count,
                                                                      total_count - total_solved))
    print('Averge CPU time: {}s'.format(avg_time))


if __name__ == "__main__":
    main()
