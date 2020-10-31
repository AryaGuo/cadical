import argparse
import csv
import re

from pathlib import Path

import matplotlib.pyplot as plt


def cactus_plot(args):
    mks = iter(['x-', 'o-', 's-', 'v-', '<-', '>-', 'P-', 'd-', '.-', '*-', 'D-'])
    mksc = iter(['x:', 'o:', 's:', 'v:', '<:', '>:', 'P:', 'd:', '.:', '*:', 'D:'])

    plt.figure()
    for fin in sorted(Path(args.baseline_dir).rglob('*.csv')):
        px, py = [0], [0]
        rtime = []
        name = fin.stem
        with open(fin) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sat = row['verdict']
                if sat != 'UNKNOWN':
                    rtime.append(float(row['time']))
        rtime.sort()
        for i, j in enumerate(rtime):
            if j > args.time_lim:
                break
            px.append(i)
            py.append(j)
        plt.plot(px, py, mks.__next__(), label=name, alpha=0.5, markersize=5)

    regex = re.compile(args.re)
    for fin in sorted(Path(args.input_dir).rglob('*.csv')):
        if not regex.match(fin.stem):
            continue
        px, py = [0], [0]
        rtime = []
        name = fin.stem
        with open(fin) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sat = row['verdict']
                if sat != 'UNKNOWN':
                    rtime.append(float(row['time']))
        rtime.sort()
        for i, j in enumerate(rtime):
            if j > args.time_lim:
                break
            px.append(i)
            py.append(j)
        plt.plot(px, py, label=name if args.label else None, alpha=0.8, markersize=5)

    plt.xlim(0)
    plt.ylim(0, args.time_lim)
    plt.legend()
    plt.xlabel('Number of solved instances')
    plt.ylabel('Time (s)')
    plt.savefig(Path(args.input_dir) / 'fig.pdf')


def gen_csv4all(args):
    # data_point,verdict,time
    csv_final = Path(args.input_dir) / 'final.csv'
    data_points = []
    runtimes = []
    fields = ['data_point']
    regex = re.compile(args.re)

    flag = True
    for csvfile in sorted(Path(args.input_dir).rglob('*.csv')):
        if not regex.match(csvfile.stem):
            continue
        fields.append(str(csvfile.stem))
        with open(csvfile) as cur:
            reader = csv.DictReader(cur)
            runtime = []
            for row in reader:
                if flag:
                    data_points.append(row['data_point'])
                if row['verdict'] != 'UNKNOWN':
                    runtime.append(float(row['time']))
                else:
                    runtime.append(args.time_lim)
            runtimes.append(runtime)
        if flag:
            flag = False
    row_datas = [data_points] + runtimes
    col_datas = zip(*row_datas)

    with open(csv_final, 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(fields)
        for col_data in col_datas:
            writer.writerow(col_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-I', '--input_dir', required=True, type=str)
    parser.add_argument('-T', '--time_lim', default=5000, type=float)
    parser.add_argument('-R', '--re', default='.*', type=str)
    parser.add_argument('-B', '--baseline', default=True, type=bool)
    parser.add_argument('-D', '--baseline_dir', default='../result_baseline', type=str)
    parser.add_argument('-L', '--label', default=False, type=bool)
    args = parser.parse_args()
    cactus_plot(args)
    # gen_csv4all(args)


if __name__ == "__main__":
    main()
