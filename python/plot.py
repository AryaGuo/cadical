import argparse
import csv
import re
import random

from cycler import cycler
from pathlib import Path
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt


def cactus_plot(args):
    colors = list(get_cmap('tab20').colors)
    colors = colors[:-1:2] + colors[1::2]
    # random.shuffle(colors)
    markers = list(Line2D.filled_markers) + ['x', '.', '+']
    num = min(len(colors), len(markers))
    # cc = cycler(marker=markers[:num]) + cycler(color=colors[:num])
    cc = cycler(color=colors)
    plt.rc('axes', prop_cycle=cc)
    # mks = iter(['x-', 'o-', 's-', 'v-', '<-', '>-', 'P-', 'd-', '.-', '*-', 'D-'])
    i_markers = iter([d + '-' for d in markers])
    plt.figure()
    if args.baseline:
        for fin in sorted(Path(args.baseline_dir).rglob('*.csv')):
            if fin.stem == 'final':
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
            plt.plot(px, py, label=name, alpha=0.8, markersize=5)

    regex = re.compile(args.re)
    for fin in sorted(Path(args.input_dir).rglob('*.csv')):
        if not regex.match(fin.stem):
            continue
        if fin.stem == 'final':
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
        plt.plot(px, py, i_markers.__next__(), label=name if args.label else None, alpha=0.8, markersize=5)

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
        if csv_final.stem == 'final':
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
    parser.add_argument('-B', '--baseline', action='store_true')
    parser.add_argument('-D', '--baseline_dir', default='result/baseline', type=str)
    parser.add_argument('-L', '--label', action='store_true')
    args = parser.parse_args()
    cactus_plot(args)
    gen_csv4all(args)


if __name__ == "__main__":
    main()
