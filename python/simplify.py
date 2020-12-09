import argparse
import random
from pathlib import Path


def read_dimacs(cnf_file, compressed: bool):
    if compressed:
        import bz2
        with bz2.open(cnf_file) as fin:
            data = fin.read().decode()
            lines = data.splitlines(keepends=True)
            for idx, line in enumerate(lines):
                tokens = line.split()
                if tokens[0] == 'c':
                    continue
                if tokens[0] == 'p':
                    nv, nc = int(tokens[-2]), int(tokens[-1])
                    lines.pop(idx)
                    return nv, nc, lines
    else:
        with open(cnf_file) as fin:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                tokens = line.split()
                if tokens[0] == 'c':
                    continue
                if tokens[0] == 'p':
                    nv, nc = int(tokens[-2]), int(tokens[-1])
                    lines.pop(idx)
                    return nv, nc, lines
    raise Exception('invalid CNF file')


def main():
    Path.mkdir(Path(args.output_folder), exist_ok=True)
    nvs, ncs = [], []

    if args.strategy == 'random':
        for fin in Path(args.input_folder).rglob('*.cnf*'):
            compressed = 'bz2' in fin.suffix
            nv, nc, lines = read_dimacs(fin, compressed)
            nvs.append(nv)
            ncs.append(nc)
            assign = []
            add_stat = max(0, nv - args.size)

            assign.append('p cnf ' + str(nv) + ' ' + str(nc + add_stat) + '\n')
            vars = random.choices(range(1, nv + 1), k=add_stat)
            for var in vars:
                assign.append(('-' if random.random() < 0.5 else '') + str(var) + ' 0' + '\n')
            assign += lines
            fout = Path(args.output_folder) / (fin.parent.name + '_' + fin.name.split('.')[0] + '.cnf')
            with open(fout, 'w') as new_file:
                new_file.writelines(assign)

    print('---Original statistics---')
    print('Vars: [{}, {}], mean = {}'.format(min(nvs), max(nvs), sum(nvs) / len(nvs)))
    print('Clauses: [{}, {}], mean = {}'.format(min(ncs), max(ncs), sum(ncs) / len(ncs)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-S', '--strategy', type=str, required=True, help='random')
    parser.add_argument('-I', '--input_folder', type=str, required=True)
    parser.add_argument('-O', '--output_folder', type=str, required=True)
    parser.add_argument('-N', '--size', type=int, default=1000)
    args = parser.parse_args()

    main()
