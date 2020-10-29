import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--threshold", default=60.0, type=float)
args = parser.parse_args()

cnt = 0

filelist = []

with open('main18.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row['solver'] == 'CaDiCaL':
            if float(row['solver time']) <= args.threshold:
                cnt += 1
                filelist.append(Path(row['benchmark']).stem)

with open('problems.txt', 'w') as fout:
    for it in filelist:
        fout.write(it + '\n')

print(cnt)
