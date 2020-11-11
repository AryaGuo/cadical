import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-T", "--threshold", default=6000, type=float)
args = parser.parse_args()

cnt = 0

input = 'python/main18.csv'
output = 'python/problems.csv'

with open(input) as csvfile:
    reader = csv.DictReader(csvfile)
    with open(output, 'w') as fout:
        writer = csv.DictWriter(fout, fieldnames=['data_point', 'time'])
        writer.writeheader()
        for row in reader:
            if row['solver'] == 'CaDiCaL':
                t = row['solver time']
                if float(t) <= args.threshold:
                    cnt += 1
                    writer.writerow({'data_point': Path(row['benchmark']).stem, 'time': t})
                    writer.writerow({'data_point': Path(row['benchmark']).name, 'time': t})

print(cnt)
