export PROJ_DIR=$(realpath $(dirname $(dirname $0)))

python $PROJ_DIR/python/solvers.py \
-S $PROJ_DIR/build/cadical \
-I $PROJ_DIR/data/Main-18 \
-O $PROJ_DIR/output \
-N 16 \
-T $1 \
-P $PROJ_DIR/python/problems.csv \
-X=$2
