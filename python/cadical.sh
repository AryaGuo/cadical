export PROJ_DIR=$(realpath $(dirname $(dirname $0)))
export DATA_DIR=$(realpath ~/Main-18/)

python $PROJ_DIR/python/solvers.py \
-S $PROJ_DIR/build/cadical \
-I $DATA_DIR \
-O $PROJ_DIR/output \
-N 16 \
-T $1 \
-P $PROJ_DIR/python/problems.txt \
-X='--sat'
