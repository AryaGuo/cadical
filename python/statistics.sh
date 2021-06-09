export PROJ_DIR=$(realpath $(dirname $(dirname $0)))
export BACKUPDIR=$(ls -td $PROJ_DIR/output/*/ | head -1)
export DIRNAME=$(basename $BACKUPDIR)

python $PROJ_DIR/python/gen_csv.py \
-S "cadical" \
-D $PROJ_DIR/data/Main-20 \
-I $BACKUPDIR \
-O $1 \
-N $DIRNAME \
-P $PROJ_DIR/python/problems.csv \
-T $2
