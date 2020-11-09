export PROJ_DIR=$(realpath $(dirname $(dirname $0)))
export DATA_DIR=$(realpath ~/Main-18/)
export BACKUPDIR=$(ls -td $PROJ_DIR/output/*/ | head -1)
export DIRNAME=$(basename $BACKUPDIR)

python $PROJ_DIR/python/gen_csv.py \
-S "cadical" \
-D $DATA_DIR \
-I $BACKUPDIR \
-N $DIRNAME \
-O $1
