#!/bin/sh
# SLRNGPU=A6000
# SLRGPUS=
SLRNAME=slave3

# Activar VirtualEnv
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda activate lazyqml
cd /home/diegogv/LazyQML/lazyqml


# Lanzar ejecucion del script
# argumentos
#   Rotationals_family = sys.argv[1].lower() == 'true'
#   Batch_auto = sys.argv[2].lower() == 'true'
#   Sequential = sys.argv[3].lower() == 'true'
#   Node = sys.argv[4].lower()

# EXECUTIONS -> Done
# true true false -> slave4; slave5; slave3
# true false false -> slave4; slave5; slave3
# false true false -> slave4; slave5; slave3
# false false false -> slave4; slave5; slave3

# true true true -> slave4; slave5; slave3
# true false true -> slave4; slave5; slave3
# false true true -> slave4; slave5; slave3
# false false true -> slave4; slave5;

python lazyqmlP.py true true true slave4
