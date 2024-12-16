#!/bin/sh
# SLRNGPU=A6000
# SLRGPUS=
SLRNAME=slave4

# Activar VirtualEnv
eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
conda activate lazyqml
cd /home/diegogv/LazyQML/lazyqml


# Lanzar ejecucion del script
# argumentos
#    Sequential = sys.argv[1].lower() == 'true'
#    Node = sys.argv[2].lower()
#    qubits = int(sys.argv[3])


python lazyqmlVote.py true slave4 16 16
