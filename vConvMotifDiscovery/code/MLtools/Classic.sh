#!/bin/bash
#SBATCH -J ClassicMotifDis
#SBATCH -p cn-long
#SBATCH -N 1
#SBATCH -o ../log/MT_%j.out
#SBATCH -e ../log/MT_%j.err
#SBATCH --no-requeue
#SBATCH -A gaog_g1
#SBATCH --qos=gaogcnl
#SBATCH -c 1


/home/gaog_pkuhpc/users/lijy/anaconda3/bin/python $1 $2


