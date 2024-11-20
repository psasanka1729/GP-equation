#!/bin/bash
# Job name:
#SBATCH --job-name=transistor
#
# Partition:
#SBATCH --partition=normal
#
# Processors:
#SBATCH --ntasks=64
#SBATCH --nodes=4
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o transistor.o%j
#------------------------------------------------------

echo "Starting job transistor"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=transistor.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job transistor complete"
