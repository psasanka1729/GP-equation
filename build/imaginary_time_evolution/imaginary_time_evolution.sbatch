#!/bin/bash
# Job name:
#SBATCH --job-name=imaginary_time_evolution
#
# Partition:
#SBATCH --partition=256i
#
# Processors:
#SBATCH --ntasks=32
#SBATCH --nodes=2
#
# Wall clock limit:
#SBATCH --time=4-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o imaginary_time_evolution.o%j
#------------------------------------------------------

echo "Starting job imaginary_time_evolution"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=imaginary_time_evolution.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job imaginary_time_evolution complete"
