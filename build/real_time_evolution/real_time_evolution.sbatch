#!/bin/bash
# Job name:
#SBATCH --job-name=real_time_evolution
#
# Partition:
#SBATCH --partition=gpu-preempt
#
# Processors:
#SBATCH --ntasks=112
#SBATCH --nodes=7
#
# Wall clock limit:
#SBATCH --time=7-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o real_time_evolution.o%j
#------------------------------------------------------

echo "Starting job real_time_evolution"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=real_time_evolution.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job real_time_evolution complete"
