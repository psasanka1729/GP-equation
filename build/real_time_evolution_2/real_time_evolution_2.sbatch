#!/bin/bash
# Job name:
#SBATCH --job-name=real_time_evolution_2
#
# Partition:
#SBATCH --partition=CMT
#
# Processors:
#SBATCH --ntasks=16
#SBATCH --nodes=1
#
# Wall clock limit:
#SBATCH --time=7-00:00:00
#
# Email at end:
#SBATCH --mail-user=sxd190113@utdallas.edu
#SBATCH --mail-type=END
#
# Specify output files:
#SBATCH -o real_time_evolution_2.o%j
#------------------------------------------------------

echo "Starting job real_time_evolution_2"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=real_time_evolution_2.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job real_time_evolution_2 complete"
