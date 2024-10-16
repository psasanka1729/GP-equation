#!/bin/bash
# Job name:
#SBATCH --job-name=double_well_potential
#
# Partition:
#SBATCH --partition=gpu-preempt
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
#SBATCH -o double_well_potential.o%j
#------------------------------------------------------

echo "Starting job double_well_potential"

module load launcher

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=double_well_potential.task
export LAUNCHER_SCHED=interleaved

$LAUNCHER_DIR/paramrun

echo "Job double_well_potential complete"
