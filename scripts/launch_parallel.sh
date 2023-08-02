#!/bin/bash

if [ ! -f "$1" ]
then
    echo "Error: file passed does not exist"
    exit 1
fi

# This convoluted way of counting also works if a final EOL character is missing
n_lines=$(grep -c '^' "$1")
max_in_parallel=256

jobs_in_parallel=$(( $n_lines < $max_in_parallel ? $n_lines : $max_in_parallel ))
job_name=$(basename $(dirname "$1"))

sbatch --array=1-${n_lines}%${jobs_in_parallel} \
	--job-name $job_name \
	"${@:2}" \
	$(dirname "$0")/launch_job.sh "$1"
