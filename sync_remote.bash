
# Exit on first error
set -e

remotes="ml-23 ml-22"
user_dir="/Scratch/al183"
conda_dir="$user_dir/miniconda3"
avalanche_dir="$user_dir/avalanche"
datasets_dir="$user_dir/datasets"
project_dir="$user_dir/dynamic-dropout"
logs_dir="$project_dir/experiment_logs"
logs_pattern="dynamic-dropout/experiment_logs*"


function push_remote {
    echo "--------------------------------------------------------------------------------"
    echo "Pushing to $1"
    echo "Setting up user directory"
    ssh $1 "mkdir -p $user_dir"

    echo "Sync conda files"
    rsync -a --info=progress2 --delete $conda_dir $1:$user_dir

    echo "Sync Avalanche files"
    rsync -a --info=progress2 --delete $avalanche_dir $1:$user_dir

    echo "Sync dataset files"
    rsync -a --info=progress2 --delete $datasets_dir $1:$user_dir

    echo "Sync project files except logs"
    rsync -a --info=progress2 --delete $project_dir --exclude=$logs_pattern $1:$user_dir
}

function pull_results {
    echo "--------------------------------------------------------------------------------"
    echo "Pulling results from $1"
    echo "Setting up user directory"

    echo "Pulling only log files"
    rsync -a --info=progress2 $1:$logs_dir $logs_dir
}

# Switch based on first argument
case $1 in
    "pull")
        ;;
    "push")
        for remote in $remotes; do
            push_remote $remote
        done
        ;;
    *)
        # Print usage
        echo "Usage: sync_remote.bash [pull|push]"
        ;;
esac