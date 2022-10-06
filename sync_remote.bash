
# Exit on first error
set -e

remotes="ml-14 ml-22 ml-23 ml-24 ml-25 ml-21"
# remotes="ml-21"
user_dir="/Scratch/al183"
conda_dir="$user_dir/miniconda3"
avalanche_dir="$user_dir/avalanche"
datasets_dir="$user_dir/datasets"
project_dir="$user_dir/dynamic-dropout"
logs_dir="$project_dir/experiment_logs"
logs_pattern="dynamic-dropout/experiment_logs*"

function rsync_cmd {
    rsync --timeout=3 $@ || echo "Timeout Skipping 'rsync $@'"
}


function push_remote {
    echo "--------------------------------------------------------------------------------"
    echo "Pushing to $1"
    echo "Setting up user directory"
    ssh $1 "mkdir -p $user_dir $logs_dir"

    echo "Sync conda files"
    rsync_cmd -a --info=progress2 --delete $conda_dir $1:$user_dir

    echo "Sync Avalanche files"
    rsync_cmd -a --info=progress2 --delete $avalanche_dir $1:$user_dir

    echo "Sync dataset files"
    rsync_cmd -a --info=progress2 --delete $datasets_dir $1:$user_dir

    echo "Sync project files except logs"
    rsync_cmd -a --info=progress2 --delete $project_dir --exclude=$logs_pattern --exclude=nohup.out $1:$user_dir
}

function pull_results {
    echo "--------------------------------------------------------------------------------"
    echo "Pulling results from $1"
    echo "Pulling only log files"
    rsync_cmd -a --info=progress2 $1:$logs_dir $project_dir
}

# Switch based on first argument
case $1 in
    "pull")
        for remote in $remotes; do
            # Do not pull from self
            if [ $remote != $(hostname) ]; then
                pull_results $remote
            fi
        done
        ;;
    "push")
        for remote in $remotes; do
            # Do not push to self
            if [ $remote != $(hostname) ]; then
                push_remote $remote
            fi
        done
        ;;
    "status")
        for remote in $remotes; do
            echo "--------------------------------------------------------------------------------"
            printf "Status of $remote"

            # If can connect
            if ssh $remote "exit" &> /dev/null ; then
                printf " [OK]\n"
            else
                printf " [FAIL]\n"
                continue
            fi

            compute_apps=$(timeout 5 ssh $remote -- "nvidia-smi --query-compute-apps=pid --format=csv | tail -n +2")
            utilization=$(timeout 5 ssh $remote -- "nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv | tail -n +2")

            util_gpu=$(echo $utilization | cut -d',' -f1 | xargs)
            util_mem=$(echo $utilization | cut -d',' -f2 | xargs)
            echo "GPU: $util_gpu"
            echo "MEM: $util_mem"
            echo

            # echo $compute_apps | 
            # Loop over lines
            while IFS= read -r line; do
                if [ -z "$line" ]; then
                    continue
                fi

                # Get PID
                pid=$(echo -n $line | cut -d',' -f1)
                # Get command
                # Needs /dev/null otherwise stdin will override the while loop
                # Bash is dumb!
                cmd=$(ssh $remote -- "ps -p $pid -o command | tail +2" < /dev/null) 
                printf "(%s) %s\n"  "$pid" "$cmd"
            done <<< $compute_apps

        done
        ;;
    *)
        # Print usage
        echo "Usage: sync_remote.bash [pull|push]"
        ;;
esac
