#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --cpus-per-task=96
#SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes=1
set -e

USE_NSYS=0

usage() {
    echo
    echo "Usage: $0 [ -p Enable nsys profiling]" 1>&2
}

exit_abnormal() {
    usage
    exit 1
}

while getopts "p" flag
do
    case $flag in
        p)
            USE_NSYS=1
            ;;
        :)
            echo "Error: -$flag requires an argument."
            exit_abnormal
            ;;
        *)
            echo "Error: unknown flag: -$OPTARG"
            exit_abnormal
            ;;
    esac
done

# <<< init ray env >>>
source /mnt/petrelfs/share_data/llm_env/env/llm-torch2.1-flash2
conda activate /mnt/petrelfs/share_data/huangting.p/envs/openrlhf
# <<< init ray env end >>>

RAY_VERSION=$(ray --version)
if [[ ${RAY_VERSION} != 'ray, version 2.12.0' ]]; then
    echo 'Ray version must be 2.12.0!'
    exit 1
fi

# __doc_nvidia_nsight_start__
nsys_cmd=/mnt/petrelfs/caifcicd/dev/nsys/opt/nvidia/nsight-systems/2022.3.4/bin/nsys
nsys_profile_name=uniscale-profile
# __doc_nvidia_nsight_end__

# __doc_head_address_start__
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
else
    head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
RAYLOG=/mnt/petrelfs/share_data/$USER/ray
SYSTEM_CONFIG='{"object_spilling_config":"{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"/mnt/petrelfs/share_data/'$USER'/spill\"}}"}'

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_GPUS_PER_TASK: $SLURM_GPUS_PER_TASK"

function getfreeport()
{
    CHECK="do while"
    while [[ ! -z $CHECK ]]; do
        port=$(( ( RANDOM % 40000 )  + 20000 ))
        CHECK=$(netstat -a | grep $port)
    done
    echo $port
}

dashboard_agent_grpc_port=$(getfreeport)
echo "Dashboard agent GRPC Port: " $dashboard_agent_grpc_port

echo "Starting HEAD at $head_node"
if (( $USE_NSYS == 1 )); then
    # start node with nsys
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
        $nsys_cmd profile -o $nsys_profile_name --wait=all --backtrace=dwarf \
            --capture-range=cudaProfilerApi --capture-range-end=repeat \
            ray start --head --node-ip-address="$head_node_ip" --port=$port \
                --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
                --temp-dir=$RAYLOG \
                --system-config=$SYSTEM_CONFIG \
                --dashboard-agent-grpc-port $dashboard_agent_grpc_port \
                --include-dashboard true --dashboard-host $head_node_ip --dashboard-port 8265 \
                --metrics-export-port=8080 --block &
else
    # start node without nsys
    srun --nodes=1 --ntasks=1 -w "$head_node" \
        --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" \
            --temp-dir=$RAYLOG \
            --system-config=$SYSTEM_CONFIG \
            --dashboard-agent-grpc-port $dashboard_agent_grpc_port \
            --include-dashboard true --dashboard-host $head_node_ip --dashboard-port 8265 \
            --metrics-export-port=8080 --block &
fi
# __doc_head_ray_end__

# __doc_script_start__
echo "End starting"
sleep infinity

# sbatch -p llm_s --exclusive --preempt --job-name=ray-head --cpus-per-task=96 --gpus-per-task=8 -w HOST-10-140-60-16 --nodes=1 ray_on_slurm_head.sh