#!/bin/bash
# shellcheck disable=SC2206
#SBATCH --cpus-per-task=80
#SBATCH --gpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --exclusive
#SBATCH --nodes=1
set -e

HEAD_IP=""
CONFIG_FILE_PATH=""
USE_NSYS=0

usage() {
    echo
    echo "Usage: $0 [ -h <head-ip-address> ] [ -f <node-config-filepath> ] [ -p Enable nsys profiling]" 1>&2
}

exit_abnormal() {
    usage
    exit 1
}

while getopts ":h:f:p" flag
do
    case $flag in
        h)
            HEAD_IP=$OPTARG
            ;;
        f)
            CONFIG_FILE_PATH=$OPTARG
            ;;
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

if [[ $HEAD_IP == "" ]]; then
    echo "Error: please input the IP address of the head node."
    exit_abnormal
fi

port=6379
ip_head=$HEAD_IP:$port

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

# __doc_worker_ray_start__
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

RAYLOG=/mnt/petrelfs/share_data/$USER/ray

# <<< parse nodes virtual cluster config file >>>
if [[ $CONFIG_FILE_PATH != "" ]]; then
    echo "Parsing nodes virtual cluster config file $CONFIG_FILE_PATH."
    declare -A nodes_conf_dict

    while read line
    do
        node=`echo $line | awk '{print $1}'`
        label=`echo $line | awk '{print $2}'`
        nodes_conf_dict[${node}]=${label}
    done < $CONFIG_FILE_PATH
fi
# <<< parse nodes virtual cluster config file end >>>

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES))

PREFIX="virtual_cluster"

for ((i = 0; i < worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"

    target_node=${node_i}
    target_label="virtual_cluster_default"

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

    # check node virtual cluster label
    if [[ $CONFIG_FILE_PATH != "" ]]; then
        is_found=0

        for key in $(echo ${!nodes_conf_dict[*]})
        do
            if [ ${key} == ${target_node} ]; then
                is_found=1
                target_label=${nodes_conf_dict[${key}]}

                if [[ "${target_label}" =~ ^"${PREFIX}".* ]]; then
                    echo -e ${target_node} ${target_label}
                    break
                else
                    echo "Error: illegal virtual cluster label '${target_label}', should be formatted as 'virtual_cluster_xxx'."
                    exit 1
                fi
            fi
        done

        if [ ${is_found} == 0 ]; then
            echo "Error: No virtual cluster label for node '${target_node}'."
            exit 1
        fi
    fi

    if (( $USE_NSYS == 1 )); then
        # start node with nsys
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
            $nsys_cmd profile -o $nsys_profile_name --wait=all \
                --capture-range=cudaProfilerApi --capture-range-end=repeat \
                ray start --address "$ip_head" \
                    --num-cpus "${SLURM_CPUS_PER_TASK}" \
                    --dashboard-agent-grpc-port $dashboard_agent_grpc_port \
                    --metrics-export-port=8080 --block &
        sleep 5
    else
        # start node without nsys
        srun --nodes=1 --ntasks=1 -w "$node_i" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gpus-per-task="${SLURM_GPUS_PER_TASK}" \
            ray start --address "$ip_head" \
                --num-cpus "${SLURM_CPUS_PER_TASK}" \
                --dashboard-agent-grpc-port $dashboard_agent_grpc_port \
                --metrics-export-port=8080 --block &
        sleep 5

    fi

done
# __doc_worker_ray_end__

# __doc_script_start__
echo "End starting"
sleep infinity

# sbatch -p llm_s --exclusive --preempt --job-name=ray-compute --cpus-per-task=96 --gpus-per-task=8 --nodes=3 ray_on_slurm_compute.sh -h 10.140.60.16