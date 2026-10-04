#!/bin/bash
# Print up to N GPU indices nobody is using (no compute process, <1 GiB used),
# comma-separated. Exit 1 if fewer than N are free. Shared lab box: never
# land on a GPU someone else holds, even if it looks idle.
#
#   GPUS=$(scripts/free_gpus.sh 3) || { echo "not enough free GPUs"; exit 1; }
set -euo pipefail
N=${1:?usage: free_gpus.sh N}
busy=$(nvidia-smi --query-compute-apps=gpu_bus_id --format=csv,noheader | sort -u)
free=()
while IFS=', ' read -r idx bus mem; do
    if [ "$mem" -lt 1024 ] && ! grep -qi "$bus" <<<"$busy"; then
        free+=("$idx")
    fi
done < <(nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader,nounits)
if [ "${#free[@]}" -lt "$N" ]; then
    echo "only ${#free[@]} free GPU(s): ${free[*]:-none}" >&2
    exit 1
fi
(IFS=,; echo "${free[*]:0:$N}")
