#!/bin/bash

# Default values
TASK="Isaac-flycrane-payload-hovering-llc-v0"
NUM_ENVS=20480

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --task) TASK="$2"; shift ;;
        --num_envs) NUM_ENVS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the training script
python3 scripts/skrl/train.py --task $TASK --num_envs $NUM_ENVS --headless --seed=-1
echo "Start training from scratch"

# if it crashed, keep restarting
while true; do
    echo "Restarting training..."
    python3 scripts/skrl/train.py --task $TASK --num_envs $NUM_ENVS --headless --seed=-1
done
