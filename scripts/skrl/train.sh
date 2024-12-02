#!/bin/bash

# Default values
TASK="Isaac-flycrane-payload-hovering-llc-v0"
NUM_ENVS=20480
RESUME=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --resume) RESUME=true; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --num_envs) NUM_ENVS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the training script
if [ "$RESUME" = true ]; then
    python3 scripts/skrl/train.py --task $TASK --num_envs $NUM_ENVS --resume --checkpoint $CHECKPOINT --headless --seed=-1
else
    python3 scripts/skrl/train.py --task $TASK --num_envs $NUM_ENVS --headless --seed=-1
    echo "Start training from scratch"
fi

# if it crashed, keep restarting with resume from the last checkpoint
while true; do
    echo "Restarting training..."
    python3 scripts/skrl/train.py --task $TASK --num_envs $NUM_ENVS --resume --headless --seed=-1
done
