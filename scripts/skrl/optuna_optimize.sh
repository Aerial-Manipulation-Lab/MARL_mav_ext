#!/bin/bash

# Default values
STUDY_NAME="example"
TASK="Isaac-flycrane-payload-hovering-llc-v0"
NUM_ENVS=32768
TIMESTEPS=150000
NUM_TRIALS=100

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --study_name) STUDY_NAME="$2"; shift ;;
        --task) TASK="$2"; shift ;;
        --num_envs) NUM_ENVS="$2"; shift ;;
        --timesteps) TIMESTEPS="$2"; shift ;;
        --num_trials) NUM_TRIALS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

STORAGE_URL="sqlite:///$STUDY_NAME.db"

# Create the study if it doesn't exist
optuna create-study --storage $STORAGE_URL --study-name $STUDY_NAME --direction maximize --skip-if-exists

for ((i=1; i<=NUM_TRIALS; i++)); do
    # Ask for a trial with the specified search space and sampler
    SUGGESTED_VALUES=$(optuna ask --storage $STORAGE_URL --study-name $STUDY_NAME --sampler TPESampler \
        --search-space '{
            "learning_rate": {"name": "FloatDistribution", "attributes": {"step": null, "low": 0.00001, "high": 0.1, "log": true}},
            "mini_batches": {"name": "IntDistribution", "attributes": {"step": 1, "low": 1, "high": 8}},
            "lambda": {"name": "FloatDistribution", "attributes": {"step": null, "low": 0.2, "high": 1.0}}
        }')

    # Extract the trial parameters
    LEARNING_RATE=$(echo $SUGGESTED_VALUES | jq -r '.params.learning_rate')
    MINI_BATCHES=$(echo $SUGGESTED_VALUES | jq -r '.params.mini_batches')
    LAMBDA=$(echo $SUGGESTED_VALUES | jq -r '.params.lambda')
    ROLLOUTS=$(echo $SUGGESTED_VALUES | jq -r '.params.rollouts')
    TRIAL_ID=$(echo "$SUGGESTED_VALUES" | jq -r '.number')

    # Run the trial
    TRIAL_OUTPUT=$(python3 scripts/skrl/hyper_param_search.py \
        --task $TASK \
        --suggested_lr $LEARNING_RATE \
        --suggested_mini_batches $MINI_BATCHES \
        --suggested_lambda $LAMBDA \
        --headless \
        --num_envs=$NUM_ENVS \
        --seed=-1 \
        --timesteps=$TIMESTEPS)

    FINAL_REWARD=$(echo "$TRIAL_OUTPUT" | grep -oP 'Mean rewards are \K\d+(\.\d+)?')
    echo "The final reward is $FINAL_REWARD"

    if [[ -z "$FINAL_REWARD" ]]; then
        echo "$TRIAL_OUTPUT"
    fi

    # Report the trial results
    optuna tell --storage $STORAGE_URL --study-name $STUDY_NAME --trial-number $TRIAL_ID --values $FINAL_REWARD --state complete
done
