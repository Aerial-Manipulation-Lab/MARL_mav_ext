#!/bin/bash

STUDY_NAME="example"
STORAGE_URL="sqlite:///example.db"
TASK="Isaac-flycrane-payload-hovering-llc-v0"
# Create the study if it doesn't exist
optuna create-study --storage $STORAGE_URL --study-name $STUDY_NAME --direction maximize --skip-if-exists

# Ask for a trial with the specified search space and sampler
TRIAL_OUTPUT=$(optuna ask --storage $STORAGE_URL --study-name $STUDY_NAME --sampler TPESampler \
    --search-space '{
        "learning_rate": {"name": "FloatDistribution", "attributes": {"step": null, "low": 0.00001, "high": 0.01, "log": true}},
        "mini_batches": {"name": "IntDistribution", "attributes": {"step": 1, "low": 2, "high": 8}},
        "lambda": {"name": "FloatDistribution", "attributes": {"step": null, "low": 0.85, "high": 1.0}},
        "rollouts": {"name": "IntDistribution", "attributes": {"step": 1, "low": 12, "high": 32}}
    }')

# Extract the trial parameters
LEARNING_RATE=$(echo $TRIAL_OUTPUT | jq -r '.params.learning_rate')
MINI_BATCHES=$(echo $TRIAL_OUTPUT | jq -r '.params.mini_batches')
LAMBDA=$(echo $TRIAL_OUTPUT | jq -r '.params.lambda')
ROLLOUTS=$(echo $TRIAL_OUTPUT | jq -r '.params.rollouts')

# Run the trial
python3 scripts/skrl/hyper_param_search.py \
    --task $TASK \
    --suggested_lr $LEARNING_RATE \
    --suggested_mini_batches $MINI_BATCHES \
    --suggested_lambda $LAMBDA \
    --suggested_rollouts $ROLLOUTS \
    --headless \
    --num_envs=1