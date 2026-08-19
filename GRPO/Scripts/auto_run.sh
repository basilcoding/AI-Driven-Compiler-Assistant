#!/bin/bash

# This script is for lightning-ai interface, it restarts the code if any error occurs
# Activate your uv environment
source ~/env2/bin/activate

# Loop forever
while true; do
    echo "Starting GRPO Training..."
    
    # Run your training script
    REWARD_WORKERS=4 python ~/run_phase4.py
    
    # If the script finishes successfully, break the loop so it doesn't restart forever
    if [ $? -eq 0 ]; then
        echo "Training completed successfully. Exiting."
        break
    fi
    
    echo "🚨 Script crashed or stopped! Restarting in 20 seconds..."
    sleep 20
done