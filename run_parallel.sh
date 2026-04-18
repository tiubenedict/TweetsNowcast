#!/bin/bash

# --- Configuration ---
CONDA_ENV_NAME="/home/btiu/Documents/Research/TweetsNowcast/.conda" # <--- IMPORTANT: Replace with your actual Conda environment name
# PATH_TO_SCRIPT="run_singletask_args.py" # <--- IMPORTANT: Replace with the actual name of your Ray Tune script
# --- End Configuration ---

# Source Conda's initialization script if not already done in your shell's profile
# This makes the 'conda' command available.
# You might need to adjust the path based on your Conda installation.
# A common location for miniconda3 is ~/miniconda3/etc/profile.d/conda.sh
# If you've already run 'conda init <shell>' for your shell, this might not be strictly necessary
# but it's good practice for standalone scripts.
# if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
#     source "$(conda info --base)/etc/profile.d/conda.sh"
# else
#     echo "Warning: conda.sh not found. Ensure Conda is initialized in your shell."
#     echo "You may need to run 'conda init' or adjust the source path."
# fi
source /home/btiu/miniconda3/etc/profile.d/conda.sh

echo "Starting parallel Ray Tune runs..."

# Function to run a single experiment chunk
run_experiment() {
    local start_month=$1
    local end_month=$2
    local script_name=$3
    
    echo "Activating Conda environment: $CONDA_ENV_NAME"
    conda activate "$CONDA_ENV_NAME" || { echo "Failed to activate Conda environment '$CONDA_ENV_NAME'. Exiting."; exit 1; }

    echo "Launching experiment for $start_month to $end_month..."
    # The 'exec' command here replaces the current shell process with the python process,
    # which can be slightly more efficient, though 'python ...' without exec also works.
    python "$script_name" --start_month "$start_month" --end_month "$end_month"
    
    # Deactivate is generally not needed when using '&' for background processes
    # or when using 'exec', as the environment is scoped to the subshell/process.
    # However, if you were running them sequentially in a single shell, you'd deactivate.
    # conda deactivate
}

# Export the function so it can be used with `&` in subshells
export -f run_experiment

# run_experiment "2017-01-31" "2017-02-28" "run_singletask_args.py" &
# run_experiment "2017-03-31" "2017-04-30" "run_singletask_args.py" &
# run_experiment "2017-05-31" "2017-06-30" "run_singletask_args.py" &
# run_experiment "2017-07-31" "2017-08-31" "run_singletask_args.py" &
# run_experiment "2017-09-30" "2017-10-31" "run_singletask_args.py" &
# run_experiment "2017-11-30" "2017-12-31" "run_singletask_args.py" &

# run_experiment "2017-01-31" "2017-12-31" "run_singletask_args.py" &
# run_experiment "2018-01-31" "2018-12-31" "run_singletask_args.py" &
# run_experiment "2019-01-31" "2019-12-31" "run_singletask_args.py" &
# run_experiment "2020-01-31" "2020-12-31" "run_singletask_args.py" &
# run_experiment "2021-01-31" "2021-12-31" "run_singletask_args.py" &
# run_experiment "2022-01-31" "2022-12-31" "run_singletask_args.py" &

run_experiment "2017-01-31" "2017-12-31" "run_multitask_args.py" &
run_experiment "2018-01-31" "2018-12-31" "run_multitask_args.py" &
run_experiment "2019-01-31" "2019-12-31" "run_multitask_args.py" &
run_experiment "2020-01-31" "2020-12-31" "run_multitask_args.py" &
run_experiment "2021-01-31" "2021-12-31" "run_multitask_args.py" &
run_experiment "2022-01-31" "2022-12-31" "run_multitask_args.py" &

# run_experiment "2017-01-31" "2017-04-30" "run_singletask_args.py" &
# run_experiment "2017-05-31" "2017-08-31" "run_singletask_args.py" &
# run_experiment "2017-09-30" "2017-12-31" "run_singletask_args.py" &
# run_experiment "2018-01-31" "2018-04-30" "run_singletask_args.py" &
# run_experiment "2018-05-31" "2018-08-31" "run_singletask_args.py" &
# run_experiment "2018-09-30" "2018-12-31" "run_singletask_args.py" &

# run_experiment "2017-01-31" "2017-04-30" "run_multitask_args.py" &
# run_experiment "2017-05-31" "2017-08-31" "run_multitask_args.py" &
# run_experiment "2017-09-30" "2017-12-31" "run_multitask_args.py" &
# run_experiment "2018-01-31" "2018-04-30" "run_multitask_args.py" &
# run_experiment "2018-05-31" "2018-08-31" "run_multitask_args.py" &
# run_experiment "2018-09-30" "2018-12-31" "run_multitask_args.py" &



# Wait for all background jobs to complete
wait

echo "All parallel Ray Tune runs completed."