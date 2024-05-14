import json
from subprocess import call, run
import os
from jinja2 import Template
from pathlib import Path

###################################################
# Only need to change this line experiments_to run
###################################################
algorithms = [
    "bfs",
    "dfs",
    "bellman_ford",
    "dijkstra",
    "bubble_sort",
    "activity_selector",
    "mst_prim",
    "task_scheduling",
    "topological_sort", 
    "find_maximum_subarray",
    "quickselect",
]
architectures = {
    "mpnn_l1_l3": {
        "regularisation_weight_l2": 1.0,
        "regularisation_weight_l3": 0.0,
    },
    "mpnn_l1_l3_max": {
        "regularisation_weight_l2": 1.0,
        "regularisation_weight_l3": 0.0,
    },
    "mpnn_l2_l3": {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.1,
    }, 
}
"""
architectures = {
    "mpnn_l1": {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.0,
    },
     "mpnn_l1_max": {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.0,
    },
    "mpnn_l2": {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.0,
    }, 
    "mpnn_l2_l3_max": {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.5,
    }, 
    "mpnn_l3" : {
        "regularisation_weight_l2": 0.0,
        "regularisation_weight_l3": 0.0,
    },
    "mpnn_l1_l3": {
        "regularisation_weight_l2": 0.5,
        "regularisation_weight_l3": 0.5,
    }, 
    "mpnn_l1_l3_max": {
        "regularisation_weight_l2": 0.5,
        "regularisation_weight_l3": 0.5,
    },
}
"""
test_flag = False
###################################################
# No need to change anything below here
###################################################

# Retrieve CRSid
CRSid = os.environ.get('USER')
# Get the name of the currently active conda environment
conda_environment = os.environ.get('CONDA_DEFAULT_ENV')
# Experiments folder
experiments_folder = "./experiments"
# Datasets folder
datasets_folder = "./datasets"
# Checkpoints folder
checkpoints_folder = "./results"
# Template file that needs to be generated for each experiment
slurm_template_file = f"{experiments_folder}/slurm_l65_gpu_template_pca"
# Output file
slurm_output_file_name = "slurm_l65_gpu_template_experiment"
# File containing the configurations for the different experiments
experiment_file_name = f"{experiments_folder}/experiments_pca.json"

# Experiment counter
initial_experiment_id = 400
experiment_id_counter = initial_experiment_id
# Iterate over experiments
for architecture_name, architecture_params in architectures.items():
    print(architecture_params)
    # Iterate over algorithms
    for seed in [42, 43, 44]:
        for algorithm in algorithms:
            experiment_options = (f"--checkpoint_path {checkpoints_folder}/{experiment_id_counter} "
                                f"--dataset_path {datasets_folder}/{experiment_id_counter} "
                                f"--processor_type {architecture_name} "
                                f"--algorithms {algorithm} "
                                f"--regularisation_weight_l2 {architecture_params['regularisation_weight_l2']} "
                                f"--regularisation_weight_l3 {architecture_params['regularisation_weight_l3']} "
                                f"--seed {seed}"
                                )

            # Read the template file
            with open(slurm_template_file, "r") as slurm_file:
                slurm_template_content = slurm_file.read()

            # Create a Jinja template object
            template = Template(slurm_template_content)

            # Render the template with the variable value
            rendered_template = template.render(
                experiment_options=experiment_options, 
                experiment_id=experiment_id_counter,
                CRSid=CRSid,
                conda_environment=conda_environment,
                test="_test" if test_flag else "",
            )

            # Write the rendered template to a new file or use it as needed
            with open(f"{experiments_folder}/{slurm_output_file_name}_{experiment_id_counter}", "w") as slurm_output_file:
                slurm_output_file.write(rendered_template)

            experiment_id_counter += 1

# Full path to the new "results" folder
results_path = Path(os.getcwd()) / 'results'
# Check if the new directory exists, and if not, create it
if not results_path.exists():
    results_path.mkdir()

for experiment_id in range(initial_experiment_id, experiment_id_counter):
    results_experiment_path = results_path / str(experiment_id)
    if not results_experiment_path.exists():
        results_experiment_path.mkdir()

# Full path to the new "datasets" folder
datasets_path = Path(os.getcwd()) / 'datasets'
# Check if the new directory exists, and if not, create it
if not datasets_path.exists():
    datasets_path.mkdir()

for experiment_id in range(initial_experiment_id, experiment_id_counter):
    datasets_experiment_path = datasets_path / str(experiment_id)
    if not datasets_experiment_path.exists():
        datasets_experiment_path.mkdir()

# Iterate over the experiments running them
for experiment_id in range(initial_experiment_id, experiment_id_counter):
    print(f'Running experiment with ID: {experiment_id}')
    # Command to execute experiment
    #call(["sbatch", f"{experiments_folder}/{slurm_output_file_name}_{experiment_id}"])


