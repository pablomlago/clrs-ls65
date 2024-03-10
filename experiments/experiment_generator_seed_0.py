import json
from subprocess import call, run
import os
from jinja2 import Template
from pathlib import Path

###################################################
# Only need to change this line experiments_to run
###################################################
experiments_to_run = list(range(1,103))
test_flag = "False"
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
datasets_folder = "./datasets_0"
# Checkpoints folder
checkpoints_folder = "./results_0"
# Template file that needs to be generated for each experiment
slurm_template_file = f"{experiments_folder}/slurm_l65_gpu_template_seed_0"
# Output file
slurm_output_file_name = "slurm_l65_gpu_template_experiment"
# File containing the configurations for the different experiments
experiment_file_name = f"{experiments_folder}/experiments.json"

# Read the template file
with open(experiment_file_name, "r") as experiment_file:

    json_experiments = json.load(experiment_file)

    for experiment_id, experiment_args in json_experiments.items():
        # Skip processing experiments which are not intended to be run
        if int(experiment_id) not in experiments_to_run:
            continue
        experiment_options = (f"--checkpoint_path {checkpoints_folder}/{experiment_id} "
                              f"--dataset_path {datasets_folder}/{experiment_id} "
                              f"--seed {43}"
                              f"--test {test_flag}"
                              )
        for key, value in experiment_args.items():
            if type(value) == bool:
                if value:
                    experiment_options += f"--{key} "
            elif type(value) == list:
                if len(value) > 0:
                    experiment_options += f"--{key} {' '.join(map(str, value))} "
            else:
                experiment_options += f"--{key} {value} "

        # Remove trailing whitespace
        experiment_options = experiment_options[:-1]

        # Read the template file
        with open(slurm_template_file, "r") as slurm_file:
            slurm_template_content = slurm_file.read()

        # Create a Jinja template object
        template = Template(slurm_template_content)

        # Render the template with the variable value
        rendered_template = template.render(
            experiment_options=experiment_options, 
            experiment_id=experiment_id,
            CRSid=CRSid,
            conda_environment=conda_environment,
        )

        # Write the rendered template to a new file or use it as needed
        with open(f"{experiments_folder}/{slurm_output_file_name}_{experiment_id}", "w") as slurm_output_file:
            slurm_output_file.write(rendered_template)

# Full path to the new "results" folder
results_path = Path(os.getcwd()) / 'results_0'
# Check if the new directory exists, and if not, create it
if not results_path.exists():
    results_path.mkdir()

for experiment_id in experiments_to_run:
    results_experiment_path = results_path / str(experiment_id)
    if not results_experiment_path.exists():
        results_experiment_path.mkdir()

# Full path to the new "datasets" folder
datasets_path = Path(os.getcwd()) / 'datasets_0'
# Check if the new directory exists, and if not, create it
if not datasets_path.exists():
    datasets_path.mkdir()

for experiment_id in experiments_to_run:
    datasets_experiment_path = datasets_path / str(experiment_id)
    if not datasets_experiment_path.exists():
        datasets_experiment_path.mkdir()

# Iterate over the experiments running them
for experiment_id in experiments_to_run:
    print(f'Running experiment with ID: {experiment_id}')
    # Command to execute experiment
    call(["sbatch", f"{experiments_folder}/{slurm_output_file_name}_{experiment_id}"])


