import json
from subprocess import call, run
import os
from jinja2 import Template
from pathlib import Path

from lsr.experiment import run_experiment

###################################################
# Only need to change this line experiments_to run
###################################################
experiments_to_run = [0, 1, 2, 3, 4, 5]
###################################################
# No need to change anything below here
###################################################

# Experiments folder
experiments_folder = "./experiments"
# Checkpoints folder
checkpoints_folder = "./results"
# File containing the configurations for the different experiments
experiment_file_name = f"{experiments_folder}/experiments_pca.json"

# Read the template file
with open(experiment_file_name, "r") as experiment_file:

    json_experiments = json.load(experiment_file)

    for experiment_id, experiment_args in json_experiments.items():
        # Skip processing experiments which are not intended to be run
        if int(experiment_id) not in experiments_to_run:
            continue
        
        # Iterating over the JSON is not strictly needed but it's just in case some configuration
        # needs to be extracted from it.
        run_experiment(path=f"{checkpoints_folder}/{experiment_id}/")