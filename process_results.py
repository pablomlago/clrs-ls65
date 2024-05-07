import re
import os
import json

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np



results = {}
# Function to parse command-line arguments into a dictionary.
def parse_command_options(command_parts):
    args_dict = {}
    it = iter(filtered_command_parts)
    for part in it:
        if part.startswith('--'):
            key = part.lstrip('-')
            value = next(it, None)
            args_dict[key] = value
    return args_dict

# Set the path to the directory containing the subfolders
directory_path = 'results_all'
# Set the filename you want to read from each subfolder
filename_to_read = 'slurm_output.out'

# Iterate over all entries in the directory specified
for entry in os.scandir(directory_path):
    if entry.is_dir():  # Check if the entry is a directory (i.e., a subfolder)
        file_path = os.path.join(entry.path, filename_to_read)
        command_line = ""
        try:
            # Attempt to open the file for reading
            with open(file_path, 'r') as file:
                # Read the content of the file
                content = file.read()

                # Use regular expressions to find the Python command line within the log text.
                match = re.search(r'python\s.*', content)
                if match:
                    command_line = match.group()

                else:
                    command_line = ""
                    print("No Python command found in the log.")

        except FileNotFoundError:
            print(f"No file named {filename_to_read} found in {entry.path}")
        except Exception as e:
            print(f"An error occurred while reading from {file_path}: {str(e)}")

        # Break if no python command cound in log
        if command_line == "":
            continue

        # Split the command line into parts and filter out the redirection part (if present).
        command_parts = command_line.split()
        filtered_command_parts = [part for part in command_parts if not part.startswith('>')]
        # Parsing the command-line arguments.
        options_dict = parse_command_options(filtered_command_parts)
        print(options_dict)

        algorithm = options_dict["algorithms"]
        processor = options_dict["processor_type"]
        if algorithm not in results:
            results[algorithm] = {}
        
        if processor not in results[algorithm]:
            results[algorithm][processor] = []

        file_path = os.path.join(entry.path, "slurm_experiment.out")
        command_line = ""
        try:
            # Attempt to open the file for reading
            with open(file_path, 'r') as file:
                # Read the content of the file
                content = file.read()

                # Regex pattern to match the entries within (test) followed by JSON-like dictionary
                pattern = r"\(test\) .* : ({.*})"

                # Find all matches in the log text
                matches = re.findall(pattern, content)

                # Parse the matched dictionary strings into Python dictionaries
                extracted_data = [json.loads(match.replace("'", '"')) for match in matches]

                # Print the results or store them as needed
                for data in extracted_data:
                    results[algorithm][processor].append(data["score"])

        except FileNotFoundError:
            print(f"No file named {filename_to_read} found in {entry.path}")
        except Exception as e:
            print(f"An error occurred while reading from {file_path}: {str(e)}")

print(results)

        

exit()
