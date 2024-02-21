import matplotlib.pyplot as plt

# Path to the log file
log_file_path = 'experiments/sample_results.txt'

# Initialize a dictionary to hold lists of values for each entry
data_dict = {
    "regularised_loss": [],
    "regularisation_weight": [],
    "mse_loss": [],
    "quality_loss": []
}

# Read log lines from the file
with open(log_file_path, 'r') as file:
    log_lines = file.readlines()

# Process each log line
for line in log_lines:
    # Remove newline characters and any surrounding whitespace
    line = line.strip()
    # Split the line into entries
    entries = line.split(" ")
    if line.startswith("[DEBUG]"):
    # Set entries
        data_dict["regularised_loss"].append(float(entries[3][:-1]))
        data_dict["regularisation_weight"].append(float(entries[6][:-1]))
        data_dict["mse_loss"].append(float(entries[9][:-1]))
        data_dict["quality_loss"].append(float(entries[12][:-1]))

# Plotting
plt.figure(figsize=(10, 6))

# Plot "Quality loss"
plt.plot(data_dict["quality_loss"], label="Quality Loss")

# Plot "Regularised loss"
plt.plot(data_dict["regularised_loss"], label="Regularised Loss")

plt.title('Quality Loss vs Regularised Loss')
plt.xlabel('Entry Index')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True)
plt.show()
