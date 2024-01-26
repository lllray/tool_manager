import re
import matplotlib.pyplot as plt
import sys
import numpy as np

# Check if the log file path is provided as a command line argument
if len(sys.argv) < 2:
    print("Please provide the path to the log file as a command line argument.")
    sys.exit(1)

log_file_path1 = sys.argv[1]

# Regular expression patterns to extract numbers, Type, and ID
number_pattern = r"[-+]?\d*\.\d+|[-+]?\d+"

# Create a dictionary to store the extracted values
data = {
    "ID": [],
    "TIME": [],
    "FPS": [],
    "VX": [],
    "VY": [],
    "QUAL": []
}

# Read the log file line by line
with open(log_file_path1, "r") as file:
    for line in file:
        # Extract numbers, Type, and ID from the current line using regular expressions
        numbers = re.findall(number_pattern, line)

        # Remove the plus/minus signs from the numbers
        numbers = [num.replace("+", "") for num in numbers]

        if numbers:
            # Store the extracted values in the dictionary
            data["ID"].append(int(numbers[0]))
            data["TIME"].append(int(numbers[1]))
            data["FPS"].append(float(numbers[2]))
            data["VX"].append(float(numbers[3]))
            data["VY"].append(float(numbers[4]))
            data["QUAL"].append(float(numbers[5])/255)

# Convert the data to NumPy arrays
data = {key: np.array(value) for key, value in data.items()}

# Plot the curves
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6),sharex=True)
#plt.plot(data["Type"], label="Type")
ax1.plot(data["TIME"], -data["VX"], label="vx")
ax2.plot(data["TIME"], -data["VY"], label="vy")
ax3.plot(data["TIME"], data["QUAL"], label="qual")

plt.xlabel("TimeStamps(ms)")
plt.ylabel("Velocities(m/s)")
ax1.legend()
ax2.legend()
#ax3.legend()
plt.show()
