import re
import matplotlib.pyplot as plt
import sys
import numpy as np

# Check if the log file path is provided as a command line argument
if len(sys.argv) < 2:
    print("Please provide the path to the log file as a command line argument.")
    sys.exit(1)

log_file_path = sys.argv[1]

# Regular expression patterns to extract numbers, Type, and ID
number_pattern = r"[-+]?\d*\.\d+|[-+]?\d+"

# Create a dictionary to store the extracted values
data = {
    "Type": [],
    "ID": [],
    "Tx": [],
    "Ty": [],
    "Tz": [],
    "Rx": [],
    "Ry": [],
    "Rz": []
}

# Read the log file line by line
with open(log_file_path, "r") as file:
    for line in file:
    	if "MarkerInfo" in line:
		# Extract numbers, Type, and ID from the current line using regular expressions
		numbers = re.findall(number_pattern, line)

		# Remove the plus/minus signs from the numbers
		numbers = [num.replace("+", "") for num in numbers]

		if numbers:
		    # Store the extracted values in the dictionary
		    data["Type"].append(int(numbers[0]))
		    data["ID"].append(int(numbers[1]))
		    data["Tx"].append(float(numbers[2])/100)
		    data["Ty"].append(float(numbers[3])/100)
		    data["Tz"].append(-float(numbers[4])/100)
		    data["Rx"].append(float(numbers[5]))
		    data["Ry"].append(float(numbers[6]))
		    data["Rz"].append(float(numbers[7]))

# Convert the data to NumPy arrays
data = {key: np.array(value) for key, value in data.items()}

# Plot the curves
plt.figure(figsize=(10, 6))
#plt.plot(data["Type"], label="Type")
plt.plot(data["ID"], label="ID")
plt.plot(data["Tx"], label="Tx")
plt.plot(data["Ty"], label="Ty")
plt.plot(data["Tz"], label="Tz")
#plt.plot(data["Rx"], label="Rx")
#plt.plot(data["Ry"], label="Ry")
#plt.plot(data["Rz"], label="Rz")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.show()
