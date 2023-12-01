import re
import matplotlib.pyplot as plt
import sys
import numpy as np

# Check if the log file path is provided as a command line argument
if len(sys.argv) < 3:
    print("Please provide the path to the log file as a command line argument.")
    sys.exit(1)

log_file_path1 = sys.argv[1]
log_file_path2 = sys.argv[2]
gt_file_path = sys.argv[3]

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

data2 = {
    "ID": [],
    "TIME": [],
    "FPS": [],
    "VX": [],
    "VY": [],
    "QUAL": []
}

# Read the log file line by line
with open(log_file_path2, "r") as file:
    for line in file:
        # Extract numbers, Type, and ID from the current line using regular expressions
        numbers = re.findall(number_pattern, line)

        # Remove the plus/minus signs from the numbers
        numbers = [num.replace("+", "") for num in numbers]

        if numbers:
            # Store the extracted values in the dictionary
            data2["ID"].append(int(numbers[0]))
            data2["TIME"].append(int(numbers[1]))
            data2["FPS"].append(float(numbers[2]))
            data2["VX"].append(float(numbers[3]))
            data2["VY"].append(float(numbers[4]))
            data2["QUAL"].append(float(numbers[5]))

gt = {
    "TIME": [],
    "VX": [],
    "VY": [],
}

# Read the log file line by line
with open(gt_file_path, "r") as file:
    for line in file:
        # Extract numbers, Type, and ID from the current line using regular expressions
        numbers = re.findall(number_pattern, line)

        # Remove the plus/minus signs from the numbers
        numbers = [num.replace("+", "") for num in numbers]

        if numbers:
            # Store the extracted values in the dictionary
            gt["TIME"].append(float(numbers[0]))
            gt["VX"].append(float(numbers[1]))
            gt["VY"].append(float(numbers[2]))

# Convert the data to NumPy arrays
data = {key: np.array(value) for key, value in data.items()}
data2 = {key: np.array(value) for key, value in data2.items()}
gt = {key: np.array(value) for key, value in gt.items()}

# Plot the curves
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
#plt.plot(data["Type"], label="Type")
ax1.plot(data["TIME"] - data["TIME"][0]-18660, -data["VX"]*1.0/92.0*42, label="px4_vx")
ax2.plot(data["TIME"] - data["TIME"][0]-18660, -data["VY"]*1.0/92.0*42, label="px4_vy")

for i in range(len(data2["ID"])):
    #print(i,data["TIME"][i+2000])
    data2["TIME"][i] = data["TIME"][i+2000]

print(data2["TIME"].shape,data2["VX"].shape)
ax1.plot(data2["TIME"] - data["TIME"][0]-18660, -data2["VX"]*1.0/92.0*42, label="klt_vx")
ax2.plot(data2["TIME"] - data["TIME"][0]-18660, data2["VY"]*1.0/92.0*42, label="klt_vy")

# ax3.plot(data["TIME"] - data["TIME"][0]-18660, data["QUAL"], label="px4_qual")
# ax3.plot(data2["TIME"] - data2["TIME"][0]-18660, data2["QUAL"], label="openmv_qual")

ax2.plot(gt["TIME"] - gt["TIME"][0], gt["VX"]*33, label="gt_vx")
ax1.plot(gt["TIME"] - gt["TIME"][0], gt["VY"]*33, label="gt_vy")


#plt.plot(data["Rx"], label="Rx")
#plt.plot(data["Ry"], label="Ry")
#plt.plot(data["Rz"], label="Rz")
plt.xlabel("TimeStamps(ms)")
plt.ylabel("Velocities(m/s)")
ax1.legend()
ax2.legend()
#ax3.legend()
plt.show()
