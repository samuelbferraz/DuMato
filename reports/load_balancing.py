import sys
import matplotlib.pyplot as plt
import numpy as np

log_file = sys.argv[1]

fp = open(log_file, "r")
lines = fp.readlines()
lines_filtered = [line.strip() for line in lines if "REPORT" in line]

time = 0
for line in lines:
    if "Time (GPU)" in line:
        time = float(line.split(":")[1].strip().split(" ")[0])
        break
print(time)

y_axis = []
for line in lines_filtered:
    tokens = line.split(",")
    amount_warps = int(tokens[3])
    y_axis.append(amount_warps)

print()

x_axis = np.arange(0,len(lines_filtered))


plt.xticks(x_axis)
plt.xlabel("Time (total: " + str(time) + ")")
plt.ylabel("Active warps")
plt.plot(x_axis, y_axis)
plt.show()


fp.close()