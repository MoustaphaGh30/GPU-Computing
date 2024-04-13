import numpy as np
import matplotlib.pyplot as plt

# # Input sizes and corresponding running times
# inputs =    [8,     64, 512, 1024, 2048, 3000]
# cpu_times = [3.38, 18.72, 286.67, 575.97, 1160.98, 1710]  # CPU running times in milliseconds
# gpu_times = [0.72, 0.73, 3.38, 6.64, 12.27, 17]  # GPU running times in milliseconds


# # Convert input sizes to log base 8
# log_inputs = np.log(inputs) / np.log(8)

# # Plotting
# plt.plot(log_inputs, cpu_times, label='CPU')
# plt.plot(log_inputs, gpu_times, label='GPU')
# plt.xlabel('Log base 8 of Input Size')
# plt.ylabel('Running Time (ms)')
# plt.title('Running Time vs. Log Base 8 of Input Size')
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# # Input sizes and corresponding running times
# inputs =    [8,     64, 512, 1024, 2048, 3000]
# cpu_times = [3.38, 18.72, 286.67, 575.97, 1160.98, 1710]  # CPU running times in milliseconds
# gpu_times = [0.72, 0.73, 3.38, 6.64, 12.27, 17]  # GPU running times in milliseconds

# # Convert input sizes to log base 8
# log_inputs = np.log(inputs) / np.log(8)

# # Plotting
# plt.plot(log_inputs, cpu_times, label='CPU')
# plt.plot(log_inputs, gpu_times, label='GPU')
# plt.xlabel('Log base 8 of Input Size')
# plt.ylabel('Running Time (ms)')
# plt.title('Running Time vs. Log Base 8 of Input Size')
# plt.legend()
# plt.grid(True)

# # Set y-axis limits to focus on GPU times
# plt.show()
# Input sizes and corresponding running times
inputs =    [8,     64, 512, 1024, 2048, 3000]
gpu_times = [0.55, 0.56, 3.13, 5.46, 10.12, 14.79]  # CPU running times in milliseconds
cpu_times = [0.72, 0.73, 3.38, 6.64, 12.27, 17]  # GPU running times in milliseconds

# Convert input sizes to log base 8
log_inputs = np.log(inputs) / np.log(8)

# Plotting
plt.plot(log_inputs, cpu_times, label='GPU with shared memory and control div')
plt.plot(log_inputs, gpu_times, label='GPU with shared memory and no control div')
plt.xlabel('Log base 8 of Input Size')
plt.ylabel('Running Time (ms)')
plt.title('No control divergence')
plt.legend()
plt.grid(True)

# Set y-axis limits to focus on GPU times
plt.show()

