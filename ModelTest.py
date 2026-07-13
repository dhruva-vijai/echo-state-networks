import ESN
import numpy as np
import matplotlib.pyplot as plt 
import math

model = ESN.model(2, 1000, 2, 0.99, 20, 0.0125, 0.2)
model.initialize()


x_values = np.linspace(0, 10, 5001)
raw_data = np.column_stack((x_values, np.sin(3*x_values)+np.sin(x_values**2)))

data_min = raw_data.min(axis=0)
data_max = raw_data.max(axis=0)
scaled_data = (raw_data - data_min) / (data_max - data_min)

input_sequence = scaled_data[:-1]
teacher_sequence = scaled_data[1:]

transient = 300
model.fit(5000, transient, input_sequence, teacher_sequence)

test_input = [row for row in input_sequence]
comparison_teacher = teacher_sequence[transient:]

results,mse,mselist = (model.evaluation(5000, transient,1000,input_sequence,teacher_sequence))

print(mse)

plt.figure(figsize=(10, 5))
plt.plot(mselist)
plt.xlabel("Time Step / Index")
plt.ylabel("MSE")
plt.title("MSE over Time")
plt.show()

logmse = [math.log10(x) for x in mselist]
plt.figure(figsize=(10, 5))
plt.plot(logmse)
plt.xlabel("Time Step / Index")
plt.ylabel("MSE")
plt.title("MSE over Time")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(comparison_teacher[:, 1], label="Target", color="black", lw=2)

plt.axvline(x=1000-transient, color='blue', linestyle=':', label='End of Teacher Forcing')
plt.axvline(x=transient, color='gray', linestyle=':', label='Transient Behaviour')

plt.plot(results[:, 1], label="ESN Prediction", color="red", linestyle="--")
plt.title("ESN Next-Step Prediction")
plt.legend()
plt.show()

print("Results shape:", results.shape)

