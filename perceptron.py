import numpy as np

# Hyperparameter
learning_rate = 0.02

bias = 2
n = 10
k = 3
original_weights = np.array([-0.1, -0.3, 0.2]).reshape(3, 1)

X_transpose = np.array([6, -5, -5, -11, -5, 5, 2, 4, 9, -8])
X_transpose = np.vstack((X_transpose, [1, 6, -11, 3, -2, -6, 0, -9, -7, 4]))
X_transpose = np.vstack((X_transpose, [-11, 7, 1, 5, 1, -7, 2, -11, -9, -8]))

X = X_transpose.T
T = np.array([0, 1, 0, 0, 1, 1, 1, 0, 0, 1]).reshape(10, 1)

Y = X @ original_weights + bias

weight_diff = (X.T @ (Y - T)) / n
weights = original_weights - learning_rate * weight_diff

bias_diff = (np.ones(n) @ (Y - T)) / n
bias -= learning_rate * bias_diff

print(f"Original weights: \n{original_weights}")
print(f"Original bias: {2}")
print(f"Weights gradient descent : \n{weights.round(2)}")
print(f"Bias after one step of gradient descent : {bias.round(2)}")
