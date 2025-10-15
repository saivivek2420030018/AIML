import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(100)

# Generate random training data
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Add bias term (x₀ = 1)
x_b = np.c_[np.ones((100, 1)), x]

# Calculate theta_best using Normal Equation
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

# Predict new values
x_new = np.array([[0], [2]])
x_new_b = np.c_[np.ones((2, 1)), x_new]
y_predict = x_new_b.dot(theta_best)

# Display results
print("Optimal parameters (theta):")
print(theta_best)

print("\nPredicted values:")
print(y_predict)

# Plot training data and regression line
plt.scatter(x, y, color='blue', label='Training data')
plt.plot(x_new, y_predict, color='red', linewidth=2, label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression from Scratch')
plt.show()