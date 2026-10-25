import math

def step(z):
    return 1 if z >= 0 else 0

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def tanh(z):
    return math.tanh(z)

def relu(z):
    return max(0, z)

z = -3

print("Step:", step(z))
print("Sigmoid:", round(sigmoid(z), 3))
print("Tanh:", round(tanh(z), 3))
print("ReLU:", relu(z))