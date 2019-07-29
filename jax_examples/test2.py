from jax import grad, jit
import jax.numpy as np
import time

def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

# Outputs probability of a label being true according to logistic model.
def logistic_predictions(weights, inputs):
    return sigmoid(np.dot(inputs, weights))

# Training loss is the negative log-likelihood of the training labels.
def loss_real(weights, inputs, targets):
    preds = logistic_predictions(weights, inputs)
    label_logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(label_logprobs).real

def loss_imag(weights, inputs, targets):
    preds = logistic_predictions(weights, inputs)
    label_logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(label_logprobs).imag

def loss(weights, inputs, targets):
    preds = logistic_predictions(weights, inputs)
    label_logprobs = np.log(preds) * targets + np.log(1 - preds) * (1 - targets)
    return -np.sum(label_logprobs)



# Build a toy dataset.
inputs = np.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = np.array([True, True, False, True])


# Define a compiled function that returns gradients of the training loss
training_gradient_fun_cpx = jit(grad(loss,holomorphic=True))
training_gradient_fun_real = jit(grad(loss_real))
training_gradient_fun_imag = jit(grad(loss_imag))

# Optimize weights using gradient descent.
weights = np.array([0.0-1.0j, 0.0+1.0j, 0.0])
print(loss(weights, inputs, targets))
print(loss_real(weights, inputs, targets))
print(loss_imag(weights, inputs, targets))
print()

#print("Initial loss: {:0.2f}".format(loss(weights, inputs, targets)))
ti=time.time()
for i in range(2):
    weights -= 0.1 * training_gradient_fun_cpx(weights, inputs, targets)
    
    print(training_gradient_fun_cpx(weights, inputs, targets))
    print(training_gradient_fun_real(weights, inputs, targets))
    print(training_gradient_fun_imag(weights, inputs, targets))
    print()

    #print("current loss in {0:d}-th epoch: {1:0.4f}".format(i,loss(weights, inputs, targets)))

exit()
tf=time.time()
print("Trained loss: {0:0.2f} took {1:0.4f} secs".format(loss(weights, inputs, targets), tf-ti))

print(weights)





