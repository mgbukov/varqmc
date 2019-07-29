import time
import itertools

import numpy.random as npr

import jax.numpy as np
#from jax.config import config
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import Dense, Relu, LogSoftmax
from examples import datasets


def loss(params, batch):
	inputs, targets = batch
	preds = predict(params, inputs)
	return -np.mean(preds * targets)

def accuracy(params, batch):
	inputs, targets = batch
	target_class = np.argmax(targets, axis=1)
	predicted_class = np.argmax(predict(params, inputs), axis=1)
	return np.mean(predicted_class == target_class)

# define DNN
init_random_params, predict = stax.serial(
											Dense(1024), Relu,
											Dense(1024), Relu,
											Dense(10), LogSoftmax
										)


if __name__ == "__main__":
	# fix seed
	rng = random.PRNGKey(0)

	# optimization hyperparams
	step_size = 0.001
	num_epochs = 10
	batch_size = 128
	momentum_mass = 0.9

	# import data and compute minibatch training sizes
	train_images, train_labels, test_images, test_labels = datasets.mnist()
	num_train = train_images.shape[0]
	num_complete_batches, leftover = divmod(num_train, batch_size)
	num_batches = num_complete_batches + bool(leftover)

	# define imnibatch generator
	def data_stream():
		rng = npr.RandomState(0)
		while True:
			perm = rng.permutation(num_train)
			for i in range(num_batches):
				batch_idx = perm[i * batch_size:(i + 1) * batch_size]
				yield train_images[batch_idx], train_labels[batch_idx]
	batches = data_stream()

	# define optimizer
	opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)


	# updates model parameters
	@jit
	def update(i, opt_state, batch):
		params = get_params(opt_state)
		return opt_update(i, grad(loss)(params, batch), opt_state)


	# define initial parameters
	_, init_params = init_random_params(rng, (-1, 28 * 28))
	# pass parameters to optimizer
	opt_state = opt_init(init_params)
	# set counter
	itercount = itertools.count()

	# start training
	print("\nStarting training...")
	for epoch in range(num_epochs):
		start_time = time.time()
		for _ in range(num_batches):
			opt_state = update(next(itercount), opt_state, next(batches))
		epoch_time = time.time() - start_time

		# evaluate training epoch
		params = get_params(opt_state)
		train_acc = accuracy(params, (train_images, train_labels))
		test_acc = accuracy(params, (test_images, test_labels))
		print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
		print("Training set accuracy {}".format(train_acc))
		print("Test set accuracy {}".format(test_acc))


