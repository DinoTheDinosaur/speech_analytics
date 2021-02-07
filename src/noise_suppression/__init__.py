from noisereduce import reduce_noise as suppress_by_noise_sample

from src.noise_suppression.nn.nn_noise_suppressor import NeuralNetworkNoiseSuppressor

__all__ = ['suppress_by_noise_sample', 'NeuralNetworkNoiseSuppressor']
