from noisereduce import reduce_noise as suppress_by_noise_sample

from src.noise_supression.model.enhance import suppress_noise_without_sample

__all__ = ['suppress_by_noise_sample', 'suppress_noise_without_sample']