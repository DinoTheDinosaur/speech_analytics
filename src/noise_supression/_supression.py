import noisereduce as nr


def suppress_by_sample(audio, noise=None, **kwargs):
    return nr.reduce_noise(audio, noise, **kwargs)


def suppress_without_sample(audio):
    pass
