# Noise suppression module

This module contains 2 ways for noise suppression.

# Suppress noise by sample using spectral gating algorithm
You can read more about this algorithm at https://timsainburg.com/noise-reduction-python.html.

#### Usage example:
```python
import noisereduce as nr
import scipy.io

# load data
rate, data = scipy.io.wavfile.read('1.wav')

# select section of data that is noise
noisy_part = data[10000:15000]

# perform noise reduction
reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part)
```

# Suppress noise without sample using pretrained neural network
Used [pretrained model](https://github.com/diff7/Denoising). In case you didn't find model weights in speech_analytics repo, you can download them from [here](https://drive.google.com/file/d/1Ih8pZ3n4i6VXgwKFYQfMWu3PwCiPtgpG/view?usp=sharing).

You can get more information about model's architecture at the [original repo](https://github.com/facebookresearch/denoiser).

#### Usage example:
```python
from pathlib import Path

import soundfile as sf

from src.noise_suppression import NeuralNetworkNoiseSuppressor

# create NeuralNetworkNoiseSuppressor instance (you must specify path to weights)
suppressor = NeuralNetworkNoiseSuppressor(Path(r'model_weights.ckpt'))

# suppress noise and get clean signal
clean = suppressor.suppress(Path(r'1.wav'), 16000)

# you can continue working with clean signal or save it using soundfile
sf.write(r'clean_1.wav', clean, 16000, format='WAV')
```