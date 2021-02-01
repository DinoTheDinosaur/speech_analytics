import librosa
import scipy


class FeatureExtractor:
    def __init__(self, audio, window_len, overlap, sample_rate) -> None:
        self.__audio = audio
        self.__fft_len = window_len
        self.__window_len = window_len
        self.__overlap = overlap
        self.__sample_rate = sample_rate
        self.__window = scipy.signal.hamming(self.__window_len, sym=False)

    def get_stft_spectrogram(self):
        return librosa.stft(self.__audio, n_fft=self.__fft_len, win_length=self.__window_len, hop_length=self.__overlap,
                            window=self.__window, center=True)

    def get_audio_from_stft_spectrogram(self, stft_features):
        return librosa.istft(stft_features, win_length=self.__window_len, hop_length=self.__overlap,
                             window=self.__window, center=True)

    def get_mel_spectrogram(self):
        return librosa.feature.melspectrogram(self.__audio, sr=self.__sample_rate, n_fft=self.__fft_len,
                                              hop_length=self.__overlap, window=self.__window, center=True)

    def get_audio_from_mel_spectrogram(self, mel):
        return librosa.feature.inverse.mel_to_audio(mel, sr=self.__sample_rate, n_fft=self.__fft_len,
                                                    hop_length=self.__overlap, win_length=self.__window_len,
                                                    window=self.__window, center=True, n_iter=32, length=None)
