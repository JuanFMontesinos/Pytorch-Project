import torch
import numpy as np
from torch import nn, istft
from torchaudio.functional import spectrogram
from torchaudio.transforms import InverseMelScale, MelScale


class LogFrequencyScale(torch.nn.Module):
    """
    Rescale linear spectrogram into log-frequency spectrogram. Namely, maps linear values into a log scale
     without modifiying them


    :param warp: True for going from linear to log, false otherwise
    :type warp: bool
    :param ordering: Spectrogram's channel order. By default 'BHWC' (Batch, H,W, channels) according to standard
    pytorch spectrogram ordering --> B,H,W,[real,imag]. Any ordering allowed eg. 'HW', 'HCWB', 'CBWH', 'HBW'
    :type ordering: str
    :param shape: Optional resizing (H_desired, W_desired). (None,None) by default --> without resizing.
    :type shape: tuple
    :param kwargs: Aditional arguments which will be parsed to pytorch's gridsample
    """

    def __init__(self, warp: bool, ordering: str = 'BHWC', shape: tuple = (None, None), adaptative=False, **kwargs):
        super().__init__()
        self.expected_dim = len(ordering)
        self.warp = warp
        self.ordering = ordering.lower()
        self.var = self.get_dims(self.ordering)
        self.instantiated = False
        self.adaptative = adaptative
        self.exp_ordering = ''.join(sorted(self.ordering))  # Expected ordering
        self.kwargs = kwargs
        self.shape = shape

    def needtoinstantiate(self):
        return (not self.instantiated) | self.adaptative

    @staticmethod
    def get_dims(ordering):
        var = {'b': None, 'h': None, 'w': None, 'c': None}
        assert 'h' in ordering
        assert 'w' in ordering
        for key in var:
            var[key] = ordering.find(key)
        return var

    def instantiate(self, sp):
        self.instantiated = True
        dims = sp.shape
        H = dims[self.var['h']] if self.shape[0] is None else self.shape[0]
        W = dims[self.var['w']] if self.shape[1] is None else self.shape[1]
        B = dims[self.var['b']] if self.var['b'] != -1 else 1
        # self.grid = self.get_grid(B, H, W).to(sp.device)
        self.register_buffer('grid', self.get_grid(B, H, W).to(sp.device), persistent=False)
        self.squeeze = []
        if self.var['b'] == -1:
            self.squeeze.append(0)
        if self.var['c'] == -1:
            self.squeeze.append(1)

    def get_grid(self, bs, HO, WO):
        # meshgrid
        x = np.linspace(-1, 1, WO)
        y = np.linspace(-1, 1, HO)
        xv, yv = np.meshgrid(x, y)
        grid = np.zeros((bs, HO, WO, 2))
        grid_x = xv
        if self.warp:
            grid_y = (np.power(21, (yv + 1) / 2) - 11) / 10
        else:
            grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
        grid[:, :, :, 0] = grid_x
        grid[:, :, :, 1] = grid_y
        grid = grid.astype(np.float32)
        return torch.from_numpy(grid)

    def forward(self, sp):
        """
        :param sp: Spectrogram decribed at :class:`.LogFrequencyScale`
        :type sp: torch.Tensor
        :return: Transformed spectrogram
        """
        ufunc = (lambda x: torch.view_as_complex(x.contiguous())) if sp.is_complex() else (lambda x: x)
        if sp.is_complex():
            sp = torch.view_as_real(sp)
        if self.needtoinstantiate():
            self.instantiate(sp)
        sp = torch.einsum(self.ordering + '->' + self.exp_ordering, sp)
        for dim in self.squeeze:
            sp.unsqueeze_(dim)
        sp = torch.nn.functional.grid_sample(sp, self.grid, **self.kwargs)
        for dim in self.squeeze[::-1]:
            sp.squeeze_(dim)
        sp = torch.einsum(self.exp_ordering + '->' + self.ordering, sp)
        return ufunc(sp)


class AudioProcessor(nn.Module):
    _ipipe = {'stft': 'istft', 'lin2log': 'log2lin', 'sp2mel': 'mel2sp', 'mag': 'add_phase'}

    def __init__(self, length, samplerate, n_fft, n_mel, hop_length):
        super(AudioProcessor, self).__init__()
        self._length = int(length)
        self._sr = samplerate
        self._n_fft = n_fft
        self._n_mel = n_mel
        self._hop_length = hop_length
        self._sp_freq_shape = n_fft // 2 + 1

        self.register_buffer('_window', torch.hann_window(self._n_fft).float(), persistent=False)

        self._sp2mel = MelScale(n_mels=self._n_mel, sample_rate=self._sr, n_stft=self._sp_freq_shape,
                                f_max=8000, norm='slaney', mel_scale='slaney')
        self._sp2mel.register_buffer('fb', self._sp2mel.fb, persistent=False)

        self._mel2sp = InverseMelScale(n_mels=self._n_mel, sample_rate=self._sr, n_stft=self._sp_freq_shape)
        self._mel2sp.register_buffer('fb', self._mel2sp.fb, persistent=False)

        self._lin2log = LogFrequencyScale(warp=True, ordering='BHW', shape=(self._sp_freq_shape, None))
        self._log2lin = LogFrequencyScale(warp=False, ordering='BHW', shape=(self._sp_freq_shape, None))

    def stft(self, waveform, **kwargs):
        s = spectrogram(waveform, pad=0, window=self._window, win_length=self._n_fft,
                        n_fft=self._n_fft, hop_length=self._hop_length,
                        power=None, normalized=False, return_complex=True)
        return s

    def istft(self, complex_spectrogram, **kwargs):
        return istft(complex_spectrogram, n_fft=self._n_fft, hop_length=self._hop_length, length=self._length,
                     window=self._window)

    def mag(self, x, **kwargs):
        assert x.is_complex(), 'Spectrogram must be complex'
        return x.abs()

    def add_phase(self, mag, **kwargs):
        phase = kwargs['phase']
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.view_as_complex(torch.stack([real, imag], dim=-1).contiguous())

    def lin2log(self, x, **kwargs):
        return self._lin2log(x)

    def log2lin(self, x, **kwargs):
        return self._log2lin(x)

    def sp2mel(self, x, **kwargs):
        return self._sp2mel(x)

    def mel2sp(self, x, **kwargs):
        with torch.enable_grad():
            return self._mel2sp(x)

    def process_sample(self, sample, funcs, **kwargs):
        chain = {}
        for f in funcs:
            sample = getattr(self, f)(sample, **kwargs)
            chain[f] = sample
            if f == 'stft':
                chain['phase'] = sample.angle()

        return sample, chain
