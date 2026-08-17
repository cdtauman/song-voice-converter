"""RMVPE network and inference -- vendored from RVC (MIT).

Ported from RVC-Project/Retrieval-based-Voice-Conversion-WebUI, `infer/rmvpe.py`
(commit 81eed5e, MIT). The architecture classes are copied verbatim because the
pretrained `rmvpe.pt` was trained against these exact module names and shapes --
renaming a layer breaks `load_state_dict`. What was removed is everything that
tied the original to its host app: the fp16 path, the DirectML/ONNX branch, the
CUDA-graph capture, and the `configs.config` import. See docs/third-party.md.

This module imports torch and librosa and must only ever be imported lazily,
from `analysis.f0` -- CI installs neither.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel

__all__ = ["RmvpeModel", "E2E"]


class BiGRU(nn.Module):
    def __init__(self, input_features: int, hidden_features: int, num_layers: int) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, momentum: float = 0.01) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] | None,
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList([ConvBlockRes(in_channels, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor):  # type: ignore[no-untyped-def]
        for conv in self.conv:
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_size: int,
        n_encoders: int,
        kernel_size: tuple[int, int],
        n_blocks: int,
        out_channels: int = 16,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _ in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(in_channels, out_channels, kernel_size, n_blocks, momentum)
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):  # type: ignore[no-untyped-def]
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_inters: int,
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList(
            [ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)]
        )
        for _ in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int],
        n_blocks: int = 1,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, (3, 3), stride, (1, 1),
                output_padding=out_padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList([ConvBlockRes(out_channels * 2, out_channels, momentum)])
        for _ in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x: torch.Tensor, concat_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv2 in self.conv2:
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_decoders: int,
        stride: tuple[int, int],
        n_blocks: int,
        momentum: float = 0.01,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for _ in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: list[torch.Tensor]) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size: tuple[int, int],
        n_blocks: int,
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2, self.encoder.out_channel, inter_layers, n_blocks
        )
        self.decoder = Decoder(self.encoder.out_channel, en_de_layers, kernel_size, n_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    """The RMVPE network. `E2E(4, 1, (2, 2))` matches the released rmvpe.pt."""

    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ) -> None:
        super().__init__()
        self.unet = DeepUnet(
            kernel_size, n_blocks, en_de_layers, inter_layers, in_channels, en_out_channels
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        n_mel_channels: int,
        sampling_rate: int,
        win_length: int,
        hop_length: int,
        n_fft: int | None = None,
        mel_fmin: float = 0.0,
        mel_fmax: float | None = None,
        clamp: float = 1e-5,
    ) -> None:
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = librosa_mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.hann_window: dict[str, torch.Tensor] = {}
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio: torch.Tensor, center: bool = True) -> torch.Tensor:
        key = str(audio.device)
        if key not in self.hann_window:
            self.hann_window[key] = torch.hann_window(self.win_length).to(audio.device)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window[key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        # mel_basis is a registered buffer (a Tensor); the cast quiets the
        # Tensor|Module type nn.Module.__getattr__ hands back.
        mel_basis = cast(torch.Tensor, self.mel_basis)
        mel_output = torch.matmul(mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RmvpeModel:
    """Load rmvpe.pt and turn 16 kHz mono audio into an F0 track (10 ms hop)."""

    SAMPLE_RATE = 16000
    HOP = 160  # samples -> 10 ms frames

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        self.mel_extractor = MelSpectrogram(
            128, self.SAMPLE_RATE, 1024, self.HOP, None, 30.0, 8000.0
        ).to(device)
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu")
        model.load_state_dict(ckpt)
        model.eval()
        self.model = model.float().to(device)
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def _mel2hidden(self, mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            n_frames = mel.shape[-1]
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            hidden = self.model(mel)
            return hidden[:, :n_frames].squeeze(0).cpu().numpy()

    def _decode(self, hidden: np.ndarray, thred: float) -> np.ndarray:
        cents = self._to_local_average_cents(hidden, thred)
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio: np.ndarray, thred: float = 0.03) -> np.ndarray:
        """audio: mono float32 at 16 kHz. Returns per-frame F0 in Hz (0 = unvoiced)."""
        tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32)).to(self.device)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        mel = self.mel_extractor(tensor, center=True)
        hidden = self._mel2hidden(mel)
        return self._decode(hidden, thred)

    def _to_local_average_cents(self, salience: np.ndarray, thred: float) -> np.ndarray:
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        starts = center - 4
        ends = center + 5
        todo_salience = []
        todo_cents = []
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[idx, starts[idx] : ends[idx]])
            todo_cents.append(self.cents_mapping[starts[idx] : ends[idx]])
        sal = np.array(todo_salience)
        cents = np.array(todo_cents)
        product_sum = np.sum(sal * cents, 1)
        weight_sum = np.sum(sal, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided
