# מטריצת תאימות — Compatibility Spike

**נוצר:** 17.08.2026 03:06
**מכונה:** Windows-10-10.0.26200-SP0
**מתאמים גרפיים:** Intel(R) Arc(TM) Graphics, Microsoft Remote Display Adapter
**כרטיס NVIDIA:** לא נמצא
**כרטיס Intel:** נמצא

> נוצר אוטומטית ע"י `spike/run_spike.py`. לא לערוך ביד.

## תוצאות

| מועמד | תיאור | תוצאה | זמן התקנה | הערה |
|-------|-------|-------|-----------|------|
| `cpu-311` | Python 3.11 + torch CPU | ✅ pass | 38s | — |
| `xpu-311` | Python 3.11 + torch XPU (Intel) | ✅ pass | 39s | — |

### `cpu-311` — Python 3.11 + torch CPU

| חבילה | גרסה שנפתרה |
|-------|--------------|
| audio-separator | 0.44.5 |
| librosa | 0.11.0 |
| numpy | 2.4.6 |
| onnxruntime | 1.28.0 |
| pyloudnorm | 0.2.0 |
| python-stretch | 0.3.1 |
| scipy | 1.17.1 |
| soundfile | 0.14.0 |
| soxr | 1.1.0 |
| torch | 2.13.0+cpu |
| torchfcpe | 0.0.4 |

| בדיקה | תוצאה | פרטים |
|-------|--------|--------|
| python | ✅ | version=3.11.15, executable=C:\dev\song-voice-converter\spike\.venvs\cpu-311\Scripts\python.exe |
| torch | ✅ | version=2.13.0+cpu, built_cuda=None, built_xpu=None, cuda_available=False, xpu_available=False, accelerators=[], matmul_device=cpu, matmul_ok=True, matmul_ms=75 |
| xpu | ❌ | RuntimeError: torch.xpu is present but reports no available device |
| numpy | ✅ | version=2.4.6 |
| soundfile | ✅ | version=0.14.0, roundtrip_ok=True |
| librosa | ✅ | version=0.11.0, yin_median_hz=440.4 |
| pyloudnorm | ✅ | version=?, lufs=-9.07 |
| pitch_shift | ✅ | module=python_stretch, input_frames=44100, output_frames=44100, length_preserved=True, finite=True |
| audio_separator | ✅ | importable=True, has_load_model=True, has_separate=True |
| torchfcpe | ✅ | importable=True, version=? |
| onnxruntime | ✅ | version=1.28.0, providers=['AzureExecutionProvider', 'CPUExecutionProvider'] |
| ffmpeg | ✅ | path=C:\Users\cdtauman\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE, version=ffmpeg version 8.1.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg  |

**עומס אמיתי לכל רכיב** (מכשירים שנבדקו: cpu)

| רכיב | cpu | מה נבדק |
|------|------|---------|
| separation | ✅ | stft+mha+istft |
| f0 | ✅ | torchfcpe end-to-end |
| conversion | ✅ | conv1d+transformer+convtranspose |
| pitch_shift | ✅ | python-stretch (cpu-only by design) |


### `xpu-311` — Python 3.11 + torch XPU (Intel)

| חבילה | גרסה שנפתרה |
|-------|--------------|
| audio-separator | 0.44.5 |
| librosa | 0.11.0 |
| numpy | 2.4.6 |
| onnxruntime | 1.28.0 |
| pyloudnorm | 0.2.0 |
| python-stretch | 0.3.1 |
| scipy | 1.17.1 |
| soundfile | 0.14.0 |
| soxr | 1.1.0 |
| torch | 2.13.0+xpu |
| torchfcpe | 0.0.4 |

| בדיקה | תוצאה | פרטים |
|-------|--------|--------|
| python | ✅ | version=3.11.15, executable=C:\dev\song-voice-converter\spike\.venvs\xpu-311\Scripts\python.exe |
| torch | ✅ | version=2.13.0+xpu, built_cuda=None, built_xpu=20260000, cuda_available=False, xpu_available=True, accelerators=['xpu'], matmul_device=cpu, matmul_ok=True, matm |
| xpu | ✅ | device_count=1, device_name=Intel(R) Arc(TM) Graphics, total_memory_gb=7.8, driver_version=1.15.39183+1, max_compute_units=128, gpu_eu_count=128, type=gpu, matm |
| numpy | ✅ | version=2.4.6 |
| soundfile | ✅ | version=0.14.0, roundtrip_ok=True |
| librosa | ✅ | version=0.11.0, yin_median_hz=440.4 |
| pyloudnorm | ✅ | version=?, lufs=-9.07 |
| pitch_shift | ✅ | module=python_stretch, input_frames=44100, output_frames=44100, length_preserved=True, finite=True |
| audio_separator | ✅ | importable=True, has_load_model=True, has_separate=True |
| torchfcpe | ✅ | importable=True, version=? |
| onnxruntime | ✅ | version=1.28.0, providers=['AzureExecutionProvider', 'CPUExecutionProvider'] |
| ffmpeg | ✅ | path=C:\Users\cdtauman\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE, version=ffmpeg version 8.1.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg  |

**עומס אמיתי לכל רכיב** (מכשירים שנבדקו: cpu, xpu)

| רכיב | cpu | xpu | מה נבדק |
|------|------|------|---------|
| separation | ✅ | ✅ | stft+mha+istft |
| f0 | ✅ | ✅ | torchfcpe end-to-end |
| conversion | ✅ | ✅ | conv1d+transformer+convtranspose |
| pitch_shift | ✅ | — | python-stretch (cpu-only by design) |

