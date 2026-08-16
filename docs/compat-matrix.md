# מטריצת תאימות — Compatibility Spike

**נוצר:** 16.08.2026 22:35
**מכונה:** Windows-10-10.0.26200-SP0
**כרטיס NVIDIA:** לא נמצא

> נוצר אוטומטית ע"י `spike/run_spike.py`. לא לערוך ביד.

## תוצאות

| מועמד | תיאור | תוצאה | זמן התקנה | הערה |
|-------|-------|-------|-----------|------|
| `cpu-311` | Python 3.11 + torch CPU | ✅ pass | 344s | — |
| `cpu-312` | Python 3.12 + torch CPU | ❌ fail | 129s | failed probes: audio_separator |
| `cu128-311` | Python 3.11 + torch CUDA 12.8 | ⏭️ skipped | — | לא נמצא כרטיס NVIDIA במכונה הזו — חייב לרוץ על מחשב היעד |
| `cu126-311` | Python 3.11 + torch CUDA 12.6 | ⏭️ skipped | — | לא נמצא כרטיס NVIDIA במכונה הזו — חייב לרוץ על מחשב היעד |
| `cu128-312` | Python 3.12 + torch CUDA 12.8 | ⏭️ skipped | — | לא נמצא כרטיס NVIDIA במכונה הזו — חייב לרוץ על מחשב היעד |

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
| torch | ✅ | version=2.13.0+cpu, built_cuda=None, cuda_available=False, matmul_device=cpu, matmul_ok=True, matmul_ms=19.2 |
| numpy | ✅ | version=2.4.6 |
| soundfile | ✅ | version=0.14.0, roundtrip_ok=True |
| librosa | ✅ | version=0.11.0, yin_median_hz=440.4 |
| pyloudnorm | ✅ | version=?, lufs=-9.07 |
| pitch_shift | ✅ | module=python_stretch, input_frames=44100, output_frames=44100, length_preserved=True, finite=True |
| audio_separator | ✅ | importable=True, has_load_model=True, has_separate=True |
| torchfcpe | ✅ | importable=True, version=? |
| onnxruntime | ✅ | version=1.28.0, providers=['AzureExecutionProvider', 'CPUExecutionProvider'] |
| ffmpeg | ✅ | path=C:\Users\cdtauman\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE, version=ffmpeg version 8.1.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg  |


### `cpu-312` — Python 3.12 + torch CPU

| חבילה | גרסה שנפתרה |
|-------|--------------|
| audio-separator | 0.44.5 |
| librosa | 1.0.0 |
| numpy | 2.5.2 |
| onnxruntime | 1.28.0 |
| pyloudnorm | 0.2.0 |
| python-stretch | 0.3.1 |
| scipy | 1.18.0 |
| soundfile | 0.14.0 |
| soxr | 1.1.0 |
| torch | 2.13.0+cpu |
| torchfcpe | 0.0.4 |

| בדיקה | תוצאה | פרטים |
|-------|--------|--------|
| python | ✅ | version=3.12.13, executable=C:\dev\song-voice-converter\spike\.venvs\cpu-312\Scripts\python.exe |
| torch | ✅ | version=2.13.0+cpu, built_cuda=None, cuda_available=False, matmul_device=cpu, matmul_ok=True, matmul_ms=25.1 |
| numpy | ✅ | version=2.5.2 |
| soundfile | ✅ | version=0.14.0, roundtrip_ok=True |
| librosa | ✅ | version=1.0.0, yin_median_hz=440.4 |
| pyloudnorm | ✅ | version=?, lufs=-9.07 |
| pitch_shift | ✅ | module=python_stretch, input_frames=44100, output_frames=44100, length_preserved=True, finite=True |
| audio_separator | ❌ | ModuleNotFoundError: No module named 'audioread' |
| torchfcpe | ✅ | importable=True, version=? |
| onnxruntime | ✅ | version=1.28.0, providers=['AzureExecutionProvider', 'CPUExecutionProvider'] |
| ffmpeg | ✅ | path=C:\Users\cdtauman\AppData\Local\Microsoft\WinGet\Links\ffmpeg.EXE, version=ffmpeg version 8.1.1-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg  |

