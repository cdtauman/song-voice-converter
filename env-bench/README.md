# SongVoice `env-bench`

This directory is a benchmark-only process boundary. It is never imported by
`svc_engine`, is not a project dependency, and is excluded from application
builds by Hatch's explicit `src/svc_engine` / `src/svc_app` package list.

Each comparison engine receives its own virtual environment under
`env-bench/runtimes/` because their historical Torch/NumPy matrices conflict.
The name `env-bench` refers to this whole isolated laboratory; it is not a
promise that incompatible engines share one `site-packages` directory.

Pinned upstream sources:

| engine | source commit | license | product status |
|---|---|---|---|
| Seed-VC | `Plachtaa/seed-vc@51383efd921027683c89e5348211d93ff12ac2a8` | GPL-3.0 | archived reference only; never production |
| DDSP-SVC | `yxlllc/DDSP-SVC@3635301027473c6662d05a1c73ef34fba7f15f90` | MIT | benchmark adapter only |

Bootstrap one engine from PowerShell:

```powershell
.\env-bench\bootstrap.ps1 -Engine seed
.\env-bench\bootstrap.ps1 -Engine ddsp
```

The script clones only the pinned commit and installs upstream requirements
inside `env-bench/runtimes/<engine>/.venv`. It never writes to SongVoice's
`.venv`, `constraints.txt`, `rvc-requirements.lock`, or `pyproject.toml`.

The adapter has one output contract for `svc-bench`: a declared WAV path.
Seed-VC additionally needs a consented 1–30 second reference voice. DDSP-SVC
needs a user-supplied checkpoint/config whose model license has been audited.
No model or audio is downloaded or committed by SongVoice.

Run the isolation gate after bootstrap:

```powershell
python env-bench/verify_isolation.py
uv pip check --python .venv/Scripts/python.exe
```
