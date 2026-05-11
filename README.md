# hpc-helper

A lightweight CLI that wraps the repetitive SSH and Slurm commands needed to run ML experiments on a restricted HPC cluster (login-node-only access, Slurm job scheduling).
The original workflow is on  https://xinchengo.github.io/ustc107/guides/ai/deep-learning-homework/.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.8+ | Local machine only |
| `ssh` on PATH | Standard on macOS/Linux; included with Windows 10+ |
| `tar` on PATH | Standard on macOS/Linux; included with Windows 10+ |
| SSH key-based auth | `~/.ssh/config` must have an alias for the login node |

One-time SSH key setup:

```bash
ssh-keygen -t ed25519
# Paste ~/.ssh/id_ed25519.pub into ~/.ssh/authorized_keys on the cluster
```

---

## Installation

```bash
git clone <this-repo>
cd hpc-helper
pip install -e .
```

This puts the `hpc` command on your PATH.

---

## Configuration

Run once to set up:

```bash
hpc init
```

| Prompt | Example | Saved as |
|---|---|---|
| SSH alias | `ustc-hpc` | `host` |
| Cluster username | `pb24000001` | `user` |
| Remote home directory | `/home/scc/pb24000001` | `remote_home` |
| Default conda environment | `base` | `conda_env` |
| Slurm account | `stu` | `account` |
| Slurm partition | `Students` | `partition` |
| Slurm QOS | `qos_stu_default` | `qos` |
| Default CPUs per job | `4` | `cpus` |
| Default GPUs per job | `1` | `gpus` |
| Default wall-time (minutes) | `200` | `walltime` |

Settings are saved to `~/.hpc-helper/config.toml`. Re-run `hpc init` at any time to update them.

---

## Commands

### `hpc up` — Allocate a GPU node

Submits a Slurm holder job and polls until it enters the running state.

```
hpc up
```

When the job starts running you will see something like
Submitted job 1048
Job 1048 is running on anode07.

The active job ID is cached so every subsequent command picks it up automatically.

| Flag | Description | Default |
|---|---|---|
| `--cpus N` | Override CPU count | config value |
| `--gpus N` | Override GPU count | config value |
| `--time N` | Wall-time in minutes | config value |
| `--name TEXT` | Slurm job name | `hpc-session` |

---

### `hpc push [LOCAL]` — Sync local code to the cluster

Uploads a local directory to `<remote_home>/projects/<dir-name>/` by default. Use `--to` to override the destination path under `<remote_home>`.

**Incremental by default.** After the first push, only files whose modification time or size changed are transferred. A manifest is stored at `~/.hpc-helper/manifests/` to track this.

```bash
hpc push                          # sync current directory (most common)
hpc push ./src                    # sync only a subdirectory
hpc push --to experiments/run_42  # sync to a custom remote path
hpc push --full                   # force a full re-sync, ignoring the manifest
```

**Excluding large files.** Create a `.hpcignore` file in your project root to skip directories or files:

```
# .hpcignore
data/
datasets/
*.zip
__pycache__/
.git/
```

Patterns follow glob syntax — directory names, wildcards, and file extensions are all supported. Files matching any pattern are excluded from every push.

> Large datasets (tens of thousands of files) should be in `.hpcignore` and uploaded separately. If more than 1,000 files appear as changed in a single incremental push, `hpc push` will warn you.

Existing remote files not present locally are left in place. To do a clean sync, open a shell with `hpc shell` and remove the remote directory first.

---

### `hpc run SCRIPT [ARGS...]` — Run a script on the GPU node

Executes a Python script inside the active Slurm allocation using `srun`. **Blocking** — streams stdout/stderr live and does not return until the remote process finishes.

```bash
hpc run train.py --lr 0.001 --epochs 50
```

Requires an active job (`hpc up`) and code on the cluster (`hpc push`).

| Flag | Description |
|---|---|
| `--env NAME` | Use a different conda env for this run |
| `--raw CMD` | Run an arbitrary shell command instead of a Python script |
| `--no-conda` | Skip conda, use system `python` |

**Sequential sweep example:**

```bash
for lr in 1e-3 5e-4 1e-4; do
  hpc run train.py --lr $lr --tag "lr_sweep"
done
```

Each run blocks until it finishes before the next starts.

---

### `hpc batch BATCH_FILE` — Submit groups of runs

For larger sweeps where you want to queue several independent groups upfront. Each group becomes one Slurm job; scripts within a group run sequentially. All groups are queued at once and Slurm runs them as resources free up.

Define groups in a YAML file:

```yaml
# batch.yaml
resources:          # shared Slurm settings for all groups
  cpus: 4
  gpus: 1
  time: 200

groups:
  - name: lr_high
    runs:
      - train.py --lr 1e-3 --epochs 100
      - eval.py  --checkpoint checkpoints/lr_high/best.pt

  - name: lr_mid
    runs:
      - train.py --lr 5e-4 --epochs 100
      - eval.py  --checkpoint checkpoints/lr_mid/best.pt
```

```bash
hpc push               # sync code first
hpc batch batch.yaml   # queue all groups
```

```
Queued lr_high  → job 104840
Queued lr_mid   → job 104841
```

Override resources per group if needed:

```yaml
groups:
  - name: big_model
    resources:
      time: 400
      gpus: 2
    runs:
      - train.py --model large --epochs 200
```

Each group's output goes to `slurm-<jobid>-<name>.out` in your remote home directory.

---

### `hpc logs` — Tail a Slurm output log

```bash
hpc logs                   # tail the most recently submitted batch group
hpc logs --group lr_mid    # tail a specific group by name
hpc logs --job 104841      # tail by job ID
hpc logs --lines 100       # show more lines before following
```

---

### `hpc pull [REMOTE]` — Download results from the cluster

No active job required — connects directly to the login node's shared filesystem.

```bash
hpc pull                         # update the local project directory in-place
hpc pull results/                # download a specific subdirectory into the current directory
hpc pull checkpoints/best.pt     # download a single file
hpc pull --to ./local_results    # save into a custom local destination
```

Slurm log files (`slurm-*.out`), `.git/`, and any patterns in your local `.hpcignore` are always excluded from the transfer.

---

### `hpc shell` — Interactive bash session on the GPU node

```bash
hpc shell
```

Drops into a bash session on the compute node via `srun --overlap --pty bash`. Type `exit` to leave without cancelling the job. Requires an active job (`hpc up`).

---

### `hpc status` — Show your jobs in the Slurm queue

```
$ hpc status
JOBID    NAME          STATE   NODE    TIME
1048   hpc-session   R       gpu07   00:12:34
```

---

### `hpc ps` — Show cached session state

```
$ hpc ps
Host           ustc-hpc
User           pb24000001
Active job     1048  (on gpu07)
Remote project /home/scc/pb24000001/projects/my-project
Conda env      torch
```

---

### `hpc down` — Cancel the job and release the GPU

```bash
hpc down
```

Runs `scancel` and clears the cached job ID.

---

## Typical workflow

```bash
hpc up                                        # allocate a node
hpc push                                      # sync code
hpc run train.py --epochs 1 --data-fraction 0.01  # sanity check
hpc run train.py --epochs 100 --lr 5e-4       # real run (blocking)
hpc pull checkpoints/                         # download results
hpc down                                      # release GPU
```



---

## Remote project layout

```
<remote_home>/
├── entry.sh                        # Slurm holder script (managed by hpc-helper)
├── .hpc-helper-batch/              # temporary batch scripts (managed by hpc-helper)
└── projects/
    └── <your-project>/             # mirrored from local by hpc push
        ├── train.py
        └── ...
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `hpc up` hangs past 5 min | Check quota with `hpc status`; cluster may be at capacity |
| `Permission denied (publickey)` | Verify your public key is in `~/.ssh/authorized_keys` on the cluster |
| `No active job` error | Run `hpc up` first, or check state with `hpc ps` |
| Job disappeared between commands | Hit wall-time limit; use `hpc up --time 400` to request more |
| Push warns about 1000+ changed files | A large dataset was added; add it to `.hpcignore` |
| `Nothing changed — skipping push` but remote is stale | Run `hpc push --full` to force a complete re-sync |


