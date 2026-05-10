# hpc-helper

A lightweight interactive CLI that wraps the repetitive SSH and Slurm commands needed to run ML experiments on a restricted HPC cluster (login-node-only access, Slurm job scheduling).

---

## The problem it solves

The raw workflow requires you to remember and type a chain of commands every session:

```
ssh login "sbatch ~/entry.sh"
ssh login "squeue -u me"          # repeat until state = R
ssh login "srun --jobid=XXXX ..."
ssh login "scancel XXXX"
```

`hpc-helper` wraps this into short, memorable sub-commands and remembers context (active job ID, remote project path, conda env) across invocations.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| SSH key-based auth | `~/.ssh/config` must have an alias for the login node |
| Python 3.8+ | Local machine only |
| `scp` / `ssh` on PATH | Standard on macOS/Linux/WSL |

One-time cluster setup (do this once via the web shell):

```bash
# On your local machine
ssh-keygen -t ed25519
# Paste the contents of ~/.ssh/id_ed25519.pub into ~/.ssh/authorized_keys on the cluster
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

Run the interactive setup wizard once:

```bash
hpc init
```

It will ask for:

| Prompt | Example value | Saved as |
|---|---|---|
| SSH alias (from `~/.ssh/config`) | `ustc-hpc` | `host` |
| Your cluster username | `pb24000001` | `user` |
| Remote home directory | `/home/scc/pb24000001` | `remote_home` |
| Default conda environment | `torch` | `conda_env` |
| Default CPU cores per job | `4` | `cpus` |
| Default GPU count | `1` | `gpus` |
| Default wall-time (minutes) | `200` | `walltime` |

Settings are saved to `~/.hpc-helper/config.toml`. You can edit it by hand or re-run `hpc init` at any time.

---

## Commands

### `hpc up` — Allocate a GPU node

Submits the Slurm holder job and polls until it enters the running state, then prints the job ID.

```
$ hpc up
Submitting job... done  (job 104832)
Waiting for allocation  ...........  Running on gpu07
Job 104832 is ready.
```

The active job ID is cached locally so every subsequent command picks it up automatically.

Options:

| Flag | Description | Default |
|---|---|---|
| `--cpus N` | Override CPU count for this session | config value |
| `--gpus N` | Override GPU count | config value |
| `--time N` | Wall-time in minutes | config value |
| `--name TEXT` | Slurm job name | `hpc-session` |

---

### `hpc status` — Check running jobs

```
$ hpc status
JOBID    NAME          STATE   NODE    TIME
104832   hpc-session   R       gpu07   00:12:34
```

---

### `hpc push [LOCAL]` — Sync local code to the cluster

> Run this before `hpc run` whenever you have local changes to upload.

Uploads a local directory to the remote project folder via `scp -r`.

The remote destination is `<remote_home>/projects/<local-dir-name>/` by default.  
Pass `--to <rel-path>` to override the sub-path under `<remote_home>`.

```bash
# Sync the current directory (most common case)
hpc push

# Sync only a subdirectory (e.g. just the model definitions)
hpc push ./src/models

# Sync to a named experiment folder instead of the default project path
hpc push --to experiments/run_42

# Sync a completely separate local project
hpc push ~/other-project
```

Files are transferred with `scp -r`; existing remote files not present locally are left in place (not deleted). To do a clean sync, run `hpc shell` and remove the remote directory manually first.

---

### `hpc run SCRIPT [ARGS...]` — Run a script on the GPU node

> Requires `hpc up` (active job) and `hpc push` (code on remote) to have been run first.

Executes a script inside the active Slurm allocation using `srun`. **By default, `hpc run` is blocking** — it streams stdout/stderr to your terminal and does not return until the remote process finishes. A second `hpc run` call will only start after the first one completes.

```bash
hpc run train.py --lr 0.001 --epochs 50
```

Expands to:

```bash
ssh <host> "srun --jobid=104832 \
  /home/scc/pb24000001/miniconda3/bin/conda run -n torch \
  python /home/scc/pb24000001/projects/my-project/train.py \
  --lr 0.001 --epochs 50"
```

The script path is resolved relative to the last `hpc push` destination.

Options:

| Flag | Description |
|---|---|
| `--env NAME` | Use a different conda env for this run |
| `--raw CMD` | Run an arbitrary shell command instead of a Python script |
| `--no-conda` | Skip conda entirely, use bare `python` on PATH |
**Example — sequential sweep:**

Each run blocks until it finishes before the next one starts. Stdout from each run is printed live.

```bash
for lr in 1e-3 5e-4 1e-4; do
  hpc run train.py --lr $lr --tag "lr_sweep"
done
```

---

### `hpc batch BATCH_FILE` — Submit groups of runs in parallel

For larger sweeps where you want to queue up several **independent groups** without babysitting each one. Each group gets its own Slurm allocation; scripts within a group run sequentially in that allocation. All groups are submitted at once and Slurm runs them one after another as resources free up.

Define the groups in a YAML file:

```yaml
# batch.yaml
resources:            # shared Slurm settings for all groups
  cpus: 4
  gpus: 1
  time: 200

groups:
  - name: lr_high
    runs:
      - train.py --lr 1e-3 --epochs 100 --tag sweep
      - eval.py  --checkpoint checkpoints/lr_high/best.pt

  - name: lr_mid
    runs:
      - train.py --lr 5e-4 --epochs 100 --tag sweep
      - eval.py  --checkpoint checkpoints/lr_mid/best.pt

  - name: lr_low
    runs:
      - train.py --lr 1e-4 --epochs 100 --tag sweep
      - eval.py  --checkpoint checkpoints/lr_low/best.pt
```

Then submit all groups:

```bash
hpc push                   # sync code first
hpc batch batch.yaml       # submits one sbatch job per group
```

Output:

```
Submitted group lr_high  → job 104840
Submitted group lr_mid   → job 104841
Submitted group lr_low   → job 104842
```

Each group writes its output to a separate log file (`slurm-<jobid>-<group-name>.out`) on the remote. Follow a specific group's log:

```bash
hpc logs --job 104841          # by job ID
hpc logs --group lr_mid        # by group name
```

A group's `resources` block can be overridden per-group if one experiment needs more time or a different GPU count:

```yaml
groups:
  - name: big_model
    resources:
      time: 400
      gpus: 2
    runs:
      - train.py --model large --epochs 200
```

> Use `hpc batch` when your groups are **independent** (different hyperparameters, different seeds) and you want to queue them all upfront rather than re-submitting manually after each one finishes. If runs must share state or feed into each other, use sequential `hpc run` calls instead.

---

### `hpc shell` — Drop into an interactive bash session on the GPU node

> Requires an active job (`hpc up`).

```
$ hpc shell
Attaching to job 104832 on gpu07...
[pb24000001@gpu07 ~]$
```

Equivalent to `srun --jobid=<ID> --overlap --pty bash`. Type `exit` to leave without cancelling the job.

---

### `hpc logs` — Tail the Slurm output log

> Requires an active job (`hpc up`).

```bash
hpc logs           # tail -f the latest slurm-<jobid>.out
hpc logs --lines 50
```

---

### `hpc pull [REMOTE]` — Download results back to local

Slurm output files (`slurm-*.out`) are always excluded from the transfer.

```bash
hpc pull                         # pulls the full remote project dir to ./
hpc pull results/                # pulls a specific subdirectory
hpc pull checkpoints/best.pt     # pulls a single file
hpc pull --to ./local_results    # save into a custom local destination
```

No active job is required — `hpc pull` connects directly to the login node's shared filesystem.

---

### `hpc down` — Cancel the job and release the GPU

```
$ hpc down
Cancelling job 104832... done.
GPU released. Remember to do this when you're finished!
```

Clears the cached job ID.

---

### `hpc ps` — Show cached session state

Prints what `hpc-helper` currently remembers about the active session (useful when picking up after a disconnect).

```
$ hpc ps
Host         : ustc-hpc
User         : pb24000001
Active job   : 104832  (Running on gpu07)
Remote path  : /home/scc/pb24000001/projects/my-project
Conda env    : torch
```

---

## Typical ML experiment session

### Single run / quick iteration

```bash
# 1. Allocate a node
hpc up

# 2. Push your latest code
hpc push

# 3. Sanity-check (small dataset, 1 epoch)
hpc run train.py --epochs 1 --data-fraction 0.01

# 4. Real run — blocks until finished, output streams live
hpc run train.py --epochs 100 --lr 5e-4 --tag v1

# 5. Tweak a hyperparameter and re-run (no re-push needed if code unchanged)
hpc run train.py --epochs 100 --lr 1e-4 --tag v2

# 6. Pull results locally
hpc pull checkpoints/

# 7. Release the GPU
hpc down
```

### Hyperparameter sweep with `hpc batch`

```bash
# 1. Push code once
hpc push

# 2. Submit all groups upfront — Slurm queues them and runs each in turn
hpc batch batch.yaml

# 3. Check which group is currently running
hpc status

# 4. Follow the active group's output
hpc logs --group lr_mid

# 5. Pull all results once the queue is drained
hpc pull results/
```

---

## Project layout (remote)

```
/home/scc/<user>/
├── entry.sh                  # Slurm holder script (managed by hpc-helper)
└── projects/
    └── <your-project>/       # mirrored from local by `hpc push`
        ├── train.py
        └── ...
```

`hpc-helper` stores the holder script in `~/entry.sh` on the cluster; your actual code lives under `~/projects/`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `hpc up` hangs past 5 min | Check quota with `hpc status`; cluster may be at capacity |
| `Permission denied (publickey)` | Re-run `hpc init` and verify your public key is in `~/.ssh/authorized_keys` on the cluster |
| `No active job` error | Run `hpc up` first, or `hpc ps` to check state |
| Job disappeared between commands | It may have hit the wall-time limit; `hpc up --time 400` to request more |

---

## License

MIT
