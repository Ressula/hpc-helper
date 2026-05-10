**Description:** Standard operating procedure for interacting with a restricted HPC compute cluster. The cluster requires connecting via a login node (`107.ustc.edu.cn`) and using Slurm to allocate a GPU compute node. Direct SSH to compute nodes is strictly blocked.

Preliminary: 
1. connect to web shell with ssh
edit ssh config:
```
Host 107.ustc.edu.cn
  HostName 107.ustc.edu.cn
  User pb2xxxxxxx
```
  
generate a public key:
`ssh-keygen -t ed25519`
then add the key in `USER/.ssh/id_ed25519`(**in your local coputer**) to the server (**run the commands on the web shell**):
```
mkdir -p ~/.ssh 
chmod 700 ~/.ssh
nano ~/.ssh/authorized_keys (paste the key)
chmod 600 ~/.ssh/authorized_keys
```

2. create a `entry.sh` under the user root
```
#!/bin/bash
#SBATCH -A stu
#SBATCH --partition=Students
#SBATCH --qos=qos_stu_default
#SBATCH --job-name=pure-cli-session
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --time=200
#SBATCH --gres=gpu:1
sleep infinity
```

## 1. Environment Context

- **Login Node SSH Alias:** `107.ustc.edu.cn` (Pre-configured in `~/.ssh/config`).
- **Target User:** `USERNAME` (e.g. pb24000001)
- **Architecture:** Shared filesystem. Files pushed to the login node are immediately accessible on the compute nodes.
- **Job Holder Script:** `entry.sh` (Used to request resources and keep the node alive via `sleep infinity`).

## 2. Workflow

### Step 1: Verify / Create the Job Holder Script

Ensure the `entry.sh` script exists on the login node's root project directory. 
`/home/scc/USERNAME/entry.sh`
### Step 2: Submit the Allocation Job

Submit the batch script to Slurm via the login node to request the GPU compute resources.

Bash
```
ssh 107.ustc.edu.cn "sbatch /home/scc/USERNAME/entry.sh"
```
### Step 3: Monitor Queue and Extract Job ID

Check the job state.
Bash

```
ssh 107.ustc.edu.cn "squeue -u USERNAME"
```

> **Action:** Parse the terminal output. Identify the row where the state is `R` (runnning). Extract and store the numeric `JOBID`. Do not proceed to Step 4 until the job is `R`.

### (optional) Step 4: Sync Code & Assets (File Transfer)

Because of the shared filesystem, transfer all necessary local project files to the login node. 

Bash

```
scp -r ./local_workspace/* 107.ustc.edu.cn:/home/scc/USERNAME/remote_workspace/
```

### Step 5: Execute Remote Commands on the GPU Node

To execute scripts, attach to the running allocation using `srun`. Pass the stored `JOBID` into the command.

Bash

```
# Example: Running a python training script
ssh 107.ustc.edu.cn "srun --jobid=<STORED_JOBID> /home/scc/USERNAME/miniconda3/bin/conda run -n my_env python /home/scc/USERNAME/remote_workspace/train.py"
```

or run a pipeline:
` ssh 107.ustc.edu.cn "srun --jobid=STORED_JOBID bash /home/scc/USERNAME/run_task.sh"`

If you want to see the logs and output, use the interactive bash:
`srun --jobid=<STORED_JOBID> --overlap --pty bash`

### Step 6: Teardown 

Once the objective is complete, kill the Slurm allocation to release the GPU back to the university pool. Failure to do so will drain the user's compute quota.

Bash

```
ssh 107.ustc.edu.cn "scancel <STORED_JOBID>"
```
