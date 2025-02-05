import json
import os
import time
import subprocess
import shutil

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="./configs", config_name="train")
def run_experiment(cfg: DictConfig):
    """
    Build and submit an sbatch script for the benchmark job.
    Copy the mixtera_server_dir folder to a local checkpoint (within output_dir)
    and update the environment variable to point to the copied folder.
    """
    # Extract configuration variables.
    data_path = cfg.data_path
    finetune_data_path = cfg.finetune_data_path
    image_path = cfg.image_path
    finetune_image_path = cfg.finetune_image_path
    pretrained_model_path = cfg.pretrained_model_path
    output_dir = cfg.output_dir

    mode = cfg.mode
    max_steps = cfg.max_steps
    num_workers = cfg.num_workers

    mixtera_server_dir = cfg.mixtera_server_dir
    mixtera_server_port = cfg.mixtera_server_port
    mixtera_chunk_size = cfg.mixtera_chunk_size
    mixture = cfg.mixture

    # Construct a unique job name and update the output directory.
    job_name = (
        f"tinyllava-{mode}-mixture_{mixture.name}-steps_{max_steps}"
        f"-workers_{num_workers}-chunksize_{mixtera_chunk_size}"
    )
    output_dir = os.path.join(output_dir, job_name)
    os.makedirs(output_dir, exist_ok=True)

    # Copy mixtera_server_dir folder to a local checkpoint directory inside output_dir.
    local_mixtera_server_dir = os.path.join(output_dir, "mixtera_server_copy")
    # If using Python 3.8 or later, dirs_exist_ok=True prevents errors if the target exists.
    shutil.copytree(mixtera_server_dir, local_mixtera_server_dir, dirs_exist_ok=True)

    # Build the sbatch header.
    sbatch_header = f"""#!/bin/bash
#SBATCH --account=a-a09
#SBATCH --job-name={job_name}
#SBATCH --output={output_dir}/output.log
#SBATCH --error={output_dir}/output.err
#SBATCH --partition=normal
#SBATCH --environment=tinyllava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
"""

    # Define environment variables.
    # Note: MIXTERA_SERVER_DIR now points to the copied folder.
    env_var_lines = (
        f"\nexport MIXTERA_SERVER_ADDR=$(hostname)\n"
        f"export MIXTERA_SERVER_DIR={local_mixtera_server_dir}\n"
        f"export MIXTERA_SERVER_PORT={mixtera_server_port}\n"
        f"export MIXTERA_CHUNK_SIZE={mixtera_chunk_size}\n"
        f"export MIXTERA_MODE={mode}\n"
        f"export MIXTERA_MIXTURE='{json.dumps(dict(mixture.components))}'\n"
        f"export MIXTERA_JOB_ID={job_name}\n"
        f"export NUM_WORKERS={num_workers}\n"
        f"export TRITON_CACHE_DIR={cfg.triton_cache_dir}\n"
        f"export HF_HOME={cfg.hf_home}\n"
        f"\nsource {cfg.venv_path}/bin/activate\n"
    )

    change_dir_cmd = f"\ncd {cfg.tinyllava_dir}"

    # Command to run the mixtera server.
    run_mixtera_server_cmd = (
        f"\n{cfg.venv_path}/bin/python -u -m mixtera.network.server.entrypoint "
        f"$MIXTERA_SERVER_DIR --host $MIXTERA_SERVER_ADDR --port $MIXTERA_SERVER_PORT &\n"
        "\nsleep 5\n"
    )

    # Determine which training script to use.
    train_script = (
        "scripts/train/finetune.sh" if mode == "finetune" else "scripts/train/train.sh"
    )
    run_cmd = f"\nbash {train_script} "

    if mode == "finetune":
        run_cmd += (
            f"{finetune_data_path} {finetune_image_path} {cfg.llm.version} "
            f"{cfg.vision_tower.version} \"\" {cfg.connector.type} {cfg.llm.conv_version} "
            f"mixtera_{cfg.mixture.name} common {cfg.llm.model_max_length} "
            f"{max_steps} {num_workers} {pretrained_model_path} {output_dir}"
        )
    elif mode == "pretrain":
        raise NotImplementedError("Pretraining is not yet supported.")

    # Combine all parts to form the complete script.
    script_content = sbatch_header + env_var_lines + change_dir_cmd + run_mixtera_server_cmd + run_cmd

    # Write the script content to a temporary .sbatch file.
    sbatch_file_path = f"./{time.time()}.sbatch"
    with open(sbatch_file_path, "w") as f:
        f.write(script_content)

    # Submit the sbatch script using subprocess and remove the file afterward.
    try:
        result = subprocess.run(
            ["sbatch", sbatch_file_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(f"Submitted job: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr.strip()}")
    finally:
        # Remove the temporary sbatch file.
        try:
            os.remove(sbatch_file_path)
            print(f"Removed temporary file: {sbatch_file_path}")
        except OSError as err:
            print(f"Error removing file {sbatch_file_path}: {err}")

if __name__ == "__main__":
    run_experiment()
