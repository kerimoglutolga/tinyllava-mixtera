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
    Then, for each checkpoint (every 500 steps), create a *separate* SBATCH script 
    and submit it as an individual job.
    
    Training and evaluation job submission are now controlled by the parameters:
      - cfg.run_train: if False, skip training.
      - cfg.run_eval: if False, skip evaluation.
      
    If training is skipped, evaluation jobs (if enabled) are submitted without a dependency.
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
        f"{cfg.llm.name}-{mode}-mixture_{mixture.name}-steps_{max_steps}"
        f"-workers_{num_workers}-chunksize_{mixtera_chunk_size}"
    )
    output_dir = os.path.join(output_dir, job_name)
    os.makedirs(output_dir, exist_ok=True)

    # Copy mixtera_server_dir folder to a local checkpoint directory inside output_dir.
    local_mixtera_server_dir = os.path.join(output_dir, "mixtera_server_copy")
    shutil.copytree(mixtera_server_dir, local_mixtera_server_dir, dirs_exist_ok=True)

    training_job_id = None
    ########################################################################
    # 1. TRAINING JOB (if enabled)
    ########################################################################
    if cfg.run_train:
        # Build the sbatch header for the training job.
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

        # Define environment variables for training.
        env_var_lines = (
            f"\nexport MIXTERA_SERVER_ADDR=$(hostname)\n"
            f"export MIXTERA_SERVER_DIR={local_mixtera_server_dir}\n"
            f"export MIXTERA_SERVER_PORT={mixtera_server_port}\n"
            f"export MIXTERA_CHUNK_SIZE={mixtera_chunk_size}\n"
            f"export MIXTERA_MODE={mode}\n"
            f"export MIXTERA_MIXTURE='{json.dumps(dict(mixture.components if mixture.components else {}))}'\n"
            f"export MIXTERA_JOB_ID={job_name}\n"
            f"export MIXTERA_MIXTURE_TYPE={mixture.type}\n"
            f"export NUM_WORKERS={num_workers}\n"
            f"export TRITON_CACHE_DIR={cfg.triton_cache_dir}\n"
            f"export HF_HOME={cfg.hf_home}\n"
            f"\nsource {cfg.venv_path}/bin/activate\n"
        )

        change_dir_cmd = f"\ncd {cfg.tinyllava_dir}"

        # Command to run the mixtera server (in the background).
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

        # Create a model name and an output directory for that model.
        llm_variant = cfg.llm.version.split("/")[-1]
        vt_variant  = cfg.vision_tower.version.split("/")[-1]
        model_name = f"tinyllava-{llm_variant}-{vt_variant}-mixtera_{mixture.name}-finetune"
        model_output_dir = os.path.join(output_dir, model_name)

        if mode == "finetune":
            run_cmd += (
                f"{finetune_data_path} {finetune_image_path} {cfg.llm.version} "
                f"{cfg.vision_tower.version} \"\" {cfg.connector.type} {cfg.llm.conv_version} "
                f"mixtera_{cfg.mixture.name} common {cfg.llm.model_max_length} "
                f"{max_steps} {num_workers} {pretrained_model_path} {model_output_dir}"
            )
        elif mode == "pretrain":
            raise NotImplementedError("Pretraining is not yet supported.")

        # Combine all parts to form the complete training script.
        train_script_content = (
            sbatch_header + env_var_lines + change_dir_cmd + run_mixtera_server_cmd + run_cmd
        )

        # Write the script content to a temporary .sbatch file for submission.
        train_sbatch_file_path = f"./{time.time()}_train.sbatch"
        with open(train_sbatch_file_path, "w") as f:
            f.write(train_script_content)

        # Submit the sbatch script using subprocess.
        try:
            result = subprocess.run(
                ["sbatch", train_sbatch_file_path],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            train_output = result.stdout.strip()
            print(f"Submitted training job: {train_output}")
            # Expected output format: "Submitted batch job 12345"
            training_job_id = train_output.split()[-1]
        except subprocess.CalledProcessError as e:
            print(f"Error submitting training job: {e.stderr.strip()}")
        finally:
            try:
                os.remove(train_sbatch_file_path)
                print(f"Removed temporary file: {train_sbatch_file_path}")
            except OSError as err:
                print(f"Error removing file {train_sbatch_file_path}: {err}")
    else:
        print("Training job skipped as per configuration (cfg.run_train is False).")
        # If training is skipped, we assume the model outputs/checkpoints already exist.
        # Set up model_output_dir and model_name accordingly.
        llm_variant = cfg.llm.version.split("/")[-1]
        vt_variant  = cfg.vision_tower.version.split("/")[-1]
        model_name = f"tinyllava-{llm_variant}-{vt_variant}-mixtera_{mixture.name}-finetune"
        model_output_dir = os.path.join(output_dir, model_name)

    ########################################################################
    # 2. EVALUATION JOBS (ONE PER CHECKPOINT, if enabled)
    ########################################################################
    #
    # Evaluation job submission will only be executed if cfg.run_eval is True.
    if cfg.run_eval:
        # If training was run, add dependency. Otherwise, leave dependency out.
        dependency_line = (
            f"#SBATCH --dependency=afterok:{training_job_id}\n" if training_job_id else ""
        )
        # Iterate over each checkpoint step.
        for step in range(500, max_steps + 1, 500):
            checkpoint_dir = os.path.join(model_output_dir, f"checkpoint-{step}")

            eval_job_name = f"{job_name}-eval-step{step}"
            eval_sbatch_header = f"""#!/bin/bash
#SBATCH --account=a-a09
#SBATCH --job-name={eval_job_name}
#SBATCH --output={output_dir}/eval-{step}.log
#SBATCH --error={output_dir}/eval-{step}.err
#SBATCH --partition=normal
#SBATCH --environment=tinyllava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=11:59:00
{dependency_line}"""

            # Build the commands for the evaluation script for this specific checkpoint.
            eval_cmd = f"\nsource {cfg.venv_path}/bin/activate\n"
            eval_cmd += f"export NUM_WORKERS={num_workers}\n"
            eval_cmd += f"export TRITON_CACHE_DIR={cfg.triton_cache_dir}\n"
            eval_cmd += f"export HF_HOME={cfg.hf_home}\n"
            eval_cmd += f"cd {cfg.tinyllava_dir}\n\n"
            eval_cmd += f'echo "Evaluating checkpoint at step {step}: {checkpoint_dir}"\n\n'
            eval_cmd += f'MODEL_PATH="{checkpoint_dir}"\n'
            eval_cmd += f'MODEL_NAME="{model_name}-step{step}"\n'
            eval_cmd += f'CONV_MODE="{cfg.llm.conv_version}"\n\n'

            # Run your various eval scripts.
            eval_cmd += 'echo "Running pope.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/pope.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running mme.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mme.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running mmvet.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmvet.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running mmmu.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/mmmu.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running sqa.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/sqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running textvqa.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0 bash scripts/eval/textvqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running vqav2.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/vqav2.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += 'echo "Running gqa.sh"\n'
            eval_cmd += 'CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/eval/gqa.sh "$MODEL_PATH" "$MODEL_NAME" "$CONV_MODE"\n\n'
            eval_cmd += 'echo "--------------------------------------------------"\n\n'

            eval_cmd += f'echo "Finished evaluating step {step}."'

            # Combine the SBATCH header and commands for this checkpoint.
            eval_script_content = eval_sbatch_header + eval_cmd

            # Write the evaluation sbatch script to a temporary file (unique for this step).
            eval_sbatch_file = f"./{time.time()}_eval_step{step}.sbatch"
            with open(eval_sbatch_file, "w") as f:
                f.write(eval_script_content)

            # Submit the evaluation job.
            try:
                eval_result = subprocess.run(
                    ["sbatch", eval_sbatch_file],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                print(f"Submitted evaluation job for step {step}: {eval_result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting evaluation job for step {step}: {e.stderr.strip()}")
            finally:
                try:
                    os.remove(eval_sbatch_file)
                    print(f"Removed temporary file: {eval_sbatch_file}")
                except OSError as err:
                    print(f"Error removing file {eval_sbatch_file}: {err}")
    else:
        print("Evaluation jobs skipped as per configuration (cfg.run_eval is False).")

if __name__ == "__main__":
    run_experiment()
