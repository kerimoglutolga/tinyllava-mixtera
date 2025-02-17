from multiprocessing import Pool
import subprocess

# Define a list of 128 mixtures (you should replace this with your actual mixtures)
mixtures = [f"mix_{i}_{mixture_type}" for i in range(1, 129) for mixture_type in ["best-effort"]]

def run_command(mixture):
    command = f"python launcher.py mixture={mixture} max_steps=5000"
    subprocess.run(command, shell=True)
    print(f"Running for {mixture}")

# Use multiprocessing to run the commands in parallel
if __name__ == '__main__':
    # Number of processes to run in parallel (you can adjust this based on your system's capabilities)
    num_processes = 32  # Adjust this based on how many processes you want to run concurrently

    # Create a Pool of processes and map the run_command function over the mixtures
    with Pool(processes=num_processes) as pool:
        pool.map(run_command, mixtures)