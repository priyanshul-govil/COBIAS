import os

directory = './data'
pkl_files = [file for file in os.listdir(directory) if file.endswith('.pkl')]

from multiprocessing import Pool

def run_shell_script(script):
    os.system(script)

scripts = [f"sbatch get-scores.slurm {file}" for file in pkl_files]

with Pool(len(pkl_files)) as p:
    p.map(run_shell_script, scripts)

exit()