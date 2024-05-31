for model in $(cat panel_models.txt); do
    sbatch generate.slurm $model
done
