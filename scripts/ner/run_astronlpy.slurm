#!/usr/bin/env bash
#SBATCH --job-name=NER-astroNLPy
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-48%3
#SBATCH --output=ner-astronlpy-%A_%a.out
#SBATCH --error=ner-astronlpy-%A_%a.err

set -euo pipefail

cd /obs/sdurna/dossier

module purge
module load python/3.12.12
source .venv/bin/activate

export HF_HOME=/data/sdurna/huggingface
export ASTRONLPY_NER_MODEL_PATH=/data/sdurna/huggingface/astroNLPy-ner
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

mapfile -d '' -t cleaned_files < <(
    find resources/group_pdfs -type f -name '*cleaned.txt' -print0 | sort -z
)

task_id="${SLURM_ARRAY_TASK_ID:?Submit this script as a SLURM array}"
if (( task_id < 0 || task_id >= ${#cleaned_files[@]} )); then
    echo "Invalid task $task_id: only ${#cleaned_files[@]} inputs found" >&2
    exit 1
fi

input_path="${cleaned_files[$task_id]}"
echo "Node: $(hostname)"
echo "Input: $input_path"
nvidia-smi

python scripts/ner/ner_astronlpy.py "$input_path"
