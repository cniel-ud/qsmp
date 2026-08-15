#!/bin/bash -l
#
# SLURM array job: Snippet-Finder sweep for the ground-truth recovery
# experiment. One array task per random seed. Each task runs
# scripts/snippetfinder_recovery.py, which writes
#   results/recovery/snippetfinder_seed-<seed>.npz
# to be aggregated (together with the QSMP/sikmeans results from the GPU box)
# by scripts/aggregate_recovery.py.
#
# Submit e.g. for seeds 0..19:
#   sbatch --array=0-19 snippetfinder_recovery.qs
#
# Snippet-Finder is CPU + thread-parallel (STUMPY/Numba); no GPU needed.
#
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name=snippetfinder-recovery
#SBATCH --partition=standard
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19
#SBATCH --open-mode=append
#SBATCH --output=%x_%A-%a.out
#SBATCH --error=%x_%A-%a.err
#SBATCH --requeue
#SBATCH --mail-user=you@example.com
#SBATCH --mail-type=END,FAIL

echo "Job running on host: $HOSTNAME"
echo "Array task (seed): ${SLURM_ARRAY_TASK_ID}"
start=$(date "+%s")

# --- Environment ------------------------------------------------------------
# Activate a conda env that has STUMPY installed (adapt to your cluster). Some
# clusters require loading anaconda as a module first, e.g. `module load
# anaconda` (or `vpkg_require anaconda/<version>` on UD Caviness).
conda activate snippetfinder            # name or path of your STUMPY env

# Keep thread libraries within the allocated cores.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

ROOT_DIR="${HOME}/qsmp"                 # path to your clone of this repo
EXEC_DIR="${ROOT_DIR}/scripts"

# The snippetfinder conda env does not install the qsmp package (it only needs
# qsmp.datasets.morlet_signal, which is pure NumPy); make it importable.
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"

# --- Parameters (edit as needed) --------------------------------------------
SUBSEQ_LEN=512
K=6
PERCENTAGE=0.30          # ~S=154 at m=512, the value the paper found works
N_WAVES=1000

cd "${ROOT_DIR}" || exit 1

python "${EXEC_DIR}/snippetfinder_recovery.py" \
    --root "${ROOT_DIR}" \
    --seed "${SLURM_ARRAY_TASK_ID}" \
    --subseq-len "${SUBSEQ_LEN}" \
    --k "${K}" \
    --percentage "${PERCENTAGE}" \
    --n-waves "${N_WAVES}"

finish=$(date "+%s")
echo "Total runtime: $(($finish - $start)) s"
