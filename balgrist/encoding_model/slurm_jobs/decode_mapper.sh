#!/bin/bash
#SBATCH --job-name=decode_mapper
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/decode_mapper_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# LOROCV Bayesian decoding of log(numerosity) from the mapper timeseries.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=balgrist3t decode_mapper.sh
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=balgrist7t,SMOOTHED=1,N_VOXELS=200 decode_mapper.sh
#
# Optional overrides:
#   MASK_DESC       (default: NPCr)
#   N_VOXELS        (default: 100; 0 → all train-R²>0)
#   SPHERICAL_NOISE (1 → isotropic; default full)
#   NATURAL_SPACE   (1 → fit in numerosity-count space)
#   SMOOTHED        (1 → 5mm smooth before clean)
#   MAX_ITER        (default: 2000)

set -euo pipefail

PARTICIPANT_LABEL="${PARTICIPANT_LABEL:?set PARTICIPANT_LABEL}"
SESSION="${SESSION:?set SESSION}"
MASK_DESC="${MASK_DESC:-NPCr}"
N_VOXELS="${N_VOXELS:-100}"
SPHERICAL_NOISE="${SPHERICAL_NOISE:-0}"
NATURAL_SPACE="${NATURAL_SPACE:-0}"
SMOOTHED="${SMOOTHED:-0}"
MAX_ITER="${MAX_ITER:-2000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-balgrist
REPO=$HOME/git/balgrist_pilots
ENV_PY=$HOME/data/conda/envs/balgrist_pilots/bin/python

ARGS=(
    "$PARTICIPANT_LABEL" "$SESSION"
    --bids-folder "$BIDS_FOLDER"
    --mask-desc "$MASK_DESC"
    --n-voxels "$N_VOXELS"
    --max-iter "$MAX_ITER"
)
[ "$SPHERICAL_NOISE" = "1" ] && ARGS+=(--spherical-noise)
[ "$NATURAL_SPACE"   = "1" ] && ARGS+=(--natural-space)
[ "$SMOOTHED"        = "1" ] && ARGS+=(--smoothed)

echo "decode_mapper: sub-${PARTICIPANT_LABEL} ses-${SESSION}  mask=${MASK_DESC}  n_voxels=${N_VOXELS}"
echo "args: ${ARGS[*]}"

export PYTHONUNBUFFERED=1
"$ENV_PY" -u "$REPO/balgrist/encoding_model/decode_mapper.py" "${ARGS[@]}"
