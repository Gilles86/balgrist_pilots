#!/bin/bash
#SBATCH --job-name=fit_mapper
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_mapper_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# Fit numerosity GaussianPRFWithHRF per (subject, session) in T1w voxel space.
#
# Usage
# -----
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=balgrist3t fit_mapper.sh
#   sbatch --export=PARTICIPANT_LABEL=01,SESSION=balgrist7t,NATURAL_SPACE=1 fit_mapper.sh
#
# Optional overrides:
#   FMRIPREP_DERIV  (default: fmriprep)
#   SMOOTHED        (set to 1 to smooth 5mm before fit)
#   NATURAL_SPACE   (set to 1 to fit LogGaussianPRFWithHRF in natural numerosity)
#   MAX_ITER        (default: 5000)

set -euo pipefail

PARTICIPANT_LABEL="${PARTICIPANT_LABEL:?set PARTICIPANT_LABEL, e.g. 01}"
SESSION="${SESSION:?set SESSION, e.g. balgrist3t}"
FMRIPREP_DERIV="${FMRIPREP_DERIV:-fmriprep}"
SMOOTHED="${SMOOTHED:-0}"
NATURAL_SPACE="${NATURAL_SPACE:-0}"
MAX_ITER="${MAX_ITER:-5000}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-balgrist
REPO=$HOME/git/balgrist_pilots
ENV_PY=$HOME/data/conda/envs/balgrist_pilots/bin/python

ARGS=(
    "$PARTICIPANT_LABEL"
    "$SESSION"
    --bids-folder "$BIDS_FOLDER"
    --fmriprep-deriv "$FMRIPREP_DERIV"
    --max-iter "$MAX_ITER"
)
[ "$SMOOTHED" = "1" ]      && ARGS+=(--smoothed)
[ "$NATURAL_SPACE" = "1" ] && ARGS+=(--natural-space)

echo "fit_mapper: sub-${PARTICIPANT_LABEL} ses-${SESSION}  natural=${NATURAL_SPACE} smoothed=${SMOOTHED}"
echo "args: ${ARGS[*]}"

export PYTHONUNBUFFERED=1
cd "$REPO"
"$ENV_PY" -u "$REPO/balgrist/encoding_model/fit_mapper.py" "${ARGS[@]}"
