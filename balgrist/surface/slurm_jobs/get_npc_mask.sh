#!/bin/bash
#SBATCH --job-name=get_npc_mask
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/get_npc_mask_%x_%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

# Transform fsaverage NPC label → T1w volume mask for one subject.
#
# Usage:
#   sbatch --export=PARTICIPANT_LABEL=01 get_npc_mask.sh            # NPC bilateral (default)
#   sbatch --export=PARTICIPANT_LABEL=01,HEMI=R get_npc_mask.sh     # right only
#   sbatch --export=PARTICIPANT_LABEL=01,ROI=NPC1 get_npc_mask.sh   # NPC1

set -euo pipefail

PARTICIPANT_LABEL="${PARTICIPANT_LABEL:?set PARTICIPANT_LABEL}"
ROI="${ROI:-NPC}"
HEMI="${HEMI:-LR}"

BIDS_FOLDER=/shares/zne.uzh/gdehol/ds-balgrist
REPO=$HOME/git/balgrist_pilots
ENV_PY=$HOME/data/conda/envs/balgrist_pilots/bin/python

# FreeSurfer from the fmriprep apptainer-sandbox (no apptainer exec needed)
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer
export FS_LICENSE=$HOME/freesurfer/license.txt
export PATH=$FREESURFER_HOME/bin:$PATH
export SUBJECTS_DIR=$BIDS_FOLDER/derivatives/freesurfer

export PYTHONUNBUFFERED=1

echo "get_npc_mask: sub-${PARTICIPANT_LABEL} roi=${ROI} hemi=${HEMI}"
echo "FREESURFER_HOME=$FREESURFER_HOME"

"$ENV_PY" -u "$REPO/balgrist/surface/get_npc_mask.py" \
    "$PARTICIPANT_LABEL" \
    --roi "$ROI" --hemi "$HEMI" \
    --bids-folder "$BIDS_FOLDER"
