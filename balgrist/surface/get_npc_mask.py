#!/usr/bin/env python3
"""Transform fsaverage NPC surface labels to a T1w volumetric mask (per subject).

Adapts neural_priors/surface/get_npc_mask.py for the balgrist_pilots layout:
freesurfer recon-all outputs live at ``derivatives/freesurfer/`` (not under
``fmriprep/sourcedata/``), and T1w + masks are subject-level (not per-session).

FreeSurfer binaries (``mri_surf2surf``) are required for the fsaverage →
fsnative surface transform. On the cluster, use the fmriprep apptainer-sandbox
copy via the SLURM wrapper ``slurm_jobs/get_npc_mask.sh`` which sets
``FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.5/opt/freesurfer``.

Layout assumed:
    derivatives/freesurfer/{fsaverage,sub-<subject>}/
    derivatives/fmriprep/sub-<subject>/anat/sub-<subject>_desc-preproc_T1w.nii.gz
    derivatives/surface_masks/desc-<ROI>_{L,R}_space-fsaverage_hemi-{lh,rh}.label.gii

Output:
    derivatives/masks/sub-<subject>/anat/
        sub-<subject>_space-T1w_desc-<roi>_mask.nii.gz   (bilateral, if both hemis)
        sub-<subject>_space-T1w_desc-<roi>l_mask.nii.gz  (left, if --hemi includes L)
        sub-<subject>_space-T1w_desc-<roi>r_mask.nii.gz  (right, if --hemi includes R)

Usage
-----
    python get_npc_mask.py 01 --hemi R                  # NPCr only
    python get_npc_mask.py 01 --roi NPC --hemi LR       # NPC bilateral
"""
import argparse
import os
from pathlib import Path

import numpy as np
from nilearn import surface
from nipype.interfaces.freesurfer import SurfaceTransform

from neuropythy.freesurfer import subject as fs_subject
from neuropythy.io import load, save
from neuropythy.mri import image_clear, to_image


def _transform(in_file, out_file, fs_hemi, target_subject, subjects_dir):
    os.environ.setdefault('SUBJECTS_DIR', str(subjects_dir))
    sxfm = SurfaceTransform(subjects_dir=str(subjects_dir))
    sxfm.inputs.source_file = str(in_file)
    sxfm.inputs.out_file = str(out_file)
    sxfm.inputs.source_subject = 'fsaverage'
    sxfm.inputs.target_subject = target_subject
    sxfm.inputs.hemi = fs_hemi
    sxfm.run()


def main(subject, bids_folder, roi='NPC', hemis=('L', 'R')):
    bids_folder = Path(bids_folder)
    subjects_dir = bids_folder / 'derivatives' / 'freesurfer'
    fs_sub_name = f'sub-{subject}'
    fs_dir = subjects_dir / fs_sub_name
    t1w = (bids_folder / 'derivatives' / 'fmriprep' / f'sub-{subject}' / 'anat'
           / f'sub-{subject}_desc-preproc_T1w.nii.gz')
    labels_dir = bids_folder / 'derivatives' / 'surface_masks'

    for p in (fs_dir, t1w):
        if not p.exists():
            raise FileNotFoundError(p)

    out_dir = (bids_folder / 'derivatives' / 'masks'
               / f'sub-{subject}' / 'anat')
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f'sub-{subject}_space-T1w_desc-{roi}'

    mask_data = {'L': None, 'R': None}
    for hemi in hemis:
        fs_hemi = 'lh' if hemi == 'L' else 'rh'
        src = labels_dir / f'desc-{roi}_{hemi}_space-fsaverage_hemi-{fs_hemi}.label.gii'
        if not src.exists():
            raise FileNotFoundError(src)
        dst = fs_dir / 'surf' / f'{fs_hemi}.{roi}.mgz'
        print(f'Transforming {hemi} {roi}: fsaverage → {fs_sub_name}', flush=True)
        _transform(src, dst, fs_hemi, fs_sub_name, subjects_dir)
        data = surface.load_surf_data(str(dst))
        mask_data[hemi] = (data > 0).astype(np.float32)
        print(f'  {hemi}: {int(mask_data[hemi].sum())} fsnative vertices')

    sub = fs_subject(str(fs_dir))
    im = load(str(t1w))
    im = to_image(image_clear(im, fill=0.0), dtype=np.int32)

    lh, rh = mask_data['L'], mask_data['R']

    if lh is not None and rh is not None:
        print('Projecting bilateral mask → T1w volume', flush=True)
        vol = sub.cortex_to_image((lh, rh), im, hemi=None,
                                  method='nearest', fill=0.0)
        save(str(out_dir / f'{prefix}_mask.nii.gz'), vol)
        print(f'  wrote {prefix}_mask.nii.gz')

    if lh is not None:
        print('Projecting left-hemisphere mask → T1w volume', flush=True)
        vol = sub.cortex_to_image(lh, im, hemi='lh',
                                  method='nearest', fill=0.0)
        save(str(out_dir / f'{prefix}l_mask.nii.gz'), vol)
        print(f'  wrote {prefix}l_mask.nii.gz')

    if rh is not None:
        print('Projecting right-hemisphere mask → T1w volume', flush=True)
        zero_lh = np.zeros_like(lh) if lh is not None \
                  else np.zeros(sub.lh.vertex_count)
        vol = sub.cortex_to_image((zero_lh, rh), im, hemi=None,
                                  method='nearest', fill=0.0)
        save(str(out_dir / f'{prefix}r_mask.nii.gz'), vol)
        print(f'  wrote {prefix}r_mask.nii.gz')

    print(f'Done. Masks in {out_dir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('--roi', default='NPC')
    p.add_argument('--hemi', choices=['L', 'R', 'LR'], default='LR')
    p.add_argument('--bids-folder',
                   default='/shares/zne.uzh/gdehol/ds-balgrist')
    args = p.parse_args()
    hemis = ['L', 'R'] if args.hemi == 'LR' else [args.hemi]
    main(args.subject, args.bids_folder, roi=args.roi, hemis=hemis)
