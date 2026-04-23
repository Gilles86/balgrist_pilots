#!/usr/bin/env python3
"""Fit an HRF-convolved 1-D Gaussian pRF to the numerosity mapper task.

Passive-viewing mapping paradigm -> we use braincoder's
``GaussianPRFWithHRF`` / ``LogGaussianPRFWithHRF`` fit directly on the
cleaned BOLD timeseries (NOT on single-trial betas).

Outputs (T1w voxel space):

    derivatives/encoding_model[.natural_space]/sub-<s>/ses-<ses>/func/
        sub-<s>_ses-<ses>_task-mapper_desc-{r2,mu,sd,amplitude,baseline}_space-T1w_pars.nii.gz
        sub-<s>_ses-<ses>_task-mapper_desc-meancleaned_space-T1w_bold.nii.gz

Usage
-----
    python fit_mapper.py 01 balgrist3t --bids-folder /shares/zne.uzh/gdehol/ds-balgrist
    python fit_mapper.py 01 balgrist7t --bids-folder /shares/zne.uzh/gdehol/ds-balgrist --smoothed
    python fit_mapper.py 01 sns3t      --bids-folder /shares/zne.uzh/gdehol/ds-balgrist --natural-space
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRFWithHRF, LogGaussianPRFWithHRF
from braincoder.optimize import ParameterFitter

from balgrist.utils.data import Subject, BIDS_FOLDER


def psc(img):
    """Convert a 4-D image to percent signal change (per-voxel mean = 100)."""
    mean = image.mean_img(img)
    return image.math_img(
        '100 * (img - mean[..., None]) / np.where(mean[..., None] == 0, 1, mean[..., None])',
        img=img, mean=mean)


def main(subject, session, bids_folder=BIDS_FOLDER,
         smoothed=False, natural_space=False, space='T1w',
         fmriprep_deriv='fmriprep',
         max_iter=5000, learning_rate=0.05):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    target_base = 'encoding_model'
    if smoothed:
        target_base += '.smoothed'
    if natural_space:
        target_base += '.natural_space'
    target_dir = sub.get_derivative_dir(session, target_base)
    print(f'[fit_mapper] sub-{subject} ses-{session}  →  {target_dir}')

    runs = sub.get_runs(session)
    tr = sub.get_tr(session)
    print(f'[fit_mapper] runs={runs}  TR={tr}')

    # ── confounds ─────────────────────────────────────────────────────────────
    fmriprep_conf = sub.get_fmriprep_confounds(session, runs=runs)
    response_hrf = sub.get_mapper_response_hrf(session, runs=runs)

    # ── paradigm (per-run, then mean across runs) ─────────────────────────────
    paradigm_per_run = sub.get_mapper_paradigm(session, natural_space=natural_space)

    # ── clean each run's BOLD in voxel space and mean across runs ─────────────
    bold_paths = sub.get_preprocessed_bold(session, runs=runs, space=space)
    conjunct_mask = sub.get_conjunct_mask(session, runs=runs, space=space)
    masker = NiftiMasker(mask_img=conjunct_mask)

    psc_dir = bids_folder / 'derivatives' / 'psc' / f'sub-{subject}' / f'ses-{session}' / 'func'
    psc_dir.mkdir(parents=True, exist_ok=True)

    per_run_data = []
    for run, bold_path in zip(runs, bold_paths):
        print(f'[fit_mapper]   cleaning run {run}: {bold_path.name}')
        d = image.load_img(str(bold_path))
        if smoothed:
            d = image.smooth_img(d, 5.0)
        d = psc(d)
        # Pad response_hrf to the confound stack if the run has entries.
        conf = fmriprep_conf[run].reset_index(drop=True)
        rh = response_hrf.xs(run, level='run').reset_index(drop=True)
        conf = pd.concat([conf, rh], axis=1).values
        d_cleaned = image.clean_img(
            d, confounds=conf, standardize=False, detrend=False,
            ensure_finite=True, t_r=tr)
        d_cleaned.to_filename(str(
            psc_dir / f'sub-{subject}_ses-{session}_task-mapper_run-{run}'
            f'_space-{space}_desc-psc_bold.nii.gz'))
        ts = masker.fit_transform(d_cleaned)  # (n_vols, n_vox)
        per_run_data.append(ts)

    # Stack → (n_runs, n_vols, n_vox) then mean across runs -> (n_vols, n_vox)
    arr = np.stack(per_run_data, axis=0)
    mean_ts = arr.mean(axis=0)
    n_vols = mean_ts.shape[0]
    frame_times = sub.get_frametimes(session, n_vols=n_vols)
    data = pd.DataFrame(mean_ts, index=pd.Index(frame_times, name='time'))
    data.columns.name = 'voxel'

    # Save mean cleaned volume for sanity-checking
    mean_img = masker.inverse_transform(mean_ts)
    mean_img.to_filename(str(
        target_dir / f'sub-{subject}_ses-{session}_task-mapper'
        f'_desc-meancleaned_space-{space}_bold.nii.gz'))

    # Paradigm for the mean: take run-1's (all runs have identical stimulus order
    # in the mapper; averaging across runs keeps the same timing).
    paradigm = paradigm_per_run.xs(runs[0], level='run').copy()
    paradigm.index = pd.Index(frame_times, name='time')

    # ── model ─────────────────────────────────────────────────────────────────
    hrf_model = SPMHRFModel(tr=tr, time_length=20)

    if natural_space:
        model = LogGaussianPRFWithHRF(hrf_model=hrf_model)
        mus = np.linspace(5, 80, 40, dtype=np.float32)
        sds = np.linspace(5, 30, 40, dtype=np.float32)
    else:
        model = GaussianPRFWithHRF(hrf_model=hrf_model)
        mus = np.log(np.linspace(5, 80, 40)).astype(np.float32)
        sds = np.log(np.linspace(2, 30, 40)).astype(np.float32)
    amplitudes = np.array([1.0], dtype=np.float32)
    baselines = np.array([0.0], dtype=np.float32)

    # Paradigm needs log == -inf mapped to the "no stimulus" sentinel used
    # historically in this pipeline (large negative).
    paradigm_array = paradigm.copy()
    paradigm_array[paradigm_array == 0.0] = -1e6

    optimizer = ParameterFitter(model, data, paradigm_array)

    grid_pars = optimizer.fit_grid(
        mus, sds, amplitudes, baselines, use_correlation_cost=True)
    print('[fit_mapper] grid:\n', grid_pars.describe())
    grid_pars = optimizer.refine_baseline_and_amplitude(grid_pars, n_iterations=2)
    print('[fit_mapper] refined:\n', grid_pars.describe())

    optimizer.fit(init_pars=grid_pars, learning_rate=learning_rate,
                  store_intermediate_parameters=False,
                  max_n_iterations=max_iter)
    print(f'[fit_mapper] optim R² median = {float(np.median(optimizer.r2)):.4f}')

    # ── write outputs ─────────────────────────────────────────────────────────
    def _save_float32(values_1d, path):
        """Write a parameter NIfTI with float32 dtype.

        NiftiMasker.inverse_transform inherits the mask's header dtype, so
        without an explicit dtype override a uint8/int8 mask would cause the
        output to be truncated on save.  This is exactly the bug we hit:
        amplitude values 0–4e6 were saved into an int8-typed NIfTI and
        quantised to near-zero on disk.
        """
        img = masker.inverse_transform(values_1d.astype(np.float32))
        img.header.set_data_dtype(np.float32)
        img.header['scl_slope'] = 1.0
        img.header['scl_inter'] = 0.0
        img.to_filename(str(path))
        return img

    r2_path = (target_dir
               / f'sub-{subject}_ses-{session}_task-mapper'
                 f'_desc-r2_space-{space}_pars.nii.gz')
    _save_float32(optimizer.r2.values, r2_path)
    print(f'[fit_mapper] wrote {r2_path.name}')

    for par_name, par_values in optimizer.estimated_parameters.T.iterrows():
        out = (target_dir
               / f'sub-{subject}_ses-{session}_task-mapper'
                 f'_desc-{par_name}_space-{space}_pars.nii.gz')
        _save_float32(par_values.sort_index().values, out)
        print(f'[fit_mapper] wrote desc-{par_name} '
              f'(min/median/max = {float(par_values.min()):.4f}/'
              f'{float(par_values.median()):.4f}/'
              f'{float(par_values.max()):.4f})')
    print('[fit_mapper] done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('session', help='session label (e.g. balgrist3t, balgrist7t, ibt7t, sns3t)')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--fmriprep-deriv', default='fmriprep')
    p.add_argument('--space', default='T1w')
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--natural-space', action='store_true')
    p.add_argument('--max-iter', type=int, default=5000)
    p.add_argument('--learning-rate', type=float, default=0.05)
    args = p.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv,
         space=args.space,
         smoothed=args.smoothed,
         natural_space=args.natural_space,
         max_iter=args.max_iter,
         learning_rate=args.learning_rate)
