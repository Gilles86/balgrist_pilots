#!/usr/bin/env python3
"""Bayesian decoding of log(numerosity) from the mapper timeseries.

Mapper is passive block-design numerosity viewing (trial_type=='stimulation'
with n_dots ∈ {5,7,10,14,20,28,40,56,80}).  We decode at TR resolution using
leave-one-run-out cross-validation with an HRF-convolved Gaussian pRF
(``GaussianPRFWithHRF``).

Overview
--------
For each (subject, session):

  1. Clean BOLD timeseries (PSC + fmriprep + response regressor) → per-run
     (n_vols, n_vox) DataFrames.  Reuses the same cleaning pipeline as
     ``fit_mapper.py`` (deterministic given fmriprep outputs).
  2. Mask by an ROI NIfTI (default: NPCr).
  3. Leave-one-run-out loop.  In each fold:
       - Fit ``GaussianPRFWithHRF`` on training runs' concatenated cleaned
         timeseries (grid search + gradient descent).
       - Voxel selection: top-N by training R² (or all with train-R² > 0 if
         --n-voxels 0).
       - Fit Student-t residual noise model (``ResidualFitter``).
       - For each TR of the held-out run, evaluate P(log(n) | BOLD_t) over a
         fine log-numerosity grid.
  4. Concatenate folds, write TSV.

Notes
-----
- The HRF-convolved model means per-TR decoding is TEMPORALLY SMEARED by
  the HRF (peak posterior lags stimulus by ~5 s).  Post-hoc, you can
  aggregate the per-TR PDFs within each stimulus block (shifted by the HRF
  peak) to get per-block estimates.

Output
------
    derivatives/decoded_pdfs/sub-<s>/ses-<ses>/func/
        sub-<s>_ses-<ses>_mask-<mask_desc>_nvoxels-<n>[_noise-spherical]_pars.tsv

Rows: one per TR across all folds.
Index levels: session, run, tr, time_s, true_log_n (0 outside stimulus).
Columns: stimulus_range (log-numerosity values).
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.maskers import NiftiMasker

from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRFWithHRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils import get_rsq

from balgrist.utils.data import Subject, BIDS_FOLDER


def psc(img):
    mean = image.mean_img(img)
    return image.math_img(
        '100 * (img - mean[..., None]) / np.where(mean[..., None] == 0, 1, mean[..., None])',
        img=img, mean=mean)


def load_cleaned_timeseries(sub, session, masker, runs, space='T1w',
                             smoothed=False):
    """Clean BOLD per run, mask, and return a per-run dict of DataFrames."""
    tr = sub.get_tr(session)
    fmriprep_conf = sub.get_fmriprep_confounds(session, runs=runs)
    response_hrf = sub.get_mapper_response_hrf(session, runs=runs)
    bold_paths = sub.get_preprocessed_bold(session, runs=runs, space=space)

    cleaned_per_run = {}
    for run, bold_path in zip(runs, bold_paths):
        d = image.load_img(str(bold_path))
        if smoothed:
            d = image.smooth_img(d, 5.0)
        d = psc(d)
        conf = fmriprep_conf[run].reset_index(drop=True)
        rh = response_hrf.xs(run, level='run').reset_index(drop=True)
        conf = pd.concat([conf, rh], axis=1).values
        d_cleaned = image.clean_img(
            d, confounds=conf, standardize=False, detrend=False,
            ensure_finite=True, t_r=tr)
        ts = masker.transform(d_cleaned)          # (n_vols, n_vox)
        frame_times = sub.get_frametimes(session, n_vols=ts.shape[0])
        cleaned_per_run[run] = pd.DataFrame(
            ts.astype(np.float32),
            index=pd.Index(frame_times, name='time'))
    return cleaned_per_run, tr


def main(subject, session, bids_folder=BIDS_FOLDER,
         mask_path=None, mask_desc='NPCr',
         n_voxels=100, n_stimulus_grid=60,
         max_iter=2000, learning_rate=0.05,
         spherical_noise=False, natural_space=False,
         smoothed=False, space='T1w',
         fmriprep_deriv='fmriprep', debug=False):

    bids_folder = Path(bids_folder)
    sub = Subject(subject, bids_folder=bids_folder, fmriprep_deriv=fmriprep_deriv)

    if mask_path is None:
        mask_path = (bids_folder / 'derivatives' / 'masks'
                     / f'sub-{subject}' / 'anat'
                     / f'sub-{subject}_space-T1w_desc-{mask_desc}_mask.nii.gz')
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    if debug:
        max_iter = 100

    runs = sub.get_runs(session)
    print(f'[decode] sub-{subject} ses-{session}  runs={runs}  mask={mask_path.name}',
          flush=True)

    # Intersect ROI with the session's conjunct brain mask so we never ask the
    # masker for voxels that are outside the BOLD FOV.
    roi_img = image.load_img(str(mask_path))
    conjunct = sub.get_conjunct_mask(session, runs=runs, space=space)
    roi_img = image.resample_to_img(roi_img, conjunct, interpolation='nearest')
    combined = image.math_img('((m > 0) & (c > 0)).astype("int8")',
                              m=roi_img, c=conjunct)
    masker = NiftiMasker(mask_img=combined).fit()
    n_vox_roi = int(np.sum(masker.mask_img_.get_fdata() > 0))
    print(f'[decode] {n_vox_roi} voxels in ROI ∩ brain', flush=True)

    # Cleaned timeseries per run; paradigm per run.
    cleaned, tr = load_cleaned_timeseries(sub, session, masker, runs,
                                           space=space, smoothed=smoothed)
    paradigm_per_run = sub.get_mapper_paradigm(session, natural_space=natural_space)

    # Stimulus grid over log-numerosity (or natural).
    all_log_n = np.unique(paradigm_per_run.values)
    positive = all_log_n[all_log_n > 0]
    if natural_space:
        stim_lo, stim_hi = float(positive.min()), float(positive.max())
    else:
        stim_lo = float(positive.min()) - 0.3
        stim_hi = float(positive.max()) + 0.3
    stimulus_range = np.linspace(stim_lo, stim_hi, n_stimulus_grid,
                                 dtype=np.float32)
    print(f'[decode] stimulus grid: {n_stimulus_grid} points '
          f'({stim_lo:.2f} … {stim_hi:.2f})', flush=True)

    hrf_model = SPMHRFModel(tr=tr, time_length=20)

    all_pdfs = []
    fold_meta = []

    for test_run in runs:
        train_runs = [r for r in runs if r != test_run]
        print(f'\n[decode] fold: hold-out run-{test_run} (train={train_runs})',
              flush=True)

        train_data = pd.concat([cleaned[r] for r in train_runs],
                                keys=train_runs, names=['run'])
        test_data = cleaned[test_run].copy()

        # Per-TR paradigm with the same "zero → -1e6" sentinel used at fit time
        train_paradigm = pd.concat(
            [paradigm_per_run.xs(r, level='run') for r in train_runs],
            keys=train_runs, names=['run']).copy()
        test_paradigm = paradigm_per_run.xs(test_run, level='run').copy()

        for frame in (train_paradigm, test_paradigm):
            frame.loc[frame['n_dots'] == 0.0, 'n_dots'] = -1e6

        # ── fit encoding model on training runs ───────────────────────────────
        model = GaussianPRFWithHRF(hrf_model=hrf_model)
        fitter = ParameterFitter(model, train_data, train_paradigm)

        mus = np.log(np.linspace(5, 80, 40)).astype(np.float32) \
              if not natural_space else np.linspace(5, 80, 40, dtype=np.float32)
        sds = np.log(np.linspace(2, 30, 20)).astype(np.float32) \
              if not natural_space else np.linspace(5, 30, 20, dtype=np.float32)
        amps = np.array([1.0], dtype=np.float32)
        baselines = np.array([0.0], dtype=np.float32)

        grid = fitter.fit_grid(mus, sds, amps, baselines,
                                use_correlation_cost=True)
        grid = fitter.refine_baseline_and_amplitude(grid, n_iterations=2)
        fitter.fit(init_pars=grid, learning_rate=learning_rate,
                   max_n_iterations=max_iter,
                   store_intermediate_parameters=False)
        pars = fitter.estimated_parameters
        train_r2 = fitter.r2
        print(f'[decode]   train R² median = {float(np.median(train_r2)):.4f}',
              flush=True)

        # ── voxel selection ───────────────────────────────────────────────────
        if n_voxels == 0:
            sel = train_r2[train_r2 > 0].index
            print(f'[decode]   {len(sel)} voxels (train R² > 0)', flush=True)
        else:
            sel = train_r2.sort_values(ascending=False).index[:n_voxels]
            print(f'[decode]   {len(sel)} voxels (top-{n_voxels} by R²; '
                  f'min R²={float(train_r2.loc[sel].min()):.3f})', flush=True)
        if len(sel) < 10:
            print(f'[decode]   WARNING: only {len(sel)} voxels selected — '
                  f'skipping fold (decoding unreliable)', flush=True)
            continue

        pars_sel = pars.loc[sel]
        train_data_sel = train_data[sel]
        test_data_sel = test_data[sel]
        fold_meta.append({'session': session, 'run': test_run,
                           'n_voxels_selected': len(sel),
                           'train_r2_median_selected':
                               float(train_r2.loc[sel].median())})

        # ── fit residual noise model on training data ─────────────────────────
        model_sel = GaussianPRFWithHRF(hrf_model=hrf_model)
        model_sel.init_pseudoWWT(stimulus_range, pars_sel)
        n_iter_noise = 200 if debug else 5000
        residfit = ResidualFitter(model_sel, train_data_sel, train_paradigm,
                                   parameters=pars_sel)
        omega, dof = residfit.fit(
            init_sigma2=0.1, init_dof=10.0, method='t',
            learning_rate=0.05, spherical=spherical_noise,
            max_n_iterations=n_iter_noise)
        print(f'[decode]   noise dof={float(dof):.1f}', flush=True)

        # ── per-TR posterior for held-out run ─────────────────────────────────
        pdf = model_sel.get_stimulus_pdf(test_data_sel, stimulus_range,
                                          parameters=pars_sel,
                                          omega=omega, dof=dof,
                                          normalize=False)
        pdf.columns = stimulus_range

        true_log_n = test_paradigm['n_dots'].values.copy()
        true_log_n[true_log_n < -1e5] = 0.0   # restore 0-sentinel
        pdf.index = pd.MultiIndex.from_arrays(
            [
                np.full(len(pdf), session, dtype=object),
                np.full(len(pdf), test_run, dtype=int),
                np.arange(len(pdf), dtype=int),
                test_data.index.values.astype(np.float32),
                true_log_n.astype(np.float32),
            ],
            names=['session', 'run', 'tr', 'time_s', 'true_log_n'])
        all_pdfs.append(pdf)

    pdfs = pd.concat(all_pdfs).sort_index()

    # ── write output ──────────────────────────────────────────────────────────
    noise_label = 'spherical' if spherical_noise else 'full'
    tag = '.smoothed' if smoothed else ''
    out_dir = (bids_folder / 'derivatives' / f'decoded_pdfs{tag}'
               / f'sub-{subject}' / f'ses-{session}' / 'func')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_fn = (out_dir /
              f'sub-{subject}_ses-{session}_mask-{mask_desc}'
              f'_nvoxels-{n_voxels}_noise-{noise_label}_pars.tsv')
    pdfs.to_csv(out_fn, sep='\t')
    print(f'\n[decode] wrote {out_fn}')

    meta_fn = out_fn.with_name(out_fn.stem.replace('_pars', '_meta') + '.tsv')
    pd.DataFrame(fold_meta).to_csv(meta_fn, sep='\t', index=False)
    print(f'[decode] wrote {meta_fn.name}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('session')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--fmriprep-deriv', default='fmriprep')
    p.add_argument('--mask-path', default=None,
                   help='Explicit ROI NIfTI (overrides default NPCr lookup)')
    p.add_argument('--mask-desc', default='NPCr',
                   help='Short label for mask in output filename (default: NPCr)')
    p.add_argument('--n-voxels', type=int, default=100,
                   help='Top-N voxels by training R² (0 = all with R²>0)')
    p.add_argument('--n-stimulus-grid', type=int, default=60)
    p.add_argument('--max-iter', type=int, default=2000)
    p.add_argument('--learning-rate', type=float, default=0.05)
    p.add_argument('--spherical-noise', action='store_true')
    p.add_argument('--natural-space', action='store_true')
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    main(args.subject, args.session,
         bids_folder=args.bids_folder,
         fmriprep_deriv=args.fmriprep_deriv,
         mask_path=args.mask_path,
         mask_desc=args.mask_desc,
         n_voxels=args.n_voxels,
         n_stimulus_grid=args.n_stimulus_grid,
         max_iter=args.max_iter,
         learning_rate=args.learning_rate,
         spherical_noise=args.spherical_noise,
         natural_space=args.natural_space,
         smoothed=args.smoothed,
         debug=args.debug)
