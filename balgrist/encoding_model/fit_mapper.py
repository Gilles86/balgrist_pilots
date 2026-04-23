#!/usr/bin/env python3
"""Fit an HRF-convolved 1-D Gaussian pRF to the numerosity mapper task.

Passive-viewing paradigm → braincoder's ``GaussianPRFWithHRF`` (log-space) or
``LogGaussianPRFWithHRF`` (natural-space), fit directly on the cleaned BOLD
timeseries (NOT on single-trial betas).

Outputs (T1w voxel space):
    derivatives/encoding_model[.smoothed][.natural_space]/sub-<s>/ses-<ses>/func/
        sub-<s>_ses-<ses>_task-mapper_desc-{r2,mu,sd,amplitude,baseline}_space-T1w_pars.nii.gz
        sub-<s>_ses-<ses>_task-mapper_desc-meancleaned_space-T1w_bold.nii.gz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

from braincoder.hrf import SPMHRFModel
from braincoder.models import GaussianPRFWithHRF, LogGaussianPRFWithHRF
from braincoder.optimize import ParameterFitter

from balgrist.utils.data import Subject, BIDS_FOLDER, save_float32


NO_STIM = np.float32(-1e6)  # "no stimulus shown" sentinel for the 1-D paradigm


def main(subject, session, bids_folder=BIDS_FOLDER,
         smoothed=False, natural_space=False, space='T1w',
         fmriprep_deriv='fmriprep', max_iter=5000, learning_rate=0.05):

    sub = Subject(subject, bids_folder=Path(bids_folder),
                  fmriprep_deriv=fmriprep_deriv)
    runs = sub.get_runs(session)
    tr = sub.get_tr(session)

    tag = 'encoding_model' + ('.smoothed' if smoothed else '') \
                           + ('.natural_space' if natural_space else '')
    target_dir = sub.get_derivative_dir(session, tag)
    print(f'[fit_mapper] sub-{subject} ses-{session}  runs={runs}  TR={tr}  '
          f'→ {target_dir}', flush=True)

    # ── clean each run, stack into (n_runs, n_vols, n_vox) ────────────────────
    masker = NiftiMasker(mask_img=sub.get_conjunct_mask(session, runs=runs, space=space))
    stacked = np.stack([masker.fit_transform(sub.get_cleaned_bold(
        session, run, space=space, smoothed=smoothed)) for run in runs])
    mean_ts = stacked.mean(axis=0)
    frame_times = sub.get_frametimes(session, n_vols=mean_ts.shape[0])
    data = pd.DataFrame(mean_ts, index=pd.Index(frame_times, name='time'))
    data.columns.name = 'voxel'

    # sanity-check run-mean timeseries (n_vols × n_vox → 4-D NIfTI)
    masker.inverse_transform(mean_ts).to_filename(str(
        target_dir / f'sub-{subject}_ses-{session}_task-mapper'
                     f'_desc-meancleaned_space-{space}_bold.nii.gz'))

    # ── paradigm: one run's timing (mapper is identical across runs) ──────────
    paradigm = sub.get_mapper_paradigm(session, natural_space=natural_space
                                       ).xs(runs[0], level='run').copy()
    paradigm.index = pd.Index(frame_times, name='time')
    paradigm[paradigm == 0.0] = NO_STIM

    # ── model + grid ──────────────────────────────────────────────────────────
    hrf = SPMHRFModel(tr=tr, time_length=20)
    if natural_space:
        model = LogGaussianPRFWithHRF(hrf_model=hrf)
        mus = np.linspace(5, 80, 40, dtype=np.float32)
        sds = np.linspace(5, 30, 40, dtype=np.float32)
    else:
        model = GaussianPRFWithHRF(hrf_model=hrf)
        mus = np.log(np.linspace(5, 80, 40)).astype(np.float32)
        sds = np.log(np.linspace(2, 30, 40)).astype(np.float32)

    opt = ParameterFitter(model, data, paradigm)
    pars = opt.fit_grid(mus, sds, [np.float32(1.0)], [np.float32(0.0)],
                        use_correlation_cost=True)
    pars = opt.refine_baseline_and_amplitude(pars, n_iterations=2)
    opt.fit(init_pars=pars, learning_rate=learning_rate,
            store_intermediate_parameters=False, max_n_iterations=max_iter)
    print(f'[fit_mapper] R² median = {float(np.median(opt.r2)):.4f}', flush=True)

    # ── save parameter maps (float32, sorted by voxel index) ──────────────────
    def _out(desc):
        return target_dir / (f'sub-{subject}_ses-{session}_task-mapper'
                             f'_desc-{desc}_space-{space}_pars.nii.gz')

    save_float32(masker, opt.r2.values, _out('r2'))
    for par_name, par_values in opt.estimated_parameters.T.iterrows():
        save_float32(masker, par_values.sort_index().values, _out(par_name))
        print(f'[fit_mapper]   desc-{par_name}: '
              f'min/median/max = {float(par_values.min()):.4f}/'
              f'{float(par_values.median()):.4f}/{float(par_values.max()):.4f}',
              flush=True)
    print('[fit_mapper] done.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('subject')
    p.add_argument('session')
    p.add_argument('--bids-folder', default=str(BIDS_FOLDER))
    p.add_argument('--fmriprep-deriv', default='fmriprep')
    p.add_argument('--space', default='T1w')
    p.add_argument('--smoothed', action='store_true')
    p.add_argument('--natural-space', action='store_true')
    p.add_argument('--max-iter', type=int, default=5000)
    p.add_argument('--learning-rate', type=float, default=0.05)
    a = p.parse_args()
    main(a.subject, a.session, bids_folder=a.bids_folder,
         fmriprep_deriv=a.fmriprep_deriv, space=a.space,
         smoothed=a.smoothed, natural_space=a.natural_space,
         max_iter=a.max_iter, learning_rate=a.learning_rate)
