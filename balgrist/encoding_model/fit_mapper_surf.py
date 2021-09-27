import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
import argparse
import os
import os.path as op
import sys
# from risk_experiment.utils import get_behavior, get_fmriprep_confounds, get_retroicor_confounds, get_tr, get_mapper_response_hrf, get_runs
# from risk_experiment.utils import get_surf_data, get_mapper_paradigm, write_gifti, get_target_dir
# from risk_experiment.utils.surface import transform_data

from balgrist.utils import (get_fmriprep_confounds, get_mapper_response_hrf,
                            get_surf_data, get_mapper_paradigm, get_target_dir,
                            get_tr, write_gifti)
from balgrist.utils.surface import transform_data
import pandas as pd

from nilearn import signal
from nipype.interfaces.freesurfer.utils import SurfaceTransform

from braincoder.models import GaussianPRFWithHRF, GaussianPRF
from braincoder.hrf import SPMHRFModel
from braincoder.optimize import ParameterFitter


def main(subject, session, bids_folder, smoothed=False, concatenate=False):

    print('yo')
    target_dir = 'encoding_model'

    if smoothed:
        target_dir += '.smoothed'
        print('SMOOTHED DATA')

    if concatenate:
        print('CONCATENATING')
        target_dir += '.concatenated'

    target_dir = get_target_dir(subject, session, bids_folder, target_dir)

    # Create confounds
    fmriprep_confounds = get_fmriprep_confounds(subject, session, bids_folder)
    response_hrf = get_mapper_response_hrf(subject, session, bids_folder)

    confounds = pd.concat((fmriprep_confounds, response_hrf), axis=1)

    # Get surface data
    surf = get_surf_data(subject, session, bids_folder, smoothed=smoothed)

    # # clean surface data
    surf_cleaned = pd.DataFrame(None, columns=surf.columns)
    for run, d in surf.groupby(['run']):
        d_cleaned = signal.clean(
            d.values, confounds=confounds.loc[run].values, standardize='psc', detrend=False)
        surf_cleaned = surf_cleaned.append(
            pd.DataFrame(d_cleaned, columns=d.columns))

    surf_cleaned.index = surf.index

    if concatenate:
        runs = get_runs(subject, session)
        paradigm = pd.concat([get_mapper_paradigm(subject, session, bids_folder, run) for run in runs],
                             keys=runs,
                             names=['run'])
        data = surf_cleaned

    else:
        paradigm = get_mapper_paradigm(subject, session, bids_folder)
        avg_data = surf_cleaned.groupby(level=1, axis=0).mean()
        data = avg_data

    paradigm[paradigm == 0.0] = -1e6

    print('DATA: ', data, data.shape)

    print('DATA MEAN', data.mean(0))
    print('DATA std', data.std(0))

    hrf_model = SPMHRFModel(tr=get_tr(subject, session), time_length=20)
    model = GaussianPRFWithHRF(hrf_model=hrf_model)

    # SET UP GRID
    mus = np.linspace(0, np.log(80), 20, dtype=np.float32)
    sds = np.linspace(.01, 2, 15, dtype=np.float32)
    amplitudes = np.linspace(1e-6, 10, 15, dtype=np.float32)
    baselines = np.linspace(-2., 0., 4, endpoint=True, dtype=np.float32)

    optimizer = ParameterFitter(model, data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines)

    r2 = optimizer.get_rsq(grid_parameters)

    target_fn = op.join(
        target_dir, f'sub-{subject}_ses-{session}_desc-r2.grid_space-fsnative') + '_hemi-{hemi}.func.gii'

    write_gifti(subject, session, bids_folder, 'fsnative', r2, target_fn)
    transform_data(target_fn, f'sub-{subject}',
                   bids_folder, target_subject='fsaverage')

    for par, values in grid_parameters.T.iterrows():
        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_desc-{par}.grid_space-fsnative') + '_hemi-{hemi}.func.gii'
        # write_gifti(subject, session, bids_folder, 'fsnative', values, target_fn)
        # transform_data(target_fn, f'sub-{subject}', bids_folder, target_subject='fsaverage')

    optimizer.fit(init_pars=grid_parameters.values, learning_rate=.01,
                  store_intermediate_parameters=False, max_n_iterations=10000)
    print(optimizer.r2)

    target_fn = op.join(
        target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-fsnative') + '_hemi-{hemi}.func.gii'
    write_gifti(subject, session, bids_folder,
                'fsnative', optimizer.r2, target_fn)
    transform_data(target_fn, f'sub-{subject}',
                   bids_folder, target_subject='fsaverage')

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        target_fn = op.join(
            target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-fsnative') + '_hemi-{hemi}.func.gii'
        write_gifti(subject, session, bids_folder,
                    'fsnative', values, target_fn)
        transform_data(target_fn, f'sub-{subject}',
                       bids_folder, target_subject='fsaverage')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--concatenate', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder,
            smoothed=args.smoothed,
         concatenate=args.concatenate)
