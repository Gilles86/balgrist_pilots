import pandas as pd
import numpy as np
import argparse
import os
import os.path as op
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix


def main(subject, session, sourcedata, fwhm=0.0):
    runs = range(1, 5)
    behavior = []

    for run in runs:
        behavior.append(pd.read_table(op.join(
            sourcedata, f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_events.tsv')))

    behavior = pd.concat(behavior, keys=runs, names=['run'])
    behavior['subject'] = subject
    behavior = behavior.reset_index().set_index(
        ['subject', 'run', 'trial_type'])

    ims = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_space-T1w_desc-preproc_bold.nii.gz') for run in runs]

    fmriprep_confounds_include = ['global_signal', 'dvars', 'framewise_displacement', 'trans_x',
                                  'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z',
                                  'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03', 'cosine00', 'cosine01', 'cosine02',
                                  'non_steady_state_outlier00', 'non_steady_state_outlier01', 'non_steady_state_outlier02']
    fmriprep_confounds = [
        op.join(sourcedata, f'derivatives/fmriprep/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-confounds_timeseries.tsv') for run in runs]
    fmriprep_confounds = [pd.read_table(
        cf)[fmriprep_confounds_include] for cf in fmriprep_confounds]

    # retroicor_confounds = [
        # op.join(sourcedata, f'derivatives/physiotoolbox/sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-mapper_run-{run}_desc-retroicor_timeseries.tsv') for run in runs]
    # retroicor_confounds = [pd.read_table(
        # cf, header=None, usecols=range(18)) for cf in retroicor_confounds]

    # confounds = [pd.concat((rcf, fcf), axis=1) for rcf, fcf in zip(retroicor_confounds, fmriprep_confounds)]
    confounds = fmriprep_confounds
    confounds = [c.fillna(method='bfill') for c in confounds]

    mask = op.join(sourcedata, 'derivatives', 'fmriprep', f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-mapper_run-1_space-T1w_desc-brain_mask.nii.gz')

    model = FirstLevelModel(t_r=2.3, slice_time_ref=.5, signal_scaling=False, drift_model=None, 
            mask_img=mask,
                        smoothing_fwhm=fwhm)
    responses = behavior.xs('response', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'trial_type']]
    responses['duration'] = 0.0
    responses = responses[responses.onset > 0]
    stimulation = behavior.xs('stimulation', 0, 'trial_type', drop_level=False).reset_index('trial_type')[['onset', 'duration', 'n_dots', 'trial_type']]
    stimulation_mod = stimulation.copy()
    stimulation_mod['modulation'] = np.log(stimulation_mod['n_dots'])
    stimulation_mod['modulation'] = (stimulation_mod['modulation'] - stimulation_mod['modulation'].mean(0)) / stimulation_mod['modulation'].std()
    stimulation_mod['trial_type'] = 'stimulation*n_dots'

    targets = behavior.xs('targets', 0, 'trial_type')
    hazard_regressor = targets[~targets.hazard1.isnull()].copy()
    hazard_regressor['modulation'] = hazard_regressor['hazard1']
    hazard_regressor['modulation'] = (hazard_regressor['modulation'] - hazard_regressor['modulation'].mean()) / hazard_regressor['modulation'].std()
    hazard_regressor['trial_type'] = 'hazard1'
    hazard_regressor['duration'] = 0.0

    hazard_regressor = hazard_regressor[['trial_type', 'duration', 'modulation', 'onset']]

    events = pd.concat((responses, stimulation, stimulation_mod, hazard_regressor))
    events['modulation'].fillna(1.0, inplace=True)
    model.fit(ims, [r for _, r in events.groupby(['run'])], confounds)

    zmaps = {}

    results_dir = op.join(sourcedata, 'derivatives', 'glm', 'simple_mapper', f'sub-{subject}', f'ses-{session}')

    if not op.exists(results_dir):
        os.makedirs(results_dir)

    for key in ['response', 'stimulation', 'stimulation*n_dots', 'hazard1']:
        zmaps[key] = model.compute_contrast(key)
    
        zmaps[key].to_filename(op.join(results_dir, f'sub-{subject}_ses-{session}_fwhm-{fwhm}_zmap_{key}.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--fwhm', default=0.0, type=float)
    parser.add_argument('--sourcedata', default='/data')
    args = parser.parse_args()

    main(args.subject, args.session, fwhm=args.fwhm, sourcedata=args.sourcedata)
