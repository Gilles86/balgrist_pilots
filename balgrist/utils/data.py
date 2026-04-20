"""Data access for the balgrist_pilots numerosity-mapper dataset.

The dataset has four scanner sessions per subject:
    ses-balgrist3t   — Siemens Prisma 3T, SNS Lab (Balgrist campus)
    ses-balgrist7t   — Siemens Terra 7T, SNS Lab (Balgrist campus)
    ses-ibt7t        — 7T, IBT
    ses-sns3t        — 3T, SNS Lab

Task: ``mapper`` — passive viewing of numerosity stimuli.
Events columns: onset, n_dots, duration, trial_type, rt, responded, isi,
                hazard1, hazard2  (trial_type == 'stimulation' holds stimuli).
"""
import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from nilearn.glm.first_level import make_first_level_design_matrix


BIDS_FOLDER = Path('/data/ds-balgrist')

SESSIONS = ('balgrist3t', 'balgrist7t', 'ibt7t', 'sns3t')


class Subject:
    """Data access for a single balgrist_pilots subject.

    Parameters
    ----------
    subject_id : str
        Subject label without ``sub-`` prefix (e.g. ``'01'`` or ``'001'``).
    bids_folder : str or Path
    fmriprep_deriv : str, default ``'fmriprep'``
    """

    def __init__(self, subject_id, bids_folder=BIDS_FOLDER,
                 fmriprep_deriv='fmriprep'):
        self.subject_id = str(subject_id)
        self.bids_folder = Path(bids_folder)
        self.fmriprep_deriv = fmriprep_deriv

    # ── paths ──────────────────────────────────────────────────────────────────

    @property
    def _fmriprep_dir(self):
        return self.bids_folder / 'derivatives' / self.fmriprep_deriv

    def _func_dir(self, session):
        return (self._fmriprep_dir / f'sub-{self.subject_id}'
                / f'ses-{session}' / 'func')

    def _raw_func_dir(self, session):
        return (self.bids_folder / f'sub-{self.subject_id}'
                / f'ses-{session}' / 'func')

    # ── sessions & runs ────────────────────────────────────────────────────────

    def get_sessions(self):
        """Return sorted list of session labels available in fmriprep output."""
        sub_dir = self._fmriprep_dir / f'sub-{self.subject_id}'
        sessions = []
        for d in sub_dir.iterdir():
            m = re.match(r'ses-(.+)$', d.name)
            if d.is_dir() and m and (d / 'func').exists():
                sessions.append(m.group(1))
        if not sessions:
            raise FileNotFoundError(f'No sessions found in {sub_dir}')
        return sorted(sessions)

    def get_runs(self, session):
        """Runs discovered from events files for this (subject, session)."""
        raw_dir = self._raw_func_dir(session)
        runs = sorted({
            int(re.search(r'run-(\d+)', f.name).group(1))
            for f in raw_dir.glob(
                f'sub-{self.subject_id}_ses-{session}_task-mapper_run-*_events.tsv')
        })
        if not runs:
            raise FileNotFoundError(f'No events files in {raw_dir}')
        return runs

    # ── JSON sidecars ──────────────────────────────────────────────────────────

    def _bold_json(self, session, run=1):
        """Prefer the fmriprep-copied JSON; fall back to raw."""
        candidates = [
            self._func_dir(session) / (
                f'sub-{self.subject_id}_ses-{session}_task-mapper'
                f'_run-{run}_space-T1w_desc-preproc_bold.json'),
            self._raw_func_dir(session) / (
                f'sub-{self.subject_id}_ses-{session}_task-mapper'
                f'_run-{run}_bold.json'),
        ]
        for p in candidates:
            if p.exists():
                with open(p) as f:
                    return json.load(f)
        raise FileNotFoundError(f'No bold sidecar for {candidates[0]}')

    def get_tr(self, session, run=1):
        meta = self._bold_json(session, run)
        if 'RepetitionTime' in meta:
            return float(meta['RepetitionTime'])
        raise KeyError(f'RepetitionTime missing in sidecar for ses-{session}')

    # ── BOLD ───────────────────────────────────────────────────────────────────

    def get_preprocessed_bold(self, session, runs=None, space='T1w'):
        """List of preprocessed BOLD paths (default T1w space)."""
        if runs is None:
            runs = self.get_runs(session)
        func_dir = self._func_dir(session)
        paths = []
        for run in runs:
            pattern = (f'sub-{self.subject_id}_ses-{session}_task-mapper'
                       f'_run-{run}_space-{space}*desc-preproc_bold.nii.gz')
            matches = sorted(func_dir.glob(pattern))
            if not matches:
                raise FileNotFoundError(
                    f'No BOLD for run-{run} (pattern {pattern}) in {func_dir}')
            paths.append(matches[0])
        return paths

    def get_brain_mask(self, session, run, space='T1w'):
        """Per-run brain mask (NIfTI image)."""
        func_dir = self._func_dir(session)
        pattern = (f'sub-{self.subject_id}_ses-{session}_task-mapper'
                   f'_run-{run}_space-{space}*desc-brain_mask.nii.gz')
        matches = sorted(func_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f'No brain mask for run-{run} in {func_dir}')
        return image.load_img(str(matches[0]))

    def get_conjunct_mask(self, session, runs=None, space='T1w'):
        """Voxels present in every run's brain mask."""
        if runs is None:
            runs = self.get_runs(session)
        masks = [self.get_brain_mask(session, run, space=space) for run in runs]
        # Resample to first run if any mismatch (occurs at 7T with varying matrices).
        ref = masks[0]
        masks = [image.resample_to_img(m, ref, interpolation='nearest')
                 if m.shape != ref.shape else m for m in masks]
        stacked = image.concat_imgs(masks)
        conj = image.math_img('mask.sum(-1) == mask.shape[-1]', mask=stacked)
        # Ensure a 3-D output (math_img can carry a trailing singleton axis).
        arr = np.squeeze(conj.get_fdata()).astype(np.int8)
        return nib.Nifti1Image(arr, conj.affine)

    # ── confounds ──────────────────────────────────────────────────────────────

    DEFAULT_CONFOUNDS = [
        'dvars', 'framewise_displacement',
        'trans_x', 'trans_y', 'trans_z',
        'rot_x', 'rot_y', 'rot_z',
        'a_comp_cor_00', 'a_comp_cor_01', 'a_comp_cor_02',
        'a_comp_cor_03', 'a_comp_cor_04',
        'cosine00', 'cosine01', 'cosine02',
    ]

    def get_fmriprep_confounds(self, session, runs=None, columns=None):
        """Return a dict ``{run: DataFrame}`` with selected confound columns.

        NaNs are back-filled (fmriprep writes NaN for the first framewise
        displacement / dvars entry).  Non-steady-state outlier columns are
        added opportunistically.
        """
        if runs is None:
            runs = self.get_runs(session)
        if columns is None:
            columns = list(self.DEFAULT_CONFOUNDS)
        out = {}
        for run in runs:
            fn = self._func_dir(session) / (
                f'sub-{self.subject_id}_ses-{session}_task-mapper'
                f'_run-{run}_desc-confounds_timeseries.tsv')
            df = pd.read_csv(fn, sep='\t')
            keep = [c for c in columns if c in df.columns]
            # Optional: non-steady-state outliers if present
            keep += [c for c in df.columns
                     if c.startswith('non_steady_state_outlier')]
            out[run] = df[keep].bfill()
        return out

    # ── events & paradigm ──────────────────────────────────────────────────────

    def get_events(self, session, runs=None):
        """Events DataFrame indexed by (run, trial_nr) with onset, n_dots, ..."""
        if runs is None:
            runs = self.get_runs(session)
        frames = []
        for run in runs:
            fn = self._raw_func_dir(session) / (
                f'sub-{self.subject_id}_ses-{session}_task-mapper'
                f'_run-{run}_events.tsv')
            df = pd.read_csv(fn, sep='\t')
            df['run'] = run
            df['trial_nr'] = np.arange(len(df))
            frames.append(df)
        return pd.concat(frames).set_index(['run', 'trial_nr'])

    def get_frametimes(self, session, run=1, n_vols=125):
        """Volume-onset times (s) for a run.  n_vols defaults to 125 (both 3T/7T)."""
        tr = self.get_tr(session, run)
        return np.linspace(0, (n_vols - 1) * tr, n_vols)

    def get_mapper_paradigm(self, session, run=None, natural_space=False,
                            n_vols=125):
        """Return a DataFrame of log-numerosity per timepoint, indexed by run.

        Resamples the event stream (trial_type == 'stimulation') onto the TR
        grid using ``nearest`` interpolation. Returned column ``n_dots`` is
        ``log(n_dots)`` by default; pass ``natural_space=True`` for raw counts.
        The index matches what ``get_bold_timeseries`` produces.
        """
        runs = [run] if run is not None else self.get_runs(session)
        tr = self.get_tr(session)
        dfs = []
        for r in runs:
            events = self.get_events(session, runs=[r]).reset_index()
            stim = events[events['trial_type'] == 'stimulation'].copy()
            stim['onset_mid'] = stim['onset'] + stim['duration'] / 2.0
            stim.index = pd.to_timedelta(stim['onset_mid'], unit='s')

            # Anchor 0 and last-volume time so resample produces full-length
            # output even when stimuli stop early.
            anchor = pd.DataFrame(
                {'n_dots': [0, 0]},
                index=pd.to_timedelta([0.0, (n_vols - 1) * tr], unit='s'))
            merged = pd.concat((anchor, stim[['n_dots']]))

            paradigm = (merged['n_dots']
                        .resample(f'{tr}s')
                        .nearest()
                        .to_frame('n_dots')
                        .astype(np.float32))
            if not natural_space:
                paradigm['n_dots'] = (np.log(paradigm['n_dots'])
                                      .replace(-np.inf, 0))
            paradigm.index = pd.Index(
                self.get_frametimes(session, run=r, n_vols=n_vols),
                name='time')
            dfs.append(paradigm)
        return pd.concat(dfs, keys=runs, names=['run'])

    def get_mapper_response_hrf(self, session, runs=None, n_vols=125):
        """Boxcar regressor for behavioural responses (optional confound).

        Responses exist only where ``rt`` is non-null.  Returns a DataFrame
        with a single column ``response`` indexed by (run, frame_time).
        """
        if runs is None:
            runs = self.get_runs(session)
        events = self.get_events(session, runs=runs).reset_index()
        responses = events[events['rt'].notna()].copy()
        if responses.empty:
            # No responses recorded in mapper (passive viewing); return zeros.
            frames = []
            for r in runs:
                ft = self.get_frametimes(session, run=r, n_vols=n_vols)
                frames.append(pd.DataFrame(
                    {'response': np.zeros(len(ft), dtype=np.float32)},
                    index=ft))
            return pd.concat(frames, keys=runs, names=['run'])

        responses['onset'] = responses['onset'] + responses['rt']
        responses['duration'] = 0.0
        responses['trial_type'] = 'response'
        out = []
        for r in runs:
            ft = self.get_frametimes(session, run=r, n_vols=n_vols)
            sub = responses[responses['run'] == r][['onset', 'duration',
                                                    'trial_type']]
            if sub.empty:
                dm = pd.DataFrame({'response': np.zeros(len(ft))}, index=ft)
            else:
                dm = make_first_level_design_matrix(
                    ft, sub, drift_model=None, drift_order=0)
                dm = dm[['response']] if 'response' in dm.columns else \
                     pd.DataFrame({'response': np.zeros(len(ft))}, index=ft)
            out.append(dm)
        return pd.concat(out, keys=runs, names=['run'])

    # ── output paths ───────────────────────────────────────────────────────────

    def get_derivative_dir(self, session, base, modality='func'):
        d = (self.bids_folder / 'derivatives' / base
             / f'sub-{self.subject_id}' / f'ses-{session}' / modality)
        d.mkdir(parents=True, exist_ok=True)
        return d


def get_all_subject_ids(bids_folder=BIDS_FOLDER, fmriprep_deriv='fmriprep'):
    """List subjects present in fmriprep derivatives."""
    root = Path(bids_folder) / 'derivatives' / fmriprep_deriv
    return sorted([p.name.split('sub-')[1]
                   for p in root.iterdir()
                   if p.is_dir() and p.name.startswith('sub-')])
