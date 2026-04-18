# balgrist_pilots

Numerosity mapper pilots acquired across four scanners:
`ses-balgrist3t`, `ses-balgrist7t`, `ses-ibt7t`, `ses-sns3t`.

## Data locations

| Context | Path |
|---------|------|
| Network drive (SMB) | `/Volumes/g_econ_department$/projects/2021/dehollander_nagy_balgristpilots/data/ds-balgrist` |
| Cluster | `/shares/zne.uzh/gdehol/ds-balgrist` |

## Phase A — numerosity PRF R² per session (voxel space, T1w)

Script: `balgrist/encoding_model/fit_mapper.py`
SLURM: `balgrist/encoding_model/slurm_jobs/fit_mapper.sh`

Fits `GaussianPRFWithHRF` (braincoder) on log-numerosity to cleaned, PSC'd
BOLD timeseries (T1w-space), one (subject, session) at a time. Writes
`derivatives/encoding_model/sub-*/ses-*/func/sub-*_ses-*_desc-{r2,mu,sd,amplitude,baseline}_space-T1w_pars.nii.gz`.

```bash
# single (subject, session) on the cluster
sbatch --export=PARTICIPANT_LABEL=01,SESSION=balgrist3t fit_mapper.sh

# full grid: 5 subjects × 4 sessions
for sub in 01 02 03 05 001; do
  for ses in balgrist3t balgrist7t ibt7t sns3t; do
    sbatch --export=PARTICIPANT_LABEL=$sub,SESSION=$ses fit_mapper.sh
  done
done
```

## Phase B — decodability (todo)

CV variant of the fit + decoding of log(n) on held-out runs.
