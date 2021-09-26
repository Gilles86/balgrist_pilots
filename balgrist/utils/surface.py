import os.path as op
from nipype.interfaces.freesurfer import SurfaceTransform
import re

def transform_data(source_fn, source_subject, sourcedata, target_fn=None, target_subject='fsaverage6'):

    reg = re.compile('.*_hemi-(?P<hemi>L|R|lh|rh|{hemi}).*')

    hemi = reg.match(source_fn).group(1)

    if hemi == '{hemi}':
        fns = []
        for hemi in ['L', 'R']:
            fns.append(transform_data(source_fn.format(hemi=hemi),
                source_subject,
                sourcedata,
                target_fn.format(hemi=hemi) if target_fn else None,
                target_subject=target_subject))
        return fns
    
    if hemi == 'L':
        fs_hemi = 'lh'
    elif hemi == 'R':
        fs_hemi = 'rh'
    else:
        fs_hemi = hemi

    freesurfer_dir = op.join(sourcedata, 'derivatives', 'freesurfer')

    
    reg = re.compile('.*_(space-[a-zA-Z0-9]+)_.*')

    if target_fn is None:
        target_fn = source_fn.replace(reg.match(source_fn).group(1), f'space-{target_subject}')

    print(target_fn)

    transformer = SurfaceTransform(source_subject=source_subject, target_subject=target_subject, subjects_dir=freesurfer_dir,
            source_file=source_fn, out_file=target_fn,
            hemi=fs_hemi)

    transformer.run()
    return target_fn

if __name__ == '__main__':
    transform_data('/data/derivatives/fmriprep/sub-02/ses-7t1/func/sub-02_ses-7t1_task-mapper_run-4_space-fsnative_hemi-{hemi}_bold.func.gii',
            'sub-02', 
            '/data')
            # target_fn='/tmp/test.gii')
