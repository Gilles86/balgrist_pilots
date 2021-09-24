mapping_new_old = {1:18, 2:19, 3:3, 4:32, 5:17}


import json
import argparse
import pandas as pd
import shutil
import glob
import os
import os.path as op
import re
from nilearn import image
from natsort import natsorted


def main(subject, root_folder):


    old_subject = mapping_new_old[subject]

    old_path = op.join(op.dirname(root_folder), 'ds-risk', f'sub-{old_subject:02d}')
    

    for old_session, new_session in zip(['3t1', '7t1'], ['sns3t', 'ibt7t']):
        for modality in ['anat', 'fmap', 'func']:
            new_path = op.join(root_folder, f'sub-{subject:02d}', f'ses-{new_session}', modality)
            if not op.exists(new_path):
                os.makedirs(new_path)


            print(op.join(old_path, f'ses-{old_session}', modality, '*'))
            fns = glob.glob(op.join(old_path, f'ses-{old_session}', modality, '*'))


            for fn in fns:
                _, fn_ = op.split(fn)
                print(fn_)
                new_fn = op.join(new_path, fn_.replace(f'ses-{old_session}', f'ses-{new_session}').replace(f'sub-{old_subject:02d}', f'sub-{subject:02d}'))

                shutil.copy(fn, new_fn)


    for old_session, session in zip(['3t1', '7t1'], ['sns3t', 'ibt7t']):
        fmap_jsons = glob.glob(op.join(root_folder, f'sub-{subject:02d}', f'ses-{session}', 'fmap', '*.json'))
        print(fmap_jsons)

        for json_fn in fmap_jsons:
            with open(json_fn) as handle:
                d = json.load(handle)

                d['IntendedFor'] = f'ses-{session}/' + d['IntendedFor'].replace(f'sub-{old_subject}', f'sub-{subject:02d}').replace(f'ses-{old_session}', f'ses-{session}')

                if d['IntendedFor'].endswith('.nii'):
                    d['IntendedFor'] += '.gz'

            with open(json_fn, 'w') as handle:
                json.dump(d, handle)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=2, type=int)
    parser.add_argument('--root_folder', default='/data/ds-balgrist')

    args = parser.parse_args()

    main(args.subject, args.root_folder)
