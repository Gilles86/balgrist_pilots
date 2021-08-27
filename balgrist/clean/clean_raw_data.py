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


def main(subject, session, root_folder):

    print(subject, session, root_folder)

    if type(subject) is int:
        subject = f'{subject:02d}'
    field = session[-2:]

    path = op.join(root_folder, 'sourcedata',
                   f'sub-{subject}', f'ses-{session}')

    new_path = op.join(root_folder, f'sub-{subject}', f'ses-{session}')
    if not op.exists(new_path):
        os.makedirs(new_path)

    if field == '3t':
        mappers = glob.glob(op.join(path, '_fMR_Mapper_*.nii.gz'))
        reg = re.compile(
            '.*/_fMR_Mapper_Run_(?P<run>[0-9]+)_(?P<phenc>L_R|R_L)_(?P<date>[0-9]+)_(?P<acq>[0-9]+)\.nii\.gz')

        func_path = op.join(new_path, 'func')
        if not op.exists(func_path):
            os.makedirs(func_path)

        df = []
        for fn in mappers:
            d = reg.match(fn).groupdict()
            run = int(d['run'])
            shutil.copy(fn, op.join(
                func_path, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_bold.nii.gz'))
            shutil.copy(fn.replace('.nii.gz', '.json'), op.join(
                func_path, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_bold.json'))

            d['fn'] = fn
            d['sidecar_json'] = fn.replace('.nii.gz', '.json')
            df.append(d)

        def make_fmap(source, target):
            fmap_path = op.join(new_path, 'fmap')

            if not op.exists(fmap_path):
                os.makedirs(fmap_path)

            fmap = image.load_img(source.fn)
            if source.name < target.name:
                fmap = image.index_img(fmap, range(-10, 0))
            else:
                fmap = image.index_img(fmap, range(10))

            with open(source.sidecar_json, 'r') as f:
                meta_fmap = json.load(f)
                meta_fmap['IntendedFor'] = f'ses-{session}/func/sub-{subject}_task-mapper_run-{target.name}.nii.gz'

            fmap.to_filename(
                op.join(fmap_path, f'sub-{subject}_ses-{session}_run-{target.name}_epi.nii.gz'))

            with open(op.join(fmap_path, f'sub-{subject}_ses-{session}_run-{target.name}_epi.json'), 'w') as f:
                json.dump(meta_fmap, f)

        df = pd.DataFrame(df).set_index('run').sort_index()
        df.index = df.index.astype(int)
        source1 = df.iloc[1]
        target1 = df.iloc[0]
        make_fmap(source1, target1)

        for ix, row in df.iloc[:-1].iterrows():
            make_fmap(row, df.loc[ix+1])

        t1w = glob.glob(op.join(path, '_t1_mprage_sag_*.nii.gz'))
        assert(len(t1w) == 1), 'Did not find exactly one MPRAGE'
        t1w = t1w[0]

        anat_path = op.join(new_path, 'anat')
        if not op.exists(anat_path):
            os.makedirs(anat_path)

        new_t1w_fn = op.join(anat_path, f'sub-{subject}_acq-mprage_T1w.nii.gz')
        shutil.copy(t1w, new_t1w_fn)
        shutil.copy(t1w.replace('.nii.gz', '.json'),
                    new_t1w_fn.replace('.nii.gz', '.json'))

        mp2rage = natsorted(
            glob.glob(op.join(path, '_t1_mp2rage_sag_p3_iso_*.nii.gz')))
        print(mp2rage)
        assert(len(mp2rage) == 3), "Did not find exactly 3 mp2rages"

        for ix, fn in enumerate([f'sub-{subject}_acq-mp2rage_inv-1_part-mag_MP2RAGE.nii.gz',
            f'sub-{subject}_acq-mp2rage_UNIT1.nii.gz',
            f'sub-{subject}_acq-mp2rage_inv-2_part-mag_MP2RAGE.nii.gz']):

            shutil.copy(mp2rage[ix],
                op.join(anat_path, fn))
            shutil.copy(mp2rage[ix].replace('.nii.gz', '.json'),
                op.join(anat_path, fn).replace('.nii.gz', '.json'))

        gre = natsorted(
            glob.glob(op.join(path, '_t2_swi_tra_p2_1.5mm_*.nii.gz')))
        print(gre)
        assert(len(gre) == 4), "Did not find exactly 4 GREs"

        for ix, fn in enumerate([f'sub-{subject}_part-mag_T2starw.nii.gz',
            f'sub-{subject}_part-phase_T2starw.nii.gz']):
            shutil.copy(gre[ix],
                op.join(anat_path, fn))
            shutil.copy(gre[ix].replace('.nii.gz', '.json'),
                op.join(anat_path, fn).replace('.nii.gz', '.json'))

    elif field == '7t':
        mappers = glob.glob(op.join(path, '_fMRI_Mapper_*.nii.gz'))
        reg = re.compile('.*/_fMRI_Mapper_Run_(?P<run>[0-9]+)_(?P<phenc>L_R|R_L)_(?P<date>[0-9]+)_(?P<acq>[0-9]+)\.nii\.gz')

        func_path = op.join(new_path, 'func')
        if not op.exists(func_path):
            os.makedirs(func_path)

        df = []
        for fn in mappers:
            print(fn)
            d = reg.match(fn).groupdict()
            run = int(d['run'])
            shutil.copy(fn, op.join(func_path, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_bold.nii.gz'))
            shutil.copy(fn.replace('.nii.gz', '.json'), op.join(func_path, f'sub-{subject}_ses-{session}_task-mapper_run-{run}_bold.json'))

            d['fn'] = fn
            d['sidecar_json'] = fn.replace('.nii.gz', '.json')
            df.append(d)

        def make_fmap(source, target):
            fmap_path = op.join(new_path, 'fmap')

            if not op.exists(fmap_path):
                os.makedirs(fmap_path)

            fmap = image.load_img(source.fn)
            if source.name < target.name:
                fmap = image.index_img(fmap, range(-10, 0))
            else:
                fmap = image.index_img(fmap, range(10))

            with open(source.sidecar_json, 'r') as f:
                meta_fmap = json.load(f)
                meta_fmap['IntendedFor'] = f'ses-{session}/func/sub-{subject}_task-mapper_run-{target.name}.nii.gz'

            fmap.to_filename(op.join(fmap_path, f'sub-{subject}_ses-{session}_run-{target.name}_epi.nii.gz'))

            with open(op.join(fmap_path, f'sub-{subject}_ses-{session}_run-{target.name}_epi.json'), 'w') as f:
                json.dump(meta_fmap, f)

        df = pd.DataFrame(df).set_index('run').sort_index()
        df.index = df.index.astype(int)
        source1 = df.iloc[1]
        target1 = df.iloc[0]
        make_fmap(source1, target1)

        for ix, row in df.iloc[:-1].iterrows():
            make_fmap(row, df.loc[ix+1])

        t1w = glob.glob(op.join(path, '_t1_mprage_sag_*.nii.gz'))
        assert(len(t1w) == 1), 'Did not find exactly one MPRAGE'
        t1w = t1w[0]

        anat_path = op.join(new_path, 'anat')
        if not op.exists(anat_path):
            os.makedirs(anat_path)

        new_t1w_fn = op.join(anat_path, f'sub-{subject}_acq-mprage_T1w.nii.gz')
        shutil.copy(t1w, new_t1w_fn)
        shutil.copy(t1w.replace('.nii.gz', '.json'),
                    new_t1w_fn.replace('.nii.gz', '.json'))

        mp2rage = natsorted(
            glob.glob(op.join(path, '_t1_mp2rage_sag_p3_*.nii.gz')))
        print(mp2rage)
        assert(len(mp2rage) == 3), "Did not find exactly 3 mp2rages"

        for ix, fn in enumerate([f'sub-{subject}_acq-mp2rage_inv-1_part-mag_MP2RAGE.nii.gz',
            f'sub-{subject}_acq-mp2rage_inv-2_part-mag_MP2RAGE.nii.gz',
            f'sub-{subject}_acq-mp2rage_UNIT1.nii.gz']):

            shutil.copy(mp2rage[ix],
                op.join(anat_path, fn))
            shutil.copy(mp2rage[ix].replace('.nii.gz', '.json'),
                op.join(anat_path, fn).replace('.nii.gz', '.json'))

        gre = natsorted(glob.glob(op.join(path, '_t2_swi_tra_p3_*_e*.nii.gz')))
        print(gre)
        assert(len(gre) == 12), "Did not find exactly 12 GREs"

        for ix, fn in enumerate([f'sub-{subject}_echo-1_part-mag_T2starw.nii.gz',
                                 f'sub-{subject}_echo-2_part-mag_T2starw.nii.gz',
                                 f'sub-{subject}_echo-3_part-mag_T2starw.nii.gz']):
            shutil.copy(gre[ix],
                op.join(anat_path, fn))
            shutil.copy(gre[ix].replace('.nii.gz', '.json'),
                op.join(anat_path, fn).replace('.nii.gz', '.json'))

        for ix, fn in enumerate([f'sub-{subject}_echo-1_part-phase_T2starw.nii.gz',
                                 f'sub-{subject}_echo-2_part-phase_T2starw.nii.gz',
                                 f'sub-{subject}_echo-3_part-phase_T2starw.nii.gz']):
            shutil.copy(gre[ix+3],
                op.join(anat_path, fn))
            shutil.copy(gre[ix+3].replace('.nii.gz', '.json'),
                op.join(anat_path, fn).replace('.nii.gz', '.json'))

        mtw = glob.glob(op.join(path, '_tfl_multiMTC_*.nii.gz'))

        if len(mtw) == 2:
            shutil.copy(mtw[0].replace('.nii.gz', '.json'), op.join(anat_path, f'sub-{subject}_acq-mean_MTw.nii.gz'))
            mtw = image.mean_img(image.concat_imgs(mtw))

            mtw.to_filename(op.join(anat_path, f'sub-{subject}_acq-mean_MTw.nii.gz'))
        else:
            print('FOUND NO MTw!')




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=2, type=int)
    parser.add_argument('session', default='3tbalgrist')
    parser.add_argument('--root_folder', default='/data/ds-balgrist')

    args = parser.parse_args()

    main(args.subject, args.session, args.root_folder)
