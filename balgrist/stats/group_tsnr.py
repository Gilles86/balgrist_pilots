import argparse
import glob
import os
import os.path as op
from nipype.algorithms.confounds import TSNR
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces import ants
from niworkflows.interfaces.bids import DerivativesDataSink
from nilearn import image


def main(session, root_folder='/data/ds-balgrist'):

    tsnrs = glob.glob(op.join(root_folder, 'derivatives', 'tsnr',
                             f'sub-*', f'ses-{session}', 'func', f'sub-*_ses-{session}_task-mapper_run-*_space-MNI152NLin6Asym_desc-tsnr_bold.nii'))

    tsnrs = image.concat_imgs([image.smooth_img(tsnrs, 6)])

    mean_tsnr = image.mean_img(tsnrs)

    target_dir = op.join(root_folder, 'derivatives', 'tsnr', 'group')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    mean_tsnr.to_filename(op.join(target_dir, f'group_acq-{session}_meantsnr.nii.gz'))

    print(tsnrs)
    # t1w_to_mni = op.join(root_folder, 'derivatives', 'fmriprep',
                         # f'sub-{subject}', 'anat', f'sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    # wf = pe.Workflow(base_dir='/tmp/workflow_folders',
                     # name=f'tsnr_balgrist_sub-{subject}_ses-{session}')

    # inputnode = pe.Node(niu.IdentityInterface(
        # fields=['bold']), name='inputnode')
    # inputnode.inputs.bold = bold

    # tsnr = pe.MapNode(TSNR(), iterfield=['in_file'], name='tsnr')

    # wf.connect([(inputnode, tsnr, [('bold', 'in_file')])])

    # applier = pe.MapNode(ants.ApplyTransforms(),
                         # iterfield=['input_image'],
                         # name='applier')

    # wf.connect([(tsnr, applier, [('tsnr_file', 'input_image')])])

    # applier.inputs.reference_image = mni_brain
    # applier.inputs.transforms = t1w_to_mni

    # ds = pe.MapNode(DerivativesDataSink(base_directory=op.join(root_folder, 'derivatives')), iterfield=['in_file', 'source_file'],
                    # out_path_base='tsnr',
                    # name='datasink')
    # ds.inputs.space = 'MNI152NLin6Asym'
    # ds.inputs.desc = 'tsnr'
    # wf.connect([(inputnode, ds, [('bold', 'source_file')]),
                # (applier, ds, [('output_image', 'in_file')])])

    # args_dict = {'n_procs': 6, 'memory_gb': 10}
    # wf.run(plugin='MultiProc', plugin_args=args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('session', nargs='?', default='balgrist7t')
    parser.add_argument(
        '--root_folder', default='/data/ds-balgrist')

    args = parser.parse_args()

    main(args.session, args.root_folder)
