import argparse
import glob
import os.path as op
from nipype.algorithms.confounds import TSNR
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces import ants
from niworkflows.interfaces.bids import DerivativesDataSink
from nipype.interfaces.fsl import TemporalFilter, MeanImage, BinaryMaths


def main(subject, session, root_folder='/data/ds-balgrist',
         mni_brain='/Users/gdehol/templateflow/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-01_T1w.nii.gz'):

    bold = glob.glob(op.join(root_folder, 'derivatives', 'fmriprep',
                             f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-mapper_run-*_space-T1w_desc-preproc_bold.nii.gz'))

    t1w_to_mni = op.join(root_folder, 'derivatives', 'fmriprep',
                         f'sub-{subject}', 'anat', f'sub-{subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')

    wf = pe.Workflow(base_dir='/tmp/workflow_folders',
                     name=f'tsnr_balgrist_sub-{subject}_ses-{session}')

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['bold']), name='inputnode')
    inputnode.inputs.bold = bold

    HP_freq = 0.008
    TR = 2.3

    mean_node = pe.MapNode(MeanImage(), iterfield=['in_file'], name='mean_node')
    wf.connect([(inputnode, mean_node, [('bold', 'in_file')])])

    filter_node = pe.MapNode(TemporalFilter(highpass_sigma = 1 / (2 * TR * HP_freq)), iterfield=['in_file'], name='filter_node')
    

    wf.connect([(inputnode, filter_node, [('bold', 'in_file')])])

    add_mean = pe.MapNode(BinaryMaths(operation='add'), iterfield=['in_file', 'operand_file'],
            name='add_mean')

    wf.connect([(mean_node, add_mean, [('out_file', 'in_file')])])
    wf.connect([(filter_node, add_mean, [('out_file', 'operand_file')])])

    tsnr = pe.MapNode(TSNR(regress_poly=1), iterfield=['in_file'], name='tsnr')

    wf.connect([(add_mean, tsnr, [('out_file', 'in_file')])])

    applier = pe.MapNode(ants.ApplyTransforms(),
                         iterfield=['input_image'],
                         name='applier')

    wf.connect([(tsnr, applier, [('tsnr_file', 'input_image')])])

    applier.inputs.reference_image = mni_brain
    applier.inputs.transforms = t1w_to_mni

    ds = pe.MapNode(DerivativesDataSink(base_directory=op.join(root_folder, 'derivatives'), out_path_base='tsnr',), iterfield=['in_file', 'source_file'],
                    name='datasink')
    ds.inputs.space = 'MNI152NLin6Asym'
    ds.inputs.desc = 'tsnr'
    wf.connect([(inputnode, ds, [('bold', 'source_file')]),
                (applier, ds, [('output_image', 'in_file')])])

    args_dict = {'n_procs': 6, 'memory_gb': 10}
    wf.run(plugin='MultiProc', plugin_args=args_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', nargs='?', default='balgrist7t')
    parser.add_argument(
        '--root_folder', default='/data/ds-balgrist')

    args = parser.parse_args()

    main(args.subject, args.session, args.root_folder)
