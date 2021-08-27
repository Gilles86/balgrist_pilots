import os.path as op
import argparse
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces import ants, afni, fsl
from nipype.interfaces import io as nio
import shutil
from niworkflows.interfaces.bids import DerivativesDataSink


def main(subject, session='balgrist7t', acquisition=None, sourcedata='/Users/gdehol/Science/balgrist/data/ds-balgrist'):
    wf = pe.Workflow(name='fix_mp2rage', base_dir='/tmp')

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['inv2', 'uni']), name='inputnode')
    if acquisition is None:
        inputnode.inputs.inv2 = op.join(
            sourcedata, f'sub-{subject}', f'ses-{session}', 'anat', f'sub-{subject}_ses-{session}_inv-2_MP2RAGE.nii')
        inputnode.inputs.uni = op.join(
            sourcedata, f'sub-{subject}', f'ses-{session}', 'anat', f'sub-{subject}_ses-{session}_T1UNI.nii')
    else:
        inputnode.inputs.inv2 = op.join(
            sourcedata, f'sub-{subject}', f'ses-{session}', 'anat', f'sub-{subject}_ses-{session}_acq-{acquisition}_inv-2_MP2RAGE.nii')
        inputnode.inputs.uni = op.join(
            sourcedata, f'sub-{subject}', f'ses-{session}', 'anat', f'sub-{subject}_ses-{session}_acq-{acquisition}_T1UNI.nii')

    biasfield_correction = pe.Node(
        ants.N4BiasFieldCorrection(num_threads=6), name='n4_bias')

    wf.connect(inputnode, 'inv2', biasfield_correction, 'input_image')

    automask = pe.Node(afni.Automask(
        num_threads=4, outputtype='NIFTI_GZ', clfrac=0.5), name='automasker')
    wf.connect(biasfield_correction, 'output_image', automask, 'in_file')

    bet = pe.Node(fsl.BET(mask=True), name='bet')
    wf.connect(biasfield_correction, 'output_image', bet, 'in_file')

    combine_masks = pe.Node(fsl.BinaryMaths(
        operation='add'), name='combine_masks')
    wf.connect(bet, 'mask_file', combine_masks, 'in_file')
    wf.connect(automask, 'out_file', combine_masks, 'operand_file')

    apply_mask = pe.Node(fsl.ApplyMask(), name='apply_mask')

    wf.connect(inputnode, 'uni', apply_mask, 'in_file')
    wf.connect(combine_masks, 'out_file', apply_mask, 'mask_file')

    ds = pe.Node(DerivativesDataSink(suffix='T1w',
                                     base_directory=op.join(sourcedata, 'derivatives')), name='datasink')
    wf.connect(inputnode, 'uni', ds, 'source_file')
    wf.connect(apply_mask, 'out_file', ds, 'in_file')

    wf.run()


    if acquisition is None:
        shutil.copy(op.join(sourcedata, 'derivatives', 'niworkflows', f'sub-{subject}', f'ses-{session}', 'anat',
                            f'sub-{subject}_ses-{session}_T1w.nii'),
                    op.join(sourcedata, f'sub-{subject}', f'ses-{session}', 'anat',
                            f'sub-{subject}_ses-{session}_T1w.nii'))
    else:
        shutil.copy(op.join(sourcedata, 'derivatives', 'niworkflows', f'sub-{subject}', f'ses-{session}', 'anat',
                            f'sub-{subject}_ses-{session}_acq-{acquisition}_T1w.nii'),
                    op.join(sourcedata, f'sub-{subject}', f'ses-{session}', 'anat',
                            f'sub-{subject}_ses-{session}_acq-{acquisition}_T1w.nii'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', nargs='?', default='balgrist7t')
    parser.add_argument('acquisition', nargs='?', default=None)
    parser.add_argument(
        '--sourcedata', default='/Users/gdehol/Science/balgrist/data/ds-balgrist')

    args = parser.parse_args()

    main(args.subject, args.session, args.acquisition, args.sourcedata)
