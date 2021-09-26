import os.path as op
from balgrist.utils import get_surf_file, get_runs, run_main
import nipype.pipeline.engine as pe
from nipype.interfaces import freesurfer
from nipype.interfaces import utility as niu
from niworkflows.interfaces.bids import DerivativesDataSink


def main(subject, session, bids_folder, space='fsnative', n_procs=12):

    wf = pe.Workflow(name=f'smooth_{subject}_{session}_{space}',
                     base_dir='/tmp')

    runs = get_runs(subject, session)
    fns_l = [get_surf_file(subject, session, run, bids_folder, 'lh')
             for run in runs]
    fns_r = [get_surf_file(subject, session, run, bids_folder, 'rh')
             for run in runs]
    fns = fns_l + fns_r

    hemis = ['lh'] * len(runs) + ['rh'] * len(runs)

    input_node = pe.Node(niu.IdentityInterface(fields=['freesurfer_subject',
                                                       'surface_files', 'hemis']),
                         name='input_node')
    input_node.inputs.freesurfer_subject = f'sub-{subject}'
    input_node.inputs.surface_files = fns
    input_node.inputs.hemis = hemis

    freesurfer_dir = op.join(bids_folder, 'derivatives', 'freesurfer')
    smoother = pe.MapNode(freesurfer.SurfaceSmooth(
        fwhm=5, subjects_dir=freesurfer_dir), iterfield=['in_file', 'hemi'], name='smoother')

    wf.connect(input_node, 'freesurfer_subject', smoother, 'subject_id')
    wf.connect(input_node, 'surface_files', smoother, 'in_file')
    wf.connect(input_node, 'hemis', smoother, 'hemi')

    def get_suffix(in_files):
        print(in_files)
        import re
        reg = re.compile(
            '.*/(?P<subject>sub-[0-9]+)_.*_hemi-(?P<hemi>L|R)_bold\.func\.gii')
        hemis = [reg.match(fn).group(2) for fn in in_files]

        return ['_hemi-{}'.format(hemi) for hemi in hemis]

    ds = pe.MapNode(DerivativesDataSink(out_path_base='smoothed',
                                        dismiss_entities=[
                                            'suffix', 'extension'],
                                        extension=".func.gii",
                                        suffix='bold'),
                    iterfield=['source_file', 'in_file'], name='datasink')
    ds.inputs.base_directory = op.join(bids_folder, 'derivatives')
    ds.inputs.desc = 'smoothed'

    wf.connect(input_node, 'surface_files', ds, 'source_file')
    wf.connect(smoother, 'out_file', ds, 'in_file')

    wf.run(plugin='MultiProc', plugin_args={'n_procs': n_procs})

if __name__ == '__main__':
    run_main(main)
