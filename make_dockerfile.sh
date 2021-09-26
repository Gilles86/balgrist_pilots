#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
rm -f $DIR/Dockerfile
neurodocker generate docker --base ubuntu --pkg-manager apt --freesurfer version=6.0.0 \
       	--output=$DIR/Dockerfile \
	--freesurfer version=6.0.1 \
	--fsl version=6.0.1 \
	--ants version=2.3.4 \
  --install zsh wget git build-essential \
    --miniconda \
      conda_install="python=3.7 pandas matplotlib scikit-learn seaborn ipython pytables tensorflow=2.5 tensorflow-probability" \
      pip_install="nilearn
		nipype
                  pybids
		  nistats
		  niworkflows
		  tensorflow_probability
		  https://github.com/Gilles86/hedfpy/archive/refactor_gilles.zip
		  pytest
		  svgutils==0.3.1" \
      create_env="neuro" \
      activate=true \
   --run 'wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true' \
   --run 'conda init zsh' \
   --run 'echo "conda activate neuro" >> ~/.zshrc && conda init' \
   --workdir /balgrist \
   --copy braincoder /braincoder \
   --run-bash "source activate neuro && cd /braincoder && python setup.py develop --no-deps" \
   --copy balgrist /balgrist \
   --copy setup.py /setup.py \
   --run-bash "source activate neuro && cd / && python setup.py develop --no-deps" \
   --copy ./nipype.cfg /root/.nipype/nipype.cfg \
