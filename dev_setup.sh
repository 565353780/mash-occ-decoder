cd ..
git clone git@github.com:565353780/base-trainer.git
git clone https://github.com/kacperkan/light-field-distance

cd ../base-trainer
./dev_setup.sh

if [ "$(uname)" == "Darwin" ]; then
  pip install open3d==0.15.1
elif [ "$(uname)" = "Linux" ]; then
  pip install -U open3d
fi

pip install -U trimesh tensorboard Cython pykdtree timm einops scikit-image
pip install -U causal-conv1d
pip install -U mamba-ssm

pip install -U PyMCubes

cd ../light-field-distance
python setup.py install

pip install -U torch torchvision torchaudio
