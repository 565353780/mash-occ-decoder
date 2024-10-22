cd ..
git clone https://github.com/kacperkan/light-field-distance

if [ "$(uname)" == "Darwin" ]; then
  pip install open3d==0.15.1
elif [ "$(uname)" = "Linux" ]; then
  pip install -U open3d
fi

pip install -U trimesh tensorboard Cython pykdtree timm einops
pip install -U causal-conv1d
pip install -U mamba-ssm

cd light-field-distance
python setup.py install

cd ../aro-net

pip install -U torch torchvision torchaudio
# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

python setup.py build_ext --inplace

mkdir ssl
cd ssl
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
