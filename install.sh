conda create -n adm python=3.8
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install blobfile
pip install tqdm
pip install mpi4py
pip install numpy
pip install Pillow
pip install opencv-python
pip install -e .
pip install tensorflow-gpu==2.2.0
pip install scipy
pip install requests
pip install tqdm
conda install protobub==3.20