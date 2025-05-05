
conda create -n elgcnet python=3.12 -y
conda activate elgcnet
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements2.txt