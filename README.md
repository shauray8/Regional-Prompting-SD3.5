# Regional-Prompting-SD3.5
SD3.5 Fork for Training-free Regional Prompting for Diffusion Transformers 🔥

### Installation
```python
# install diffusers locally
git clone https://github.com/huggingface/diffusers.git
cd diffusers

pip install -e ".[torch]"
cd ..

# install other dependencies
pip install -U transformers sentencepiece protobuf PEFT

# clone this repo
git clone https://github.com/shauray8/Regional-Prompting-SD3.5.git

# replace file in diffusers
cd Regional-Prompting-SD3.5
cp transformer_sd3.py ../diffusers/src/diffusers/models/transformers/transformer_sd3.py
```

### Quick start

```python
python ./infer.py
```
