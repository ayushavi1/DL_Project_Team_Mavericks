# mavNet

<p align="center">
  <img src="./assets/mavNet1.png">
    <br/><em>For Training</em>
  <img src="./assets/mavNet2.png">
    <br/><em>For Retrieval</em>
</p>

## Setup
First install Miniconda for your operating system. 
### Conda
After installing Miniconda, create a virtual environment.
```bash
conda create -n mavnet
```
Install Faiss GPU using conda
```bash
conda install -c conda-forge faiss-gpu
```
Install Pytorch for CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Install remaining dependencies
```bash
pip install wandb tqdm tensorboardX scipy scikit-learn h5py termcolor
```

### Download
Download the dataset files from the following link and put all the files in the ./data/descData/netvlad-pytorch/ folder.
https://drive.google.com/drive/folders/1wraXOQl0Oaj2Lwnx3J3HwxbooJeAbkvQ?usp=sharing

For downloading the pretrained models use this link and put the downloaded folder in ./data/runs
https://drive.google.com/drive/folders/1Qv_f_Iwt6HwQc-Bmw3ofu6SpYH0crgY2

## Run

### Train
To train mavNet on the Oxford dataset with sequence matching:
```python
python main.py --mode train --seqL 5 --pooling --dataset oxford-v1.0 --loss_trip_method meanOfPairs --neg_trip_method meanOfPairs --expName ox10_MoP_negMoP
```
For the Nordland dataset:
```python
python main.py --mode train --seqL 5 --pooling --dataset nordland-sw --loss_trip_method meanOfPairs --neg_trip_method meanOfPairs --expName nord-sw_MoP_negMoP
```

To train without sequence matching (Basically seqNet):
```python
python main.py --mode train --seqL 5 --pooling --dataset oxford-v1.0 --loss_trip_method centerOnly --neg_trip_method centerOnly --expName ox10_CO_negCO
```

### Test
```python
python main.py --mode test --seqL 5 --pooling --dataset oxford-v1.0 --split test --resume ./data/runs/<name_of_the_model_file>
```

## Acknowledgement
The code in this repository is based on [oravus/seqNet](https://github.com/oravus/seqNet) and [Nanne/pytorch-NetVlad](https://github.com/Nanne/pytorch-NetVlad).

