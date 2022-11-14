# Semi Supervised Learning - Deep Co-Training

Application of Deep Co-Training for audio tagging on multiple audio dataset.

If you meet problems to run experiments, you can contact me at `labbeti.pub@gmail.com`.

# Requirements
```bash
git clone https://github.com/Labbeti/semi-supervised

cd semi-supervised
conda env create -f environment.yaml
conda activate ssl

pip install -e .

```
<!--
## Manually
```bash
conda create -n dct python=3 pip
conda activate dct

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install numpy
conda install pandas
conda install scikit-learn
conda install scikit-image
conda install tqdm
conda install h5py
conda install pillow
conda install librosa -c conda-forge

pip install hydra-core
pip install advertorch
pip install torchsummary
pip install tensorboard

cd Deep-Co-Training
pip install -e .
```
## Fix missing package
- It is very likely that the `ubs8k` will be missing. It a code to manage the UrbanSound8K dataset I wrote almost two years ago before I start using `torchaudio`.
- `pytorch_metrics` is a basic package I wrote to handle many of the metrics I used during my experiments.
- `augmentation_utils` is a package I wrote to test and apply many different augmentation during my experiments.
```bash
pip install --upgrade git+https://github.com/leocances/UrbanSound8K.git@
pip install --upgrade git+https://github.com/leocances/pytorch_metrics.git@v2
pip install --upgrade git+https://github.com/leocances/augmentation_utils.git
```
I am planning on release a much cleaner implementation that follow the torchaudio rules.
-->

## Reproduce full supervised learning for UrbanSound8k dataset
```bash
conda activate dct
python standalone/full_supervised/full_supervised.py --from-config DCT/util/config/ubs8k/100_supervised.yml
```

# Train the systems
The directory `standalone/` contains the scripts to execute the different semi-supervised methods and the usual supervised approach. Each approach has it own working directory which contains the python scripts.

The handling of running arguments is done using [hydra](hydra.cc) and the configuration files can be found in the directory `config/`. There is one configuration file for each dataset and methods.

## Train for speechcommand
```bash
conda activate ssl
cd semi-supervised/standalone/supervised

python supervised.py -cn ../../config/supervised/speechcommand.yaml
```

You can override the parameters from the configuration file by doing the following, allowing you to change the model to use, training parameters or some augmentation parameters. please read the configuration file for more detail.

## Train for speechcommand using ResNet50
```bash
python supervised.py -cn ../../config/supervised/speechcommand.yaml model.model=resnet50
```

# Basic API
I am currently trying to make a main script from which it will possible to train and use the models easily.
This documentation is more for my personnal use and is not exaustif yet. It is better to use directly the proper training script with the conrresponding configuration file.

## commands
- **train**
    - **--model**: The model architecture to use. See [Available models](#available-models)
    - **--dataset**: The dataset you want to use for the training. See [Install datasets](#install-datasets)
    - **--method**: The learning method you want to use for the training. See [Available methods](#available-methods)
    - \[hydra-override\]
    - **Exemple**
    ```bash
    python run_ssl train --dataset speechcommand --model wideresnet28_2 --method mean-teacher [hydra-override-args ...]
    ```

- **inference**
    - **--model**: The model architecture to use. See [Available models](#available-models)
    - **--dataset**: The dataset used to train the model. See [Install datasets](#install-datasets)
    - **--method**: The learning method used for the training. See [Available methods](#available-methods)
    - **-w | --weights**: The path to the weight of the model. If left empty will use the latest file available
    - **-f | --file**: The path to the file that will be fed to the model
    - **-o | --output**: The output expected from \{logits | softmax | sigmoid | pred\}
    - **--cuda**: Use the GPU if this flag is added
    - **Exemple**
    ```bash
    python run_ssl inference \
        --dataset ComParE2021-PRS \
        -o softmax \
        -f ../datasets/ComParE2021-PRS/dist/devel_00001.wav \
        -w ../model_save/ComParE2021-PRS/supervised/wideresnet28_2/wideresnet28_2__0.003-lr_1.0-sr_50000-e_32-bs_1234-seed.best
    ```

<!-- - **cross-validation**
    - WIP    


## Available models
WIP

## Install datasets
WIP

## Available methods
WIP -->

## Cite this repository
If you use this code, you can cite the following paper associated :
```
@article{cances_comparison_2022,
	title = {Comparison of semi-supervised deep learning algorithms for audio classification},
	volume = {2022},
	issn = {1687-4722},
	url = {https://doi.org/10.1186/s13636-022-00255-6},
	doi = {10.1186/s13636-022-00255-6},
	abstract = {In this article, we adapted five recent SSL methods to the task of audio classification. The first two methods, namely Deep Co-Training (DCT) and Mean Teacher (MT), involve two collaborative neural networks. The three other algorithms, called MixMatch (MM), ReMixMatch (RMM), and FixMatch (FM), are single-model methods that rely primarily on data augmentation strategies. Using the Wide-ResNet-28-2 architecture in all our experiments, 10\% of labeled data and the remaining 90\% as unlabeled data for training, we first compare the error rates of the five methods on three standard benchmark audio datasets: Environmental Sound Classification (ESC-10), UrbanSound8K (UBS8K), and Google Speech Commands (GSC). In all but one cases, MM, RMM, and FM outperformed MT and DCT significantly, MM and RMM being the best methods in most experiments. On UBS8K and GSC, MM achieved 18.02\% and 3.25\% error rate (ER), respectively, outperforming models trained with 100\% of the available labeled data, which reached 23.29\% and 4.94\%, respectively. RMM achieved the best results on ESC-10 (12.00\% ER), followed by FM which reached 13.33\%. Second, we explored adding the mixup augmentation, used in MM and RMM, to DCT, MT, and FM. In almost all cases, mixup brought consistent gains. For instance, on GSC, FM reached 4.44\% and 3.31\% ER without and with mixup. Our PyTorch code will be made available upon paper acceptance at https://github.com/Labbeti/SSLH.},
	number = {1},
	journal = {EURASIP Journal on Audio, Speech, and Music Processing},
	author = {Cances, Léo and Labbé, Etienne and Pellegrini, Thomas},
	month = sep,
	year = {2022},
	pages = {23},
}
```

## Contact
- Etienne Labbé "Labbeti" (maintainer) : labbeti.pub@gmail.com
