
# REINVENT
## Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning

Searching chemical space as described in:

[Molecular De Novo Design through Deep Reinforcement Learning](https://arxiv.org/abs/1704.07555)

![Video demonstrating an Agent trained to generate analogues to Celecoxib](https://github.com/MarcusOlivecrona/REINVENT/blob/master/images/celecoxib_analogues.gif "Training an Agent to generate analogues of Celecoxib")


## Notes
The current version is a PyTorch implementation that differs in several ways from the original implementation described in the paper. This version works better in most situations and is better documented, but for the purpose of reproducing results from the paper refer to [Release v1.0.1](https://github.com/MarcusOlivecrona/REINVENT/releases/tag/v1.0.1)

Differences from implmentation in the paper:
* Written in PyTorch/Python3.6 rather than TF/Python2.7
* SMILES are encoded with token index rather than as a onehot of the index. An embedding matrix is then used to transform the token index to a feature vector.
* Scores are in the range (0,1).
* A regularizer that penalizes high values of total episodic likelihood is included.
* Sequences are only considered once, ie if the same sequence is generated twice in a batch only the first instance contributes to the loss.
* These changes makes the algorithm more robust towards local minima, means much higher values of sigma can be used if needed.

## Requirements

This package requires:
* Python 3.6
* PyTorch 0.1.12 
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Scikit-Learn (for QSAR scoring function)
* tqdm (for training Prior)
* pexpect

## Usage

To train a Prior starting with a SMILES file called mols.smi:

* First filter the SMILES and construct a vocabulary from the remaining sequences. `./data_structs.py mols.smi`   - Will generate data/mols_filtered.smi and data/Voc. A filtered file containing around 1.1 million SMILES and the corresponding Voc is contained in "data".

* Then use `./train_prior.py` to train the Prior. A pretrained Prior is included.

To train an Agent using our Prior, use the main.py script. For example:

* `./main.py --scoring-function activity_model --num-steps 1000`

Training can be visualized using the Vizard bokeh app. The vizard_logger.py is used to log information (by default to data/logs) such as structures generated, average score, and network weights.

* `cd Vizard`
* `./run.sh ../data/logs`
* Open the browser at http://localhost:5006/Vizard


