# REINVENT
Molecular De Novo design using Recurrent Neural Networks and Reinforcement Learning
=============

Source code for the method described in:

[Molecular De Novo Design through Deep Reinforcement Learning](https://arxiv.org/abs/1704.07555)

## Requirements

This package requires:
* Python 2.7
* Tensorflow 1.0 
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Scikit-Learn (for QSAR scoring function)

## Usage

To use with a custom SMILES file:
* Create MolData and Vocabulary instances using "python data_structs.py [location of SMILES file]"
* Run the model.py pretrain_rnn method pointing to the MolData and Vocabulary files created
* The Prior will be saved in the folder specified.
* Run the model.py train_agent method to train agent, and the model.py sample method (after, or both before and after for comparison) to generate samples

The datasets and pretrained Priors used in the paper can be found under "Releases"

## To Do

* Make a package
* Add argparser to facilitate usage

Any further comments/suggestions/criticism/questions are welcome! 

For contact outside of GitHub: m.olivecrona@gmail.com
