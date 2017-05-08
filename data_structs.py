import numpy as np
import random
import re
import pickle
from rdkit import Chem
import sys

class Vocabulary(object):
    def __init__(self, max_length=140):
        self.special_tokens = ['EOS', 'GO']
        self.additional_chars = set()
        self.chars = self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.iteritems()}
        self.max_length = max_length
        
    def encode(self, char_list, seq_len=None):
        if seq_len is None: seq_len = self.max_length
        smiles_matrix = np.zeros([seq_len, self.vocab_size], dtype=int)
        for i,char in enumerate(char_list):
            smiles_matrix[i, self.vocab[char]] = 1
        return smiles_matrix
        
    def decode(self, matrix):
        chars = []
        for i in range(np.shape(matrix)[0]):
            if np.sum(matrix[i, :])<0.5: break
            idx = np.argmax(matrix[i, :])
            chars.append(self.reversed_vocab[idx])
            if np.argmax(matrix[i, :])==self.vocab['EOS']: break
        smiles = "".join(chars).replace('EOS', '')
        smiles = smiles.replace("L", "Cl").replace("R", "Br")
        return smiles            
        
    def add_characters(self, chars):  
        for char in chars:
            self.additional_chars.add(char)
        char_list = list(self.additional_chars)    
        char_list.sort()
        self.chars = char_list + self.special_tokens
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reversed_vocab = {v: k for k, v in self.vocab.iteritems()}
                       
class MolData(object):
    def __init__(self, mols, voc):
        self.mols = mols
        self.voc = voc
        self.size = np.shape(mols)[0]
        
    def sample(self, n):
        mols_selection = [self.mols[i] for i in random.sample(range(len(self.mols)), n)]
        mol_list = []
        lengths = []
        for mol in mols_selection:
            regex = '(\[[^\[\]]{1,6}\])'
            smiles = replace_halogen(mol)
            char_list = re.split(regex, smiles)
            smiles = []
            for char in char_list:
                if char.startswith('['):
                    smiles.append(char)
                else:
                    chars = [unit for unit in char]
                    [smiles.append(unit) for unit in chars]      
            smiles.append('EOS')
            mol_list.append(smiles) 
            lengths.append(len(smiles))
        max_len = max(lengths)
        mols = [self.voc.encode(mol, seq_len=max_len) for mol in mol_list]
        mols = np.array(mols)
        lengths = np.array(lengths, dtype=np.int32)
        return mols, lengths

def replace_halogen(string):

    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    
    return string

def tokenize(smiles):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        tokenized = []
        for char in char_list:
            if char.startswith('['):
                tokenized.append(char)
            else:
                chars = [unit for unit in char]
                [tokenized.append(unit) for unit in chars]      
        tokenized.append('EOS')
        return tokenized

def canonicalize_smiles_from_file(fname):
    smiles_list = []
    with open(fname, 'r') as f:
        for line in f:
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                smiles_list.append(Chem.MolToSmiles(mol))
    return smiles_list

def construct_moldata_from_smiles(smiles_list, voc, fname):
    with open(fname + '_MolData', 'wb') as f:
        pickle.dump(MolData(smiles_list, voc), f)

def construct_vocabulary(smiles_list, fname):
    voc = Vocabulary()
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]      

    voc.add_characters(add_chars)
    print voc.vocab_size     
    print voc.chars
    
    with open(fname + '_Voc', 'wb') as f:
        pickle.dump(voc, f)
    return voc

if __name__ == "__main__":
    smiles_file = sys.argv[1]
    print "Reading smiles..."
    smiles_list = canonicalize_smiles_from_file(smiles_file)
    print "Constructing vocabulary..."
    voc = construct_vocabulary(smiles_list, smiles_file)
    print "Constructing MolData instance..."
    construct_moldata_from_smiles(smiles_list, voc, smiles_file)
    
