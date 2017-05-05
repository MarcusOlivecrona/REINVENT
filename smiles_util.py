import re
import pickle
from data_structs import MolData, Vocabulary

def mols_from_smiles(fname):
    mols = []
    with open(fname, 'r') as f:
        for line in f:
            smiles = line.split(" ")[0]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append(mol)

def construct_moldata_from_smiles(fname, voc):
    mols = mols_from_smiles(fname)
    with open(fname + '_MolData', 'wb') as f:
        pickle.dump(MolData(mols, voc), f)

def construct_vocabulary(fname):
    voc = Vocabulary()
    mols = mols_from_smiles(fname)
    add_chars = set()

    for i, mol in enumerate(mols):
        regex = '(\[[^\[\]]{1,6}\])'
        smiles = replace_halogen(mol)
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


   
