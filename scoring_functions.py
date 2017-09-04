#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit import DataStructs
from sklearn import svm
import pickle
rdBase.DisableLog('rdApp.error')

"""Scoring function should be a class where some tasks that are shared for every call
   can be reallocated to the __init__, and has a __call__ method which takes a list of
   smiles as an argument and returns a matching np.array of floats."""

class no_sulphur(object):
    """Scores structures based on not containing sulphur."""
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol!=None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        has_sulphur = [16 not in [atom.GetAtomicNum() for atom in mol.GetAtoms()] for mol in valid_mols]
        sulphur_score =  [1 if ele else -1 for ele in has_sulphur]
        score = np.full(len(smiles), 0, dtype=np.float32)
        for idx, value in zip(valid_idxs, sulphur_score):
            score[idx] =  value
        return np.array(score, dtype=np.float32)

class tanimoto(object):
    """Scores structures based on Tanimoto similarity to a query structure.
       Scores are only scaled up to k=(0,1), after which no more reward is given."""
    def __init__(self, k=1.0,
                 query_structure="Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F",
                 *args, **kwargs):
        self.k = k
        query_mol = Chem.MolFromSmiles(query_structure)
        self.query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=False, useFeatures=True)

    def __call__(self, smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol!=None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = [AllChem.GetMorganFingerprint(mol, 2, useCounts=False, useFeatures=True) for mol in valid_mols]

        tanimoto = np.array([DataStructs.TanimotoSimilarity(self.query_fp, fp) for fp in fps])
        tanimoto = np.minimum(tanimoto, self.k) / self.k
        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, tanimoto):
            score[idx] =  value
        return np.array(score, dtype=np.float32)

class activity_model(object):
    """Scores based on an ECFP classifier for activity."""
    def __init__(self, clf_path="data/clf.pkl", **kwargs):
        with open(clf_path, "rb") as f:
            self.clf = pickle.load(f)

    def __call__(self, smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol!=None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = activity_model.fingerprints_from_mols(valid_mols)
        activity_score = self.clf.predict_proba(fps)[:, 1]

        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, activity_score):
            score[idx] =  value
        return np.array(score, dtype=np.float32)

    @classmethod
    def fingerprints_from_mols(cls, mols):
        fps = [AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True) for mol in mols]
        size = 2048
        nfp = np.zeros((len(fps), size), np.int32)
        for i, fp in enumerate(fps):
            for idx,v in fp.GetNonzeroElements().items():
                nidx = idx%size
                nfp[i, nidx] += int(v)
        return nfp

def get_scoring_function(name, *args, **kwargs):
    """Function that initializes and returns a scoring function by name"""
    scoring_functions = [no_sulphur, tanimoto, activity_model]
    try:
        scoring_function = [f for f in scoring_functions if f.__name__ == name][0]
        return scoring_function(*args, **kwargs)
    except IndexError:
        print("Scoring function must be one of {}".format([f.__name__ for f in scoring_functions]))
        raise
