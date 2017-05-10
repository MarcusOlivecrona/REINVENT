#!/usr/bin/env python
from __future__ import division
import numpy as np
from sklearn import svm
from rdkit import Chem
from rdkit import rdBase
from rdkit import DataStructs
from rdkit.Chem import AllChem
import pickle
rdBase.DisableLog('rdApp.error')

"""Scoring functions should take as input an array of SMILES and return an array of floats between -1 and 1"""

def no_sulphur(smiles):
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

def tanimoto(smiles):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    valid = [1 if mol!=None else 0 for mol in mols]
    valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
    valid_mols = [mols[idx] for idx in valid_idxs]

    fps = [AllChem.GetMorganFingerprint(mol, 2, useCounts=False, useFeatures=True) for mol in valid_mols]
    query_mol = Chem.MolFromSmiles("Cc1ccc(cc1)c2cc(nn2c3ccc(cc3)S(=O)(=O)N)C(F)(F)F")
    query_fp = AllChem.GetMorganFingerprint(query_mol, 2, useCounts=False, useFeatures=True)

    tversky_similarity = np.array([DataStructs.TverskySimilarity(fragment_fp, fp, 1, 1) for fp in fps])
    tversky_similarity = np.minimum(tversky_similarity, k)
    tversky_similarity = [-1 + 2 * x / k for x in tversky_similarity]
    score = np.full(len(smiles), -1, dtype=np.float32)

    for idx, value in zip(valid_idxs, tversky_similarity):
        score[idx] =  value
    return np.array(score, dtype=np.float32)

def restore_activity_model():
    with open("data/clf.pkl", "r") as f:
        clf = pickle.load(f)
    return clf

def fingerprints_from_mols(mols):
        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 3) for mol in mols] 
        np_fps = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            np_fps.append(arr)
        return np_fps

def activity_model(clf):
    def classifier(smiles):
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol!=None else 0 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean==1]
        valid_mols = [mols[idx] for idx in valid_idxs]

        fps = fingerprints_from_mols(valid_mols)
        activity_score = clf.predict_proba(fps)[:, 1]    
        activity_score = -1 + 2 * activity_score

        score = np.full(len(smiles), -1, dtype=np.float32)

        for idx, value in zip(valid_idxs, activity_score):
            score[idx] =  value
        return np.array(score, dtype=np.float32)
    return classifier

