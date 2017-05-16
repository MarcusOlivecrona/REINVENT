from matplotlib.offsetbox import OffsetImage, AnnotationBbox, TextArea
import numpy as np
import time
import matplotlib
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

class DynamicPlot(object):
    def __init__(self, n_steps):

        self.fig, (self.mol_ax, self.score_ax) = plt.subplots(2, 1, 
            figsize=(14, 8), gridspec_kw = {'height_ratios':[3.5, 1]})

        self.score_ax.set_xlim(0, n_steps)
        self.score_ax.set_ylim(-1, 1)
        self.score_ax.set_ylabel(r"$\mathrm{Average\ Score}$")
        self.score_ax.set_xlabel(r"$\mathrm{Training\ Step}$")

        self.mol_ax.set_title(r"$\mathrm{Generated\ Molecules}$", y=0.97)
        self.mol_ax.axis("off")
        plt.tight_layout()

        plt.show(False)
        plt.draw()

        self.scores = [[], []]
        self.updated = False

    def update(self, data, smiles):

        self.scores[0].append(data[0])
        self.scores[1].append(data[1])

        if not self.updated:
            self.data = self.score_ax.plot(self.scores)[0]
            self.updated = True

        self.data.set_data(self.scores)
        self.fig.canvas.draw()

        mols = []
        for smile in smiles:
            if len(mols)==8:
                break
            mol = Chem.MolFromSmiles(smile) 
            if mol is not None and "." not in smile:
                mols.append(mol)

        mol_img = Draw.MolsToGridImage(mols, subImgSize=(400, 400), molsPerRow=4)
        self.mol_ax.images = []
        self.mol_ax.imshow(mol_img, interpolation="bicubic")
        

