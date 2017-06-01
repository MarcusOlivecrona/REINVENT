import matplotlib
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

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
        """data is in the form of (step, [scores]) 
           and smiles is a list with the same length
           as [scores]"""

        x, y = data[0], np.mean(data[1])
        self.scores[0].append(x)
        self.scores[1].append(y)

        if not self.updated:
            self.data = self.score_ax.plot(self.scores)[0]
            self.updated = True

        self.data.set_data(self.scores)

        mols = []
        scores = []
        for idx, smile in enumerate(smiles):
            if len(mols)==8:
                break
            mol = Chem.MolFromSmiles(smile) 
            if mol is not None and "." not in smile:
                mols.append(mol)
                scores.append(str(round(data[1][idx], 2)))

        mol_img = Draw.MolsToGridImage(mols, subImgSize=(400, 400),
                                       molsPerRow=4, legends=scores)
        self.mol_ax.images = []
        self.mol_ax.imshow(mol_img, interpolation="bicubic")

        self.fig.canvas.draw()

        

