import pickle
import os

import pulp
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

RDLogger.DisableLog("rdApp.*")

from myopic_mces.myopic_mces import MCES


class MyopicMCES:
    def __init__(
        self,
        ind=0,
        solver=pulp.listSolvers(onlyAvailable=True)[0],
        threshold=15,
        always_stronger_bound=True,
        solver_options=None,
    ):
        self.ind = ind
        self.solver = solver
        self.threshold = threshold
        self.always_stronger_bound = always_stronger_bound
        if solver_options is None:
            solver_options = dict(msg=0)  # make ILP solver silent
        self.solver_options = solver_options

    def __call__(self, smiles_1, smiles_2):
        retval = MCES(
            s1=smiles_1,
            s2=smiles_2,
            ind=self.ind,
            threshold=self.threshold,
            always_stronger_bound=self.always_stronger_bound,
            solver=self.solver,
            solver_options=self.solver_options,
        )
        dist = retval[1]
        return dist

def evaluate_ranked_structures(df):
    top_ks = [1,10]

    mces_thld = 100
    mces_cache = {}
    myopic_mces = MyopicMCES()

    total_len = len(df)

    for k in top_ks:
        result_metric = {"accuracy": 0, "similarity": 0, "MCES": 0}
        for idx, row in tqdm(df.iterrows()):
            smile = row["true"]
            pred_smiles = row["pred"][:k]
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                total_len -= 1
                continue
            pred_mols = [Chem.MolFromSmiles(pred) for pred in pred_smiles]
            in_top_k = MolToInchiKey(mol).split("-")[0] in [
                MolToInchiKey(pred).split("-")[0] if pred is not None else None
                for pred in pred_mols
            ]
            result_metric["accuracy"] += int(in_top_k)
            dists = []
            for pred, pred_mol in zip(pred_smiles, pred_mols):
                if pred_mol is None:
                    dists.append(mces_thld)
                else:
                    if (smile, pred) not in mces_cache:
                        mce_val = myopic_mces(smile, pred)
                        mces_cache[(smile, pred)] = mce_val
                    dists.append(mces_cache[(smile, pred)])
            mol_fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            pred_fps = [
                GetMorganFingerprintAsBitVect(pred, radius=2, nBits=2048) if pred is not None else None for pred in pred_mols
            ]
            sims = [
                TanimotoSimilarity(mol_fp, pred) if pred is not None else 0 for pred in pred_fps
            ]

            result_metric["similarity"] += max(sims)
            result_metric["MCES"] += min(min(dists), mces_thld)

        for key in result_metric:
            result_metric[key] = result_metric[key] / total_len #MARGARET changed len(true_smile)

        print('Top ' + str(k) + ' results:')
        print(result_metric)
