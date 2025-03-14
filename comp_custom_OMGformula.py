"""Compute scores to aid with spectra annotation

written by Margaret Martin
"""

from __future__ import annotations

# __all__ = ["ChemProp"]
from dataclasses import dataclass, field
from typing import List
import logging

# import chemprop
import numpy as np
from rdkit import Chem
import re

from .component_results import ComponentResults
from .add_tag import add_tag
from reinvent.scoring.utils import suppress_output
from ..normalize import normalize_smiles

logger = logging.getLogger('reinvent')


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    formula: List[str]


@add_tag("__component")
class numCarbons:
    def __init__(self, params: Parameters):

        self.formula = params.formula[0]

        self.smiles_type = 'rdkit_smiles'


    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        atoms_in_vocab = ['Br', 'C', 'Cl', 'F', 'N', 'O', 'S', 'H']

        assert 'C' in self.formula
        start = self.formula.find('C')
        loc = start + 1
        while loc < len(self.formula) and self.formula[loc].isnumeric():
            loc += 1
        if loc == start + 1:
            target_carbons = 1
        else:
            target_carbons = int(self.formula[start + 1:loc])

        scores = []

        for smiles in smilies:
            m = Chem.MolFromSmiles(smiles)
            f = Chem.rdMolDescriptors.CalcMolFormula(m)
            f_dict = {}
            for atom in atoms_in_vocab:
                num = 0
                if atom in f:
                    start = f.find(atom)
                    loc = start+1
                    while loc<len(f) and f[loc].isnumeric():
                        loc+=1
                    # if loc == len(f):
                    if loc==start+1:
                        num = 1
                    else:
                        num = int(f[start+1:loc])
                f_dict[atom] = num

            if f_dict['C'] == target_carbons:
                scores.append(1.)
            else:
                scores.append(0.)

        return ComponentResults([np.array(scores)])



@add_tag("__component")
class numOtherHeavyAtoms:
    def __init__(self, params: Parameters):

        self.formula = params.formula[0]
        self.smiles_type = 'rdkit_smiles'


    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        other_heavy_atoms_in_vocab = ['Br', 'Cl', 'F', 'N', 'O', 'S']

        target_f_dict = {}
        for atom in other_heavy_atoms_in_vocab:
            num = 0
            if atom in self.formula:
                start = self.formula.find(atom)
                loc = start + 1
                while loc < len(self.formula) and self.formula[loc].isnumeric():
                    loc += 1
                if loc == start + 1:
                    num = 1
                else:
                    num = int(self.formula[start + 1:loc])
            target_f_dict[atom] = num


        scores = []

        for smiles in smilies:
            m = Chem.MolFromSmiles(smiles)
            f = Chem.rdMolDescriptors.CalcMolFormula(m)
            f_dict = {}
            for atom in other_heavy_atoms_in_vocab:
                num = 0
                if atom in f:
                    start = f.find(atom)
                    loc = start+1
                    while loc<len(f) and f[loc].isnumeric():
                        loc+=1
                    if loc==start+1:
                        num = 1
                    else:
                        num = int(f[start+1:loc])
                f_dict[atom] = num

            if f_dict == target_f_dict:
                scores.append(1.)
            else:
                scores.append(0.)

        return ComponentResults([np.array(scores)])


@add_tag("__component")
class MolecularFormula:
    def __init__(self, params: Parameters):
        self.formula = params.formula[0]
        self.smiles_type = 'rdkit_smiles'

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        heavy_atoms_in_vocab = ['Br', 'C', 'Cl', 'F', 'N', 'O', 'S']

        target_f_dict = {}
        for atom in heavy_atoms_in_vocab:
            num = 0
            if atom in self.formula:
                start = self.formula.find(atom)
                loc = start + 1
                while loc < len(self.formula) and self.formula[loc].isnumeric():
                    loc += 1
                # if loc == len(f):
                if loc == start + 1:
                    num = 1
                else:
                    num = int(self.formula[start + 1:loc])
            target_f_dict[atom] = num


        scores = []

        for smiles in smilies:
            m = Chem.MolFromSmiles(smiles)
            f = Chem.rdMolDescriptors.CalcMolFormula(m)
            f_dict = {}
            for atom in heavy_atoms_in_vocab:
                num = 0
                if atom in f:
                    start = f.find(atom)
                    loc = start+1
                    while loc<len(f) and f[loc].isnumeric():
                        loc+=1
                    if loc==start+1:
                        num = 1
                    else:
                        num = int(f[start+1:loc])
                f_dict[atom] = num

            if f_dict == target_f_dict:
                scores.append(1.)
            else:
                scores.append(0.)

        return ComponentResults([np.array(scores)])


