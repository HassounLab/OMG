import pickle
from rdkit.Chem import AllChem
from rdkit import Chem
import yaml
from utils_step2 import DatasetBuilder, MultiView_data, collate_contr_views, Print, load_models, contrastive_loss
from utils_step2 import fp_bce_loss, fp_cos_loss, fp_cos, print_hp, Spectra_data, collate_spectra_data
from utils_step2 import MyEarlyStopping, save_all_models
from dataset import load_contrastive_data, load_spectra_wneg_data, load_cand_test_data
from dataset import mol_to_graph
import sys
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import pickle
from models import MolEnc, SpecEncMLP_BIN, SpecEncMLP_SIN, SpecEncTFM, INTER_MLP, INTER_MLP2
import matplotlib.pyplot as plt
import time
from train_contr import train_contr
import torch.nn.functional as F
from sklearn.metrics import auc, roc_auc_score, roc_curve, average_precision_score
from collections import defaultdict
import numpy as np

from rdkit import DataStructs
import csv
import pandas as pd
from argparse import ArgumentParser




from utils_step2 import create_JESTR_molgraph_dict, apply_JESTR

from evaluation import evaluate_ranked_structures



parser = ArgumentParser()
parser.add_argument('--data_path', default='../JESTR1-main/data/NPLIB1/')
parser.add_argument('--OMG_data_path', default='../data/')
parser.add_argument('--evaluation', default=True)
args = vars(parser.parse_args())
data_path = args['data_path']
evaluation = args['evaluation']
input_path = args['OMG_data_path']

save_path = data_path + 'output/ranking/'
csv_path = data_path + 'output/molecular_generation/filtered_generated_candidates/'
input_data = pd.read_csv(data_path+'output/molecular_generation/generated_config_files/struct_info.csv')


######################

if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path+'model_output/'):
    os.mkdir(save_path+'model_output/')


ik_spec_dict = {}
for idx, row in input_data.iterrows():
    if row[1] not in ik_spec_dict.keys():
        ik_spec_dict[row[1]] = [row[0]]
    else:
        ik_spec_dict[row[1]].append(row[0])


######################



#JESTR params
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
with open('../JESTR1-main/params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

if len(sys.argv) > 1:
    for i in range(len(sys.argv) - 1):
        key, value_raw = sys.argv[i + 1].split("=")
        print(str(key) + ": " + value_raw)
        try:
            params[key] = int(value_raw)
        except ValueError:
            try:
                params[key] = float(value_raw)
            except ValueError:
                params[key] = value_raw


dir_path = "./"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logfile = params['logfile']
output = open(logfile, "a")

ms_intensity_threshold = 0.0

dataset_builder = DatasetBuilder(params['exp'])
dataset_builder.init(dir_path, params['fp_path'], ms_intensity_threshold)

JESTR_data_path = dir_path + dataset_builder.data_dir

#############################################################################
if os.path.exists(save_path + "model_output/molgraph_dict_with_cands.pkl"):
    with open(save_path + "model_output/molgraph_dict_with_cands.pkl", 'rb') as f:
        molgraph_dict = pickle.load(f)
    with open(save_path + "model_output/ik_to_id_dict_spec.pkl", 'rb') as f:
        ik_to_id_dict_spec = pickle.load(f)
    with open(save_path + "model_output/match_results.pkl", 'rb') as f:
        match_results = pickle.load(f)

else:
    molgraph_dict, ik_to_id_dict_spec, match_results = create_JESTR_molgraph_dict(dataset_builder, params, device, csv_path, save_path)

dataset_builder.molgraph_dict = molgraph_dict
del molgraph_dict


#############################################################################
# Collect generated smiles into inchikey and smiles dict
if os.path.exists(save_path + "model_output/ik_smiles_dict.pkl"):
    with open(save_path + "model_output/ik_smiles_dict.pkl", 'rb') as f:
        ik_smiles_dict = pickle.load(f)
else:
    ik_smiles_dict = {}
    for file in os.listdir(csv_path):
        assert file[-25:] == '_generated_candidates.csv'
        l_smiles = []
        try:
            l_smiles = pd.read_csv(csv_path + file, header=None)[0].to_list()
        except pd.errors.EmptyDataError:
            continue
        for smi in l_smiles:
            m = Chem.MolFromSmiles(smi)
            Chem.RemoveStereochemistry(m)
            gen_ik = Chem.inchi.MolToInchiKey(m)
            if gen_ik not in ik_smiles_dict.keys():
                ik_smiles_dict[gen_ik] = Chem.MolToSmiles(m)

    for idx, row in input_data.iterrows():
        if row[1] not in ik_smiles_dict.keys():
            ik_smiles_dict[row[1]] = row[7]

    # dump the dictionaries
    with open(save_path + "model_output/ik_smiles_dict.pkl", "wb") as f:
        pickle.dump(ik_smiles_dict, f)


#############################
if os.path.exists(save_path+"model_output/results_smiles.pkl"):
    with open(save_path + "model_output/results_smiles.pkl", 'rb') as f:
        smiles_sim_dict = pickle.load(f)
else:
    smiles_sim_dict = apply_JESTR(ik_smiles_dict, match_results, params, dataset_builder, ik_to_id_dict_spec, output, device, JESTR_data_path, save_path)


###############################
#evaluate against benchmarks
if evaluation:
    temp = []
    for idx, row in input_data.iterrows():
        true = row[7]
        formula = row[2]
        temp_id = row[0]
        if temp_id in smiles_sim_dict.keys():
            sort_cands = {k: v for k, v in
                          sorted(smiles_sim_dict[temp_id].items(), key=lambda item: item[1], reverse=True)}
            if len(sort_cands.keys()) > 0:
                temp.append([true, list(sort_cands.keys())])
            else:
                temp.append([true, ['-']])
        else:
            temp.append([true, ['-']])
    df = pd.DataFrame(temp, columns=["true","pred"])

    evaluate_ranked_structures(df)




