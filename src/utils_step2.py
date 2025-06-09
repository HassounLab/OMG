import os
import sys
import csv
import pandas as pd
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import torch
from dataset import mol_to_graph
from train_contr import train_contr
from torch.utils.data import DataLoader

# Add the JESTR utils directory to sys.path
utils_path = os.path.abspath(os.path.join('..', 'JESTR1-main', 'utils'))
sys.path.append(utils_path)
from utils import collate_spectra_data, Spectra_data



def compute_ecfp_similarity(m1, m2):
    fp1 = AllChem.GetMorganFingerprint(m1, 2)
    fp2 = AllChem.GetMorganFingerprint(m2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

def create_JESTR_molgraph_dict(dataset_builder, params, device, csv_path, save_path):
    molgraph_dict = {}

    #############################################################################
    # add necessary info to molgraph_dict
    # collect any necessary metadata/ground truth data

    ik_to_id_dict_spec = {}
    # molecular data
    match_results = {}  # [mm, cm, em]
    for file in os.listdir(csv_path):
        assert file[-25:] == '_generated_candidates.csv'
        spec_id = file[0:-25]
        ik = dataset_builder.data_dict[spec_id]['inchikey']
        if ik in ik_to_id_dict_spec.keys():
            ik_to_id_dict_spec[ik].append(spec_id)
        else:
            ik_to_id_dict_spec[ik] = [spec_id]

        match_results[spec_id] = {}
        l_smiles = []
        try:
            l_smiles = pd.read_csv(csv_path + file, header=None)[0].to_list()
        except pd.errors.EmptyDataError:
            print(file + " is empty")
        for smi in l_smiles:
            m = Chem.MolFromSmiles(smi)
            Chem.RemoveStereochemistry(m)
            gen_ik = Chem.inchi.MolToInchiKey(m)
            if gen_ik not in molgraph_dict.keys():
                mol_to_graph(m, gen_ik, molgraph_dict, params,
                             device)

            if gen_ik not in match_results[spec_id].keys():
                match_results[spec_id][gen_ik] = 0

    # dump the dictionaries
    with open(save_path + "jestr_output/molgraph_dict_with_cands.pkl", "wb") as f:
        pickle.dump(molgraph_dict, f)
    with open(save_path + "jestr_output/ik_to_id_dict_spec.pkl", "wb") as f:
        pickle.dump(ik_to_id_dict_spec, f)
    return molgraph_dict, ik_to_id_dict_spec, match_results


def load_spectra_data_single_OMG(dataset_builder, params, ik, id_list, device):
    data_dict = dataset_builder.data_dict
    mol_dict = dataset_builder.mol_dict
    molgraph_dict = dataset_builder.molgraph_dict
    fp_dict = dataset_builder.fp_dict
    in_to_id_dict = dataset_builder.in_to_id_dict
    ret_list = []
    cnt = 0

    mol = mol_dict.get(ik, None)

    if not mol:
        return None, None

    if not mol_to_graph(mol, ik, molgraph_dict, params, device):
        return None, None

    l_spec_ids = []
    for idx in id_list:
        v = data_dict[idx]
        if v['Precursor'] != '[M+H]+':
            continue
        cnt += 1
        if params['debug'] and cnt > 100:
            break

        inchi2 = v['inchikey']
        mol = mol_dict.get(inchi2, None)
        if not mol:
            continue

        if not mol_to_graph(mol, inchi2, molgraph_dict, params, device):
            continue

        msi = v['ms']
        fp = [0.0] * 4096
        ret_list.append((ik, msi, fp, 1))
        l_spec_ids.append(idx)

    return ret_list, l_spec_ids

def apply_JESTR(ik_smiles_dict, match_results, params, dataset_builder, ik_to_id_dict_spec, output, device, data_path, save_path):
    params['pretrained_mol_enc_model'] = './pretrained_models/pretrained_mol_enc_model_1707829192911_best.pt'
    params['pretrained_spec_enc_model'] = './pretrained_models/pretrained_spec_enc_model_1707829192911_best.pt'

    mol_enc_model_contr, spec_enc_model_contr, models_list_contr = train_contr(dataset_builder,
                                                                               dataset_builder.molgraph_dict, params,
                                                                               output,
                                                                               device, data_path, True)

    mol_enc_model_contr.eval()
    spec_enc_model_contr.eval()

    smiles_sim_dict = {}
    sim_dict = {}
    mol_enc_total = {}
    spec_enc_total = {}
    dist = torch.nn.CosineSimilarity()
    for ik in ik_to_id_dict_spec.keys():
        id_list = ik_to_id_dict_spec[ik]
        spec_list, l_spec_ids = load_spectra_data_single_OMG(dataset_builder, params, ik, id_list,
                                                             device)  # spec_list is a list of spectra (one spectrum per id in id_list)

        if spec_list != None:
            if len(spec_list) != 0:
                spec_test_ds = Spectra_data(spec_list)  # convert spec_list to spectra data object
                collate_fn = collate_spectra_data(dataset_builder.molgraph_dict,
                                                  params)  # utils.collate_spectra_data object

                dl_params = {'batch_size': params['batch_size_val_final'],
                             'shuffle': False}
                spec_test_dl = DataLoader(spec_test_ds, collate_fn=collate_fn, **dl_params)


                spec_enc_temp = torch.Tensor().to(torch.device(device))

                assert sum(1 for _ in enumerate(spec_test_dl)) == 1
                for batch_id, (batch_g, mz_b, int_b, pad, fp_b, y, lengths, inchi) in enumerate(spec_test_dl):
                    batch_g = batch_g.to(torch.device(device))
                    mz_b = mz_b.to(torch.device(device))
                    int_b = int_b.to(torch.device(device))
                    pad = pad.to(torch.device(device))
                    fp_b = fp_b.to(torch.device(device))
                    y = y.to(torch.device(device))
                    with torch.no_grad():
                        spec_enc = spec_enc_model_contr(mz_b, int_b, pad,
                                                        lengths)  # embedding of corresponding spectrum
                    spec_enc_temp = torch.cat([spec_enc_temp, spec_enc])
                assert len(l_spec_ids) == len(spec_enc_temp)
                for i in range(0, len(l_spec_ids)):
                    spec_enc_total[l_spec_ids[i]] = spec_enc_temp[i]

                for id in id_list:
                    if id in sim_dict.keys():
                        print(id + " is ranked again by JESTR")
                    sim_dict[id] = {}

                    smiles_id = ik_smiles_dict[ik]
                    smiles_sim_dict[smiles_id] = {}
                    if id in spec_enc_total.keys():
                        for gen_ik in match_results[id].keys():
                            if gen_ik in mol_enc_total.keys():
                                mol_enc = mol_enc_total[gen_ik]
                            else:
                                batch_g = dataset_builder.molgraph_dict[gen_ik]
                                batch_g = batch_g.to(torch.device(device))
                                with torch.no_grad():
                                    mol_enc = mol_enc_model_contr(batch_g, batch_g.ndata[
                                        'h'])  # embedding of molecule
                                mol_enc_total[gen_ik] = mol_enc
                            sim = dist(mol_enc, spec_enc_total[id])
                            sim_dict[id][gen_ik] = float(sim[0])

                            gen_smiles = ik_smiles_dict[gen_ik]
                            smiles_sim_dict[smiles_id][gen_smiles] = float(sim[0])

            else:
                spec_list, l_spec_ids = load_spectra_data_single_OMG(dataset_builder, params, ik, id_list,
                                                                     device)  # spec_list is a list of spectra (one spectrum per id in id_list)

        else:
            spec_list, l_spec_ids = load_spectra_data_single_OMG(dataset_builder, params, ik, id_list,
                                                                 device)  # spec_list is a list of spectra (one spectrum per id in id_list)
            for id in id_list:
                sim_dict[id] = {}

    with open(save_path + "jestr_output/jestr_sim_dict.pkl", "wb") as f:
        pickle.dump(sim_dict, f)

    with open(save_path + "jestr_output/results_smiles.pkl", "wb") as f:
        pickle.dump(smiles_sim_dict, f)

    return smiles_sim_dict