import os
import time
import csv
import multiprocessing as mp
import subprocess
import pubchempy as pcp
from rdkit import Chem
import pandas as pd
import os
import toml
import tomllib
import math
import re

tokens_not_in_vocab = ['[CH', '[O]','[C', '[o+', '[O+', '[N]', '[NH', '[c', '[OH']

def run_reinvent_cmd(cmd, spec_id):
    p_start = time.time()

    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)

    print(spec_id + " : " + str(time.time() - p_start))
    print(proc.stdout)
    print(proc.stderr)
    return

def run_cmds(filename, max_processes):

    with open(filename, 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        sp_args = list(csv_reader)


    idx = 0
    processes = []
    original_start = time.time()
    while idx < len(sp_args):

        while len(processes) < max_processes:
            if idx < len(sp_args):
                if 'transfer_learning' in sp_args[idx][0]:
                    spec_id = sp_args[idx][0].split('/')[-1].split('_transfer_learning')[0]
                elif 'staged_learning' in sp_args[idx][0]:
                    spec_id = sp_args[idx][0].split('/')[-1].split('_staged_learning')[0]
                else:
                    spec_id = sp_args[idx][0].split('/')[-1].split('_sampling')[0]

                p = mp.Process(target=run_reinvent_cmd, args=[sp_args[idx][0], spec_id])
                p.start()
                processes.append(p)
                idx += 1
            else:
                break
        for p in processes:
            p.join()
        for p in processes:
            p.terminate()
        processes = []



def get_mass(formula):

    parts = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    mass = 0

    for index in range(len(parts)):
        if parts[index].isnumeric():
            continue

        atom = Chem.Atom(parts[index])
        multiplier = int(parts[index + 1]) if len(parts) > index + 1 and parts[index + 1].isnumeric() else 1
        mass += atom.GetMass() * multiplier

    return mass

def generate_cands(k, formula, save_path, target_ik=None):
    cands = []
    try:
        compounds = pcp.get_compounds(formula, 'formula', record_format='json', listkey_count=1000)
        for compound in compounds:
            if not any(substring in compound.canonical_smiles for substring in tokens_not_in_vocab):
                m = Chem.MolFromSmiles(compound.canonical_smiles)
                if m is not None:
                    ik = Chem.inchi.MolToInchiKey(m).split('-')[0]
                    if ik != target_ik:
                        cands.append(compound.canonical_smiles)

        cands = list(set(cands))  # len(cands) == 371
        # write files -> need training and validation
        #80/20 split
        val_len = round(len(cands)*0.2)
        if val_len == 0:
            val_len += 1
        with open(save_path+'generated_config_files/smi/' + k + '_cands.smi', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(0, len(cands) - val_len):
                writer.writerow([cands[i]])
        with open(save_path+'generated_config_files/smi/' + k + '_cands_val.smi', 'w', newline='') as file:
            writer = csv.writer(file)
            for i in range(len(cands) - val_len, len(cands)):
                writer.writerow([cands[i]])
    except:
        cands = []
    return len(cands)

def get_cands(id, formula, save_path, target=None):
    if not os.path.isfile(save_path+'generated_config_files/smi/' + id + '_cands.smi') or not os.path.isfile(
            save_path+'generated_config_files/smi/' + id + '_cands_val.smi'):
        print("Generating candidates for ..." + id)
        num_cands = generate_cands(id, formula, save_path, target)
    else:
        with open(save_path+'generated_config_files/smi/' + id + '_cands.smi') as csvfile:
            rows = csv.reader(csvfile)
            cands = list(rows)
        with open(save_path+'generated_config_files/smi/' + id + '_cands_val.smi') as csvfile:
            rows = csv.reader(csvfile)
            cands_val = list(rows)
        num_cands = len(cands) + len(cands_val)
    return num_cands

def preprocess_data_groundtruth(data,split,mol_dict,save_path):
    spectra = {}
    iks = split['test']
    for k, v in data.items():
        precursor_mz = v['PrecursorMZ']
        adduct = v['Precursor'].split('+')[1].split(']')[0]
        adduct_mass = get_mass(adduct)
        compound_mass = float(precursor_mz) - adduct_mass
        v['CalculatedMass'] = compound_mass
        if v['inchikey'] in iks:
            spectra[k] = v['inchikey']

    structure_info_total = []
    structures = []
    count = 1
    for k, v in spectra.items():
        formula = data[k]['Formula']
        target_ik = data[k]['inchikey'].split('-')[0]
        target_mass = data[k]['CalculatedMass']
        precursor_mz = data[k]['PrecursorMZ']
        m = mol_dict[v]
        smiles = Chem.MolToSmiles(m)
        count += 1
        num_cands = get_cands(k, formula, save_path, target=target_ik)
        structure_info_total.append([k, v, formula, target_ik, target_mass, precursor_mz, num_cands, smiles])
        structures.append(target_ik)

    with open(save_path+'generated_config_files/struct_info.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(structure_info_total)

    return structure_info_total


def preprocess_data(data,save_path):
    structure_info_total = []
    count = 1
    for k, v in data.items():
        formula = v['Formula']
        precursor_mz = v['PrecursorMZ']
        adduct = v['Precursor'].split('+')[1].split(']')[0]
        adduct_mass = get_mass(adduct)
        compound_mass = float(precursor_mz) - adduct_mass
        v['CalculatedMass'] = compound_mass
        count += 1
        num_cands = get_cands(k, formula, save_path, target=None)
        structure_info_total.append([k, v, formula, compound_mass, precursor_mz, num_cands])

    with open(save_path+'generated_config_files/struct_info.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(structure_info_total)

    return structure_info_total


def write_tomls(query, save_path, REINVENT_PATH, hyperparameters):
    toml_path = save_path + 'generated_config_files/toml/'
    file_path = save_path
    reinvent_path = REINVENT_PATH
    tl_num_epochs = hyperparameters['tl_num_epochs']
    tl_save_every_n_epochs = hyperparameters['tl_save_every_n_epochs']
    num_carbons_min_steps = hyperparameters['num_carbons_min_steps']
    num_carbons_max_steps = hyperparameters['num_carbons_max_steps']
    rl_min_steps = hyperparameters['rl_min_steps']
    rl_max_steps = hyperparameters['rl_max_steps']
    rl_max_score = hyperparameters['rl_max_score']
    rl_weight = hyperparameters['rl_weight']
    rl_batch_size = hyperparameters['rl_batch_size']

    num_cands = int(query[6])
    train_cands = math.ceil(num_cands * 0.8)
    tl_batch_size = 32
    if tl_batch_size * tl_batch_size < train_cands:
        tl_batch_size = math.ceil(train_cands / tl_batch_size)

    # transfer_learning (candidates)
    with open('./toml_templates/REINVENT_transfer_learning.toml', "rb") as f:
        new_toml = tomllib.load(f)

    # change any parameters from the standard
    new_toml['json_out_config'] = file_path + 'gen_model_output/json/' + query[0] + '_transfer_learning.json'
    new_toml['parameters']['num_epochs'] = tl_num_epochs
    new_toml['parameters']['save_every_n_epochs'] = tl_save_every_n_epochs
    new_toml['parameters']['batch_size'] = tl_batch_size
    new_toml['parameters']['input_model_file'] = reinvent_path + 'priors/reinvent.prior'
    new_toml['parameters']['smiles_file'] = file_path + 'generated_config_files/smi/' + query[2] + '_cands.smi'
    new_toml['parameters']['validation_smiles_file'] = file_path + 'generated_config_files/smi/' + query[
        2] + '_cands_val.smi'

    new_toml['parameters']['output_model_file'] = file_path + 'gen_model_output/TL/' + query[0] + '_TL_reinvent.model'

    with open(toml_path + query[0] + '_transfer_learning.toml', 'w') as f:
        toml.dump(new_toml, f)

    # reinforcement stage 1 learning (number of carbons)
    with open('./toml_templates/REINVENT_staged_learning_numCar.toml', "rb") as f:
        new_toml = tomllib.load(f)

    new_toml['json_out_config'] = file_path + 'gen_model_output/json/' + query[0] + '_staged_learning_numCar.json'
    new_toml['parameters']['summary_csv_prefix'] = file_path + 'gen_model_output/csv/' + query[0] + '_staged_learning_numCar'
    new_toml['parameters']['prior_file'] = reinvent_path + 'priors/reinvent.prior'
    new_toml['parameters']['agent_file'] = file_path + 'gen_model_output/TL/' + query[0] + '_TL_reinvent.model'
    new_toml['parameters']['batch_size'] = rl_batch_size
    new_toml['stage'][0]['chkpt_file'] = file_path + 'gen_model_output/RL/' + query[0] + '_staged_numCar.chkpt'
    new_toml['stage'][0]['max_score'] = rl_max_score
    new_toml['stage'][0]['min_steps'] = num_carbons_min_steps
    new_toml['stage'][0]['max_steps'] = num_carbons_max_steps
    new_toml['stage'][0]['scoring']['component'][0]['numCarbons']['endpoint'][0]['weight'] = rl_weight
    new_toml['stage'][0]['scoring']['component'][0]['numCarbons']['endpoint'][0]['params']['formula'] = query[2]

    with open(toml_path + query[0] + '_staged_learning_numCar.toml', 'w') as f:
        toml.dump(new_toml, f)

    # reinforcement stage 2 learning (number of all heavy atoms)
    with open('./toml_templates/REINVENT_staged_learning_molForm.toml', "rb") as f:
        new_toml = tomllib.load(f)

    new_toml['json_out_config'] = file_path + 'gen_model_output/json/' + query[0] + '_staged_learning_molForm.json'
    new_toml['parameters']['summary_csv_prefix'] = file_path + 'gen_model_output/csv/' + query[0] + '_staged_learning_molForm'
    new_toml['parameters']['prior_file'] = reinvent_path + 'priors/reinvent.prior'
    new_toml['parameters']['agent_file'] = file_path + 'gen_model_output/RL/' + query[0] + '_staged_numCar.chkpt'
    new_toml['parameters']['batch_size'] = rl_batch_size
    new_toml['stage'][0]['chkpt_file'] = file_path + 'gen_model_output/RL/' + query[0] + '_staged_molForm.chkpt'
    new_toml['stage'][0]['max_score'] = rl_max_score
    new_toml['stage'][0]['min_steps'] = rl_min_steps
    new_toml['stage'][0]['max_steps'] = rl_max_steps
    new_toml['stage'][0]['scoring']['component'][0]['MolecularFormula']['endpoint'][0]['weight'] = rl_weight
    new_toml['stage'][0]['scoring']['component'][0]['MolecularFormula']['endpoint'][0]['params']['formula'] = query[2]

    with open(toml_path + query[0] + '_staged_learning_molForm.toml', 'w') as f:
        toml.dump(new_toml, f)

    # reinforcement staged learning (mass)
    with open('./toml_templates/REINVENT_staged_learning_molFormMass.toml', "rb") as f:
        new_toml = tomllib.load(f)

    new_toml['json_out_config'] = file_path + 'gen_model_output/json/' + query[0] + '_staged_learning_molFormMass.json'
    new_toml['parameters']['summary_csv_prefix'] = file_path + 'gen_model_output/csv/' + query[0] + '_staged_learning_molFormMass'
    new_toml['parameters']['prior_file'] = reinvent_path + 'priors/reinvent.prior'
    new_toml['parameters']['agent_file'] = file_path + 'gen_model_output/RL/' + query[0] + '_staged_molForm.chkpt'
    new_toml['parameters']['batch_size'] = rl_batch_size
    new_toml['stage'][0]['chkpt_file'] = file_path + 'gen_model_output/RL/' + query[0] + '_staged_molFormMass.chkpt'
    new_toml['stage'][0]['max_score'] = rl_max_score
    new_toml['stage'][0]['min_steps'] = rl_min_steps
    new_toml['stage'][0]['max_steps'] = rl_max_steps
    new_toml['stage'][0]['scoring']['component'][0]['MolecularFormulaAndMass']['endpoint'][0]['weight'] = rl_weight
    # new_toml['stage'][0]['scoring']['component'][0]['numOtherHeavyAtoms']['endpoint'][0]['params.target_carbons'] = target_carbons
    new_toml['stage'][0]['scoring']['component'][0]['MolecularFormulaAndMass']['endpoint'][0]['params']['formula'] = \
    query[2]
    new_toml['stage'][0]['scoring']['component'][0]['MolecularFormulaAndMass']['endpoint'][0]['params']['mass'] = float(
        query[4])

    with open(toml_path + query[0] + '_staged_learning_molFormMass.toml', 'w') as f:
        toml.dump(new_toml, f)

    # sampling ALL
    sample_cmds = write_sample_tomls(query, save_path)

    run_cmds = ["reinvent -l " + file_path + "gen_model_output/log/" + query[
        0] + "_transfer_learning.log " + file_path + "generated_config_files/toml/" + query[
                    0] + "_transfer_learning.toml",
                "reinvent -l " + file_path + "gen_model_output/log/" + query[
                    0] + "_staged_learning_numCar.log " + file_path + "generated_config_files/toml/" + query[
                    0] + "_staged_learning_numCar.toml",
                "reinvent -l " + file_path + "gen_model_output/log/" + query[
                    0] + "_staged_learning_molForm.log " + file_path + "generated_config_files/toml/" + query[
                    0] + "_staged_learning_molForm.toml",
                "reinvent -l " + file_path + "gen_model_output/log/" + query[
                    0] + "_staged_learning_molFormMass.log " + file_path + "generated_config_files/toml/" + query[
                    0] + "_staged_learning_molFormMass.toml"]
    run_cmds.extend(sample_cmds)

    return run_cmds


def write_sample_tomls(query, save_path):
    toml_path = save_path + 'generated_config_files/toml_samples/'

    if not os.path.isdir(toml_path):
        os.mkdir(toml_path)

    output_path = save_path + 'gen_model_output/'
    sample_models = ['TL_reinvent.model', 'staged_numCar.chkpt', 'staged_molForm.chkpt', 'staged_molFormMass.chkpt']
    sample_extensions = []
    for model in sample_models:
        sample_extensions.append(model + '_samples/')
    for ext in sample_extensions:
        if not os.path.isdir(output_path + ext):
            os.mkdir(output_path + ext)
    num_smiles_sampled = 1000

    run_cmds = []
    # sampling tomls
    for i in range(0, len(sample_extensions)):
        with open('./toml_templates/REINVENT_sampling.toml', "rb") as f:
            new_toml = tomllib.load(f)

        new_toml['json_out_config'] = output_path + sample_extensions[i] + query[0] + '_sampling.json'
        if sample_models[i][0:2] == 'TL':
            new_toml['parameters']['model_file'] = output_path + 'TL/' + query[0] + '_' + sample_models[i]
        else:
            new_toml['parameters']['model_file'] = output_path + 'RL/' + query[0] + '_' + sample_models[i]
        new_toml['parameters']['output_file'] = output_path + sample_extensions[i] + query[0] + '_sampling.csv'
        new_toml['parameters']['num_smiles'] = num_smiles_sampled

        with open(toml_path + query[0] + '_sampling_' + sample_models[i] + '.toml', 'w') as f:
            toml.dump(new_toml, f)

        run_cmds.append(
            "reinvent -l " + output_path + sample_extensions[i] + query[0] + "_sampling_" + sample_models[
                i] + ".log " + toml_path + query[
                0] + "_sampling_" + sample_models[i] + ".toml")

    return run_cmds


def filter_generated_strucures(input_data, ik_spec_dict, save_path, output_path):
    for idx, row in input_data.iterrows():
        id = row[0]
        if not os.path.isfile(save_path + 'all_generated_candidates/' + id + '_generated_candidates.csv'):
            for val in ik_spec_dict[row[1]]:
                if os.path.isfile(save_path + 'all_generated_candidates/' + val + '_generated_candidates.csv'):
                    try:
                        gen_cands = pd.read_csv(save_path + 'all_generated_candidates/' + val + '_generated_candidates.csv',
                                                header=None)[0].to_list()
                    except:
                        gen_cands = []
                    with open(save_path + 'all_generated_candidates/' + id + '_generated_candidates.csv', 'w',
                              newline='') as file:
                        writer = csv.writer(file)
                        for smi in gen_cands:
                            writer.writerow([smi])

    return

def evaluate_generated_structures(input_data, evaluation, save_path, csv_path, smi_path):
    no_results = []
    novelty = {}
    num_gen_cands_dict = {}
    corr_gen = []
    num_pubchem_cands = []
    min_pubchem = 1000000
    max_pubchem = -1
    count_empty = 0
    for idx, row in input_data.iterrows():
        id = row[0]
        if evaluation:
            target_ik = row[1].split('-')[0]
        formula = row[2]
        try:
            gen_cands = \
                pd.read_csv(save_path + 'all_generated_candidates/' + id + '_generated_candidates.csv', header=None)[
                    0].to_list()
            cands = []
            cnads_val = []
            if os.path.exists(smi_path + formula + '_cands.smi'):
                with open(smi_path + formula + '_cands.smi') as csvfile:
                    rows = csv.reader(csvfile)
                    cands = list(rows)
                with open(smi_path + formula + '_cands_val.smi') as csvfile:
                    rows = csv.reader(csvfile)
                    cands_val = list(rows)
            else:
                with open(smi_path + id + '_cands.smi') as csvfile:
                    rows = csv.reader(csvfile)
                    cands = list(rows)
                with open(smi_path + id + '_cands_val.smi') as csvfile:
                    rows = csv.reader(csvfile)
                    cands_val = list(rows)
            pubchem_iks = []
            for c in cands:
                ik = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(c[0]))
                pubchem_iks.append(ik.split('-')[0])
            for c in cands_val:
                ik = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(c[0]))
                pubchem_iks.append(ik.split('-')[0])
            num_pubchem_cands.append(len(pubchem_iks))
            if len(pubchem_iks) > max_pubchem:
                max_pubchem = len(pubchem_iks)
            if len(pubchem_iks) < min_pubchem:
                min_pubchem = len(pubchem_iks)
            pubchem_iks = list(set(pubchem_iks))

            gen_iks = {}
            for c in gen_cands:
                ik = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(c))
                ikblock1 = ik.split('-')[0]
                if ikblock1 not in gen_iks.keys():
                    gen_iks[ikblock1] = c

            new = [value for value in list(gen_iks.keys()) if
                   value not in pubchem_iks]  # need a list of smiles not inchikey block 1
            novelty[id] = len(new) / len(gen_iks.keys())
            num_gen_cands_dict[id] = len(gen_iks.keys())

            if evaluation:
                if target_ik in gen_iks.keys():
                    corr_gen.append(id)

            # save only novel new structures
            with open(csv_path + id + '_generated_candidates.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                for ik in new:
                    writer.writerow([gen_iks[ik]])

        except pd.errors.EmptyDataError:
            no_results.append(id)
            count_empty += 1

    print("Number of queries: " + str(len(novelty)))
    print("Novelty: " + str(sum(novelty.values()) / (len(novelty) + count_empty)))
    print("Average number of generated molecules: " + str(
        sum(num_gen_cands_dict.values()) / (len(num_gen_cands_dict) + count_empty)))
    if evaluation:
        print("Accuracy: " + str(len(corr_gen) / (len(novelty) + count_empty)))
    print("Average number of PubChem candidates: " + str(sum(num_pubchem_cands) / len(num_pubchem_cands)))
    print("Min number of PubChem candidates: " + str(min_pubchem))
    print("Max number of PubChem candidates: " + str(max_pubchem))

    return







