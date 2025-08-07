import os
import csv
import pickle
from rdkit import Chem
from argparse import ArgumentParser


from utils_step1 import preprocess_data, preprocess_data_groundtruth, write_tomls, run_cmds, filter_generated_strucures, evaluate_generated_structures


parser = ArgumentParser()
parser.add_argument('--spectra_data_path', default='../JESTR1-main/data/NPLIB1/')
parser.add_argument('--OMG_data_path', default='../data/')
parser.add_argument('--evaluation', default=True)
parser.add_argument('--reinvent_path', default='../REINVENT4/')
parser.add_argument('--max_processes', default=1)
args = vars(parser.parse_args())
data_path = args['spectra_data_path']
input_path = args['OMG_data_path']
evaluation = args['evaluation']
REINVENT_PATH = args['reinvent_path']
max_processes = args['max_processes']


hyperparameters = {
    "tl_num_epochs" : 100,
    "tl_save_every_n_epochs" : 10,
    "num_carbons_min_steps" : 200,
    "num_carbons_max_steps" : 400,
    "rl_min_steps" : 100,
    "rl_max_steps" : 150,
    "rl_max_score" : 0.9,
    "rl_weight" : 0.7,
    "rl_batch_size" : 32,
    }

# data
#############################
if evaluation:
    with open(data_path + 'data_dict.pkl', 'rb') as f:
        data = pickle.load(f)
    with open(data_path + 'split.pkl', 'rb') as f:
        split = pickle.load(f)
    if os.path.exists(data_path + 'mol_dict.pkl'):
        with open(data_path + 'mol_dict.pkl', 'rb') as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
else:
    with open(data_path + 'data_dict.pkl', 'rb') as f:
        data = pickle.load(f)
#############################



save_path = input_path + 'output/molecular_generation/'
csv_path = save_path + 'filtered_generated_candidates/'
smi_path = save_path + 'generated_config_files/smi/'

if not os.path.isdir(input_path):
    os.mkdir(input_path)
if not os.path.isdir(input_path+'output/'):
    os.mkdir(input_path+'output/')
if not os.path.isdir(input_path+'output/molecular_generation/'):
    os.mkdir(input_path+'output/molecular_generation/')
if not os.path.isdir(save_path + 'generated_config_files'):
    os.mkdir(save_path + 'generated_config_files')
if not os.path.isdir(save_path + 'generated_config_files/toml'):
    os.mkdir(save_path + 'generated_config_files/toml')
if not os.path.isdir(save_path + 'generated_config_files/smi'):
    os.mkdir(save_path + 'generated_config_files/smi')
if not os.path.isdir(save_path + 'gen_model_output'):
    os.mkdir(save_path + 'gen_model_output')
if not os.path.isdir(save_path + 'gen_model_output/TL'):
    os.mkdir(save_path + 'gen_model_output/TL')
if not os.path.isdir(save_path + 'gen_model_output/RL'):
    os.mkdir(save_path + 'gen_model_output/RL')
if not os.path.isdir(save_path + 'gen_model_output/csv'):
    os.mkdir(save_path + 'gen_model_output/csv')
if not os.path.isdir(save_path + 'gen_model_output/log'):
    os.mkdir(save_path + 'gen_model_output/log')
if not os.path.isdir(save_path + 'gen_model_output/json'):
    os.mkdir(save_path + 'gen_model_output/json')
if not os.path.isdir(csv_path):
    os.mkdir(csv_path)
if not os.path.isdir(save_path + 'all_generated_candidates/'):
    os.mkdir(save_path + 'all_generated_candidates/')

output_path = [save_path + 'gen_model_output/TL_reinvent.model_samples/',
               save_path + 'gen_model_output/staged_numCar.chkpt_samples/',
               save_path + 'gen_model_output/staged_molForm.chkpt_samples/',
               save_path + 'gen_model_output/staged_molFormMass.chkpt_samples/']



#Query data
if os.path.isfile(save_path+'generated_config_files/struct_info.csv'):
    with open(save_path+'generated_config_files/struct_info.csv') as csvfile:
        rows = csv.reader(csvfile)
        # structure_info = list(zip(*rows))
        structure_info = list(rows)
else:
    if evaluation:
        structure_info = preprocess_data_groundtruth(data,split,mol_dict,save_path)
    else:
        structure_info = preprocess_data(data,save_path)


#REINVENT4 commands
l_cmds = []
for row in structure_info:
    l_cmds.extend(write_tomls(row, save_path, REINVENT_PATH, hyperparameters))
num_cmds = 8
order_cmds = []
for j in range(0, num_cmds):
    for i in range(j, len(l_cmds), num_cmds):
        order_cmds.append(l_cmds[i])
with open(save_path+'REINVENT4_cmds.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows((item,) for item in order_cmds)

#Implement REINVENT4
run_cmds(save_path+'REINVENT4_cmds.csv', max_processes)


ik_spec_dict = {}
for idx, row in structure_info.iterrows():
    if row[1] not in ik_spec_dict.keys():
        ik_spec_dict[row[1]] = [row[0]]
    else:
        ik_spec_dict[row[1]].append(row[0])

filter_generated_strucures(structure_info, ik_spec_dict, save_path, output_path)

evaluate_generated_structures(structure_info, evaluation, save_path, csv_path, smi_path)