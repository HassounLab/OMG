# Optimized de novo molecular generation (OMG) for mass spectra annotation using transfer and reinforcement learning

This repository contains code to implement the Optimized Molecular Generation (**OMG**) [[1]](#1). OMG performs de novo molecular generation for mass spectra annotation, by dividing this task into two steps: molecular generation and candidate ranking.

## Installation and Requirements

OMG implements previous tools, REINVENT4 (https://github.com/MolecularAI/REINVENT4) [[2]](#2) and JESTR (https://github.com/HassounLab/JESTR1/) [[3]](#3). To use OMG, these tools need to be downloaded and included under the "OMG-main/" directory. Then, comp_custom_OMGformula.py and comp_custom_OMGmass.py from the OMG respository must be placed in the directory: "REINVENT4-main/reinvent_plugins/components/". Lastly, both conda environments need to be created as described in those repositories and named "reinvent4" and "jestr" respectively.

## Usage

All input information required may be specified in runOMG.sh. Once the code and conda environments are structured as specified in the Installation and Requirements section above, run the runOMG.sh file as it is to evaluate the CANOPUS dataset. Note that no conda environment needs to be activated. Otherwise, follow the instructions in the following sections to run OMG for other uses.

## Perform evaluation of OMG using the CANOPUS and MassSpecGym datasets

All data necessary to run the evaluation of OMG described in our paper is included in the JESTR repository and the data folder here. All variables in runOMG.sh currently point to the CANOPUS dataset. As a result, running the code as is will generate and rank candidates for the CANOPUS test set. To change to evaluating the MassSpecGym test set, specify --data_path to be '../JESTR1-main/data/MassSpecGym/' and then run the runOMG.sh file.

## Perform OMG using another dataset and PubChem-derived TL candidates

To apply OMG to another dataset of queries, include a dictionary of spectra queries called "data_dict.pkl" and specify it's location using --data_path. The dictionary must be organized in the following way. Each item in the dictionary represents one spectrum. The key is the ID and the value is another dictionary with the keys: 'Formula', 'PrecursorMZ', 'Precursor', and 'ms'. Also, --evaluation must be set to False and hyperparameters for finetuning may be altered in step1.py

## References 
<a id="1">[1]</a> 
Martin, M.R. & Hassoun, S. (2025). Optimized de novo molecular generation (OMG) for mass spectra annotation using transfer and reinforcement learning. In preparation.

<a id="2">[2]</a> 
Loeffler, H. H., He, J., Tibo, A., Janet, J. P., Voronov, A., Mervin, L. H., & Engkvist, O. (2024). Reinvent 4: Modern AI–driven generative molecule design. Journal of Cheminformatics, 16(1), 20.

<a id="3">[3]</a> 
Kalia, A., Krishnan, D., & Hassoun, S. (2024). Jestr: Joint embedding space technique for ranking candidate molecules for the annotation of untargeted metabolomics data. arXiv preprint arXiv:2411.14464.