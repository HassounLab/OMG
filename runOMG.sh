#!/bin/bash

spectra_data_path='../JESTR1-main/data/NPLIB1/'
OMG_data_path='../data/'
evaluation=True

reinvent_path='../REINVENT4/'
max_processes=1

conda run -n reinvent4 python ./src/step1.py --spectra_data_path "$spectra_data_path" --OMG_data_path "$OMG_data_path" --evaluation "$evaluation" --reinvent_path "$reinvent_path" --max_processes "$max_processes" >> out_OMG_step1.txt

conda run -n jestr python ./src/step2.py --data_path "$spectra_data_path" --OMG_data_path "$OMG_data_path" --evaluation "$evaluation" >> out_OMG_step2.txt
