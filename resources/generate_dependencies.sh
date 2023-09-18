#!/usr/bin/env conda run -n threat-hunting-ia python


#
# This file generates the environment files: environment.yml and requirements.txt
#
# conda install -n threat-hunting-ia requirements.txt
# conda env create -n threat-hunting-ia --file environment.yml

conda list -e > requirements.txt
conda env export --file environment.yml
