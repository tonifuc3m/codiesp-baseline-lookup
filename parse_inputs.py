#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 15:23:12 2020

@author: antonio
"""

import pandas as pd

def parse_tsv(gs_path, sub_track):
    if (sub_track == 1) | (sub_track == 2):
        df_annot = pd.read_csv(gs_path, sep='\t', 
                               names=['clinical_case', 'label', 'code', 'reference'])
    elif sub_track == 3: 
        print('\nWe are on Explainable AI track\n')
        df_annot = pd.read_csv(gs_path, sep='\t', 
                               names=['clinical_case', 'label', 'code', 'reference', 'offset'])
        df_annot = df_annot.drop(['offset'], axis=1)
    else:
        raise ValueError('Incorrect sub-track value')
        
    return df_annot