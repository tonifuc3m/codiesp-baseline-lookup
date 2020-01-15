#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 16:39:02 2020

@author: antonio
"""


#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '.')

import re
import pandas as pd
import itertools
import os
import time
import argparse
from utils import (format_ann_info, format_text_info, tokenize_span, normalize_tokens,
                   normalize_annot, eliminate_contained_annots)


def get_codes(df_annot, original_annot):
    return df_annot[df_annot["span"] == original_annot].drop_duplicates(subset=['code'])["code"].to_list()

def store_prediction(pos_matrix, predictions, off0, off1, original_label, 
                     df_annot, original_annot, txt):
                                        
    # 1. Eliminate old annotations if the new one contains them
    (pos_matrix, 
     predictions) = eliminate_contained_annots(pos_matrix, predictions, off0, off1)
    
    # 2. STORE NEW PREDICTION
    codes = get_codes(df_annot, original_annot)
    for code in codes:
        predictions.append([txt[off0:off1], off0, off1, original_label, code])   
        pos_matrix.append([off0, off1])
        
    return predictions, pos_matrix

def check_surroundings(txt, span, original_annot, n_chars, n_words, original_label,
                       predictions, pos_matrix):
    # DESCRIPTION: explore the surroundings of the match.
    #           Do not care about extra whitespaces or punctuation signs in 
    #           the middle of the annotation.
    
    ## 1. Get normalized surroundings ##
    large_span = txt[max(0, span[0]-n_chars):min(span[1]+n_chars, len(txt))]

    # remove half-catched words
    first_space = re.search('( |\n)', large_span).span()[1]
    last_space = (len(large_span) - re.search('( |\n)', large_span[::-1]).span()[0])
    large_span_reg = large_span[first_space:last_space]
    
    # Tokenize text span 
    token_span2id, id2token_span_pos, token_spans = tokenize_span(large_span_reg,
                                                                  n_words)
    # Normalize
    original_annotation_processed = normalize_annot(original_annot)
    token_span_processed2token_span = normalize_tokens(token_spans, min_upper)
    
    ## 2. Match ##
    try:
        res = token_span_processed2token_span[original_annotation_processed]
        id_ = token_span2id[res]
        pos = id2token_span_pos[id_]
        off0 = (pos[0] + first_space + max(0, span[0]-n_chars))
        off1 = (pos[1] + first_space + max(0, span[0]-n_chars))
        
        # Check new annotation is not contained in a previously stored new annotation
        if not any([(item[0]<=off0) & (off1<= item[1]) for item in pos_matrix]):
            # STORE PREDICTION and eliminate old predictions contained in the new one.
            predictions, pos_matrix = store_prediction(pos_matrix, predictions,
                                                       off0, off1, 
                                                       original_label, df_annot,
                                                       original_annot, txt)
    except: 
        pass
    
    return predictions, pos_matrix

def find_predictions(datapath, min_upper, annot2label, annot2annot_processed, 
                         annotations_final, df_annot):
    start = time.time()
    
    predictions_dict = {}
    c = 0
    for root, dirs, files in os.walk(datapath):
        for filename in files:
            print(filename)
            
            #### 0. Initialize, etc. ####
            predictions = []
            pos_matrix = []
             
            #### 1. Get text ####
            txt = open(os.path.join(root,filename)).read()
    
            #### 2. Format text information ####
            words_final, words_processed2pos = format_text_info(txt, min_upper)
            
            #### 3. Intersection ####
            # Generate candidates
            words_in_annots = words_final.intersection(annotations_final)            
             
            #### 4. For every token of the intersection, get all original 
            #### annotations associated to it and all matches in text.
            #### Then, check surroundings of all those matches to check if any
            #### of the original annotations is in the text ####
            # For every token
            for match in words_in_annots:
                
                # Get annotations where this token is present
                original_annotations = [k for k,v in annot2annot_processed.items() if match in v]
                # Get text locations where this token is present
                match_text_locations = words_processed2pos[match]

                # For every original annotation where this token is present:
                for original_annot in original_annotations:
                    original_label = annot2label[original_annot]
                    n_chars = len(original_annot)
                    n_words = len(original_annot.split())
                    
                    if len(original_annot.split()) > 1:
                        # For every match of the token in text, check its 
                        # surroundings and generate predictions
                        for span in match_text_locations:   
                            (predictions, 
                             pos_matrix) = check_surroundings(txt, span, 
                                                              original_annot,
                                                              n_chars, n_words,
                                                              original_label,
                                                              predictions,
                                                              pos_matrix)
                                                              
                    # If original_annotation is just the token, no need to 
                    # check the surroundings
                    elif len(original_annot.split()) == 1:
                        for span in match_text_locations:
                            # Check span is surrounded by spaces or punctuation signs &
                            # span is not contained in a previously stored prediction
                            if (((txt[span[0]-1].isalnum() == False) & 
                                 (txt[span[1]].isalnum()==False)) & 
                                (not any([(item[0]<=span[0]) & (span[1]<=item[1]) 
                                          for item in pos_matrix]))):
                                
                                # STORE PREDICTION and eliminate old predictions
                                # contained in the new one
                                (predictions, 
                                 pos_matrix) = store_prediction(pos_matrix, 
                                                                predictions,
                                                                span[0], span[1], 
                                                                original_label,
                                                                df_annot,
                                                                original_annot,
                                                                txt)
 
            #### 5. Remove duplicates ####
            predictions.sort()
            predictions_no_duplicates = [k for k,_ in itertools.groupby(predictions)]
                        
            # Final appends
            c = c + len(predictions_no_duplicates)
            predictions_dict[filename] = predictions_no_duplicates
                
    total_t = time.time() - start
    
    return total_t, predictions_dict, c


def parse_arguments():
    
    # DESCRIPTION: Parse command line arguments
    
    parser = argparse.ArgumentParser(description='process user given parameters')
    parser.add_argument("-gs", "--gs_path", required = True, dest = "gs_path", 
                        help = "path to GS file")
    parser.add_argument("-data", "--data_path", required = True, dest = "data_path", 
                        help = "path to text files")
    parser.add_argument("-out", "--out_path", required = True, dest = "out_path", 
                        help = "path to output predictions")
    parser.add_argument("-t", "--sub_track", required = True, dest = "sub_track", 
                        help = "sub_track number (1 or 2 for code prediction, 3 for Explainable AI")
    
    args = parser.parse_args()
    gs_path = args.gs_path
    data_path = args.data_path
    out_path = args.out_path
    sub_track = int(args.sub_track)
    
    return gs_path, data_path, out_path, sub_track

if __name__ == '__main__':
    ######## GET GS INFORMATION ########    
    # Get DataFrame
    '''sub_track=0
    path = '/home/antonio/Documents/Projects/Tasks/CodiEsp/data/final_dataset/'
    data_path = path + 'dev/text_files'
    
    gs_path = 'train/train_annotations_task2_processed.tsv'
    out_path = '/home/antonio/dev_task2.tsv'
    '''
    
    gs_path, data_path, out_path, sub_track = parse_arguments()
    
    if (sub_track == 1) | (sub_track == 2):
        df_annot = pd.read_csv(gs_path, sep='\t', 
                               names=['clinical_case', 'label', 'code', 'reference'])
    elif sub_track == 3: 
        print('We are on Explainable AI track')
        df_annot = pd.read_csv(gs_path, sep='\t', 
                               names=['clinical_case', 'label', 'code', 'reference', 'offset'])
        df_annot = df_annot.drop(['offset'], axis=1)
    else:
        raise ValueError('Incorrect sub-track value')
    filenames_gs = df_annot['clinical_case'].drop_duplicates()
    
    ######## FORMAT ANN INFORMATION #########
    print('\n\nExtracting original annotations...\n\n')
    min_upper = 3
    annot2label, annot2annot_processed, annotations_final = format_ann_info(df_annot,
                                                                            min_upper)
    
    
    ######## FIND MATCHES IN TEXT ########
    print('\n\nPredicting codes...\n\n')
    total_t, predictions_dict, c = find_predictions(data_path, min_upper, annot2label,
                                                    annot2annot_processed,
                                                    annotations_final, df_annot)
    print('Elapsed time: {}s'.format(round(total_t, 3)))
    print('Number of predicted codes: {}'.format(c))
    
    ######## FORMAT OUTPUT ########
    df = pd.DataFrame(columns =['ref', 'pos', 'label', 'code', 'clinical_case'])
    for filename in predictions_dict.keys():
        if predictions_dict[filename]:
            df_this = pd.DataFrame(predictions_dict[filename], 
                                   columns=['ref', 'pos0', 'pos1', 'label', 'code'])
            df_this['clinical_case'] = filename[:-4] # remove file extension .txt
            
            df_this['pos'] = df_this.apply(lambda x: str(x['pos0']) + ' ' + str(x['pos1']), axis=1)
            df_this = df_this.drop(['pos0', 'pos1'], axis=1)
            
            df = df.append(df_this)
    
    if (sub_track == 1) | (sub_track == 2):
        df_final = df[['clinical_case', 'code']]
    
    elif sub_track == 3: 
        df_final = df[['clinical_case', 'pos', 'label', 'code']]
    else:
        raise ValueError('Incorrect sub-track value')
        
    ######## SAVE OUTPUT ########
    df_final.to_csv(out_path, sep='\t', index=False, header=False)
    


