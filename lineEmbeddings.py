import json
import pandas as pd
import torch
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import random
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
import argparse
import logging
import math
#from cuml import RandomForestClassifier as cuRF
if __name__ == "__main__":
    # %%
    #set up logging
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
    logging.info("start")

    feature_length=15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("cuda: "+str(torch.cuda.is_available()))
    #tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    tokenizer = RobertaTokenizer.from_pretrained("./model")
    model = RobertaModel.from_pretrained("./model")
    #model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    logging.info("model and tokenizer loaded")
    def calculateFeatures(line, previousLines):
        code_tokens = tokenizer.tokenize(line)[:feature_length]
        source_tokens = [tokenizer.cls_token]
        for pLine in previousLines:
            source_tokens+=tokenizer.tokenize(pLine)[:feature_length]+[tokenizer.sep_token]
        source_tokens+=code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        context_embeddings=model(torch.tensor(source_ids).to(device)[None,:])[0]
        return context_embeddings.cpu().sum(dim=1)[0].detach().numpy()
    
    def functionToDF(data):
        return pd.DataFrame.from_records([(ind,line.strip(),line in data['flaw_line_no']) for ind, line in enumerate(data['code'].split('\n')) if line.strip() not in ['','{','}','};'] ], columns=['originalIndex', 'line', 'vulnerable'])
    previousLines=2
    def dataSetToDataFrameLineLevel(originalDataset):
        result=pd.DataFrame()
        for index,data in originalDataset.iterrows():
            newLine=functionToDF(data)
            for i in range(1,previousLines+1):
                newLine['prevousLine'+str(i)] = newLine['line'].shift(periods=i)
            newLine['prevous']=newLine[['prevousLine'+str(i) for i in range(1,previousLines+1)]].apply(lambda row: list(row.values.astype(str)), axis=1)
            newLine['features']=newLine.apply(lambda row:calculateFeatures(row['line'],row['prevous']),axis=1)
            result = pd.concat([result,newLine.drop(columns=['prevousLine'+str(i) for i in range(1,previousLines+1)]) ])
        return result
    logging.info("preparing bigvul dataset")
    df_bigVul=json.load(open("Big-Vul-dataset/data.json"))
    df_bigVul=pd.DataFrame(df_bigVul)
    df_bigVul.drop(columns=['bigvul_id'])#remove columns that are not needed
    logging.info("bigvul dataset loaded")

    df_vul=df_bigVul.loc[df_bigVul['vul']==1]
    df_not_vul=df_bigVul.loc[df_bigVul['vul']==0]

    for index in range(5,len(df_vul.index),5):
        logging.info("currently processing range to "+str(index))
        df_current=pd.concat([df_vul.loc[index-100:index], df_not_vul.loc[index-100:index]])
        dataSetToDataFrameLineLevel(df_current).to_pickle("LineEncodings"+str(index))
        #store = pd.HDFStore('LineEncodings'+str(index)+'.h5')
        #store['df'] = dataSetToDataFrameLineLevel(df_current)  # save it
        #store['df']  # load it  
        

    
    