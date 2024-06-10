# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% jupyter={"is_executing": true}
import json
import pandas as pd
import torch
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

# %% [markdown]
# # preprocessing functions for both datasets

# %%
if __name__ == "__main__":
    # %%
    #set up logging
    logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)
    logging.info("Preparing arguments")
    # %%
    #this does the argument parsing
    parser = argparse.ArgumentParser()
    #these are the number of functions used from the datasets. 0 for none at all and -1 for unlimited
    parser.add_argument("-i", "--ilmvul",nargs='?', const=-1, type=int,default=0)
    parser.add_argument("-b", "--bigvul",nargs='?', const=-1, type=int,default=0) 
    parser.add_argument("-l",'--level',nargs='?', const=-1, type=int,default=0)#0 for line level, 1 for function level
    args = parser.parse_args()
    ilmvul=args.ilmvul
    bigvul=args.bigvul
# %%
#for testing in jupyter notebooks 
#ilmvul=10
#bigvul=10


# %% [markdown]
# This function calculates the features. 
    # %%
    feature_length=15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("cuda: "+str(torch.cuda.is_available()))
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    #tokenizer = RobertaTokenizer.from_pretrained("./model")
    #model = RobertaModel.from_pretrained("./model")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    logging.info("model and tokenizer loaded")
    def calculateFeatures(line, previousLines):
        code_tokens = tokenizer.tokenize(line)[:feature_length]
        source_tokens = [tokenizer.cls_token]
        for pLine in previousLines:
            source_tokens+=tokenizer.tokenize(pLine)[:feature_length]+[tokenizer.sep_token]
        source_tokens+=code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        context_embeddings=model(torch.tensor(source_ids)[None,:])[0]
        return context_embeddings.sum(dim=1)[0].detach().numpy()


# %% [markdown]
#     # This takes a function with metadata as it is extracted from either the big vul dataset or the ILm vul dataset (the columns code and flaw_line_no are used) and outputs it as a dataframe with the columns 'originalIndex', 'line', 'vulnerable'. Empty lines and brackets only lines are removed. 

    # %%
    def functionToDF(data):
        return pd.DataFrame.from_records([(ind,line.strip(),line in data['flaw_line_no']) for ind, line in enumerate(data['code'].split('\n')) if line.strip() not in ['','{','}','};'] ], columns=['originalIndex', 'line', 'vulnerable'])


# %% [markdown]
#     # This function prepares the original dataset with entire methods stored in one line into a dataframe with individual lines in of code per line. Additionally, it is possible to define the size of the context used as the number of previous lines. These are stored in a list in reverse order. 

    # %%
    previousLines=2
    def dataSetToDataFrameLineLevel(originalDataset):
        result=pd.DataFrame()
        for index,data in originalDataset.iterrows():
            newLine=functionToDF(data)
            for i in range(1,previousLines+1):
                newLine['prevousLine'+str(i)] = newLine['line'].shift(periods=i)
            newLine['prevous']=newLine[['prevousLine'+str(i) for i in range(1,previousLines+1)]].apply(lambda row: list(row.values.astype(str)), axis=1)
            newLine['features']=newLine.apply(lambda row:calculateFeatures(row['line'].toDevice(device),row['prevous'].toDevice(device)),axis=1)
            result = pd.concat([result,newLine.drop(columns=['prevousLine'+str(i) for i in range(1,previousLines+1)]) ])
        return result

    # %%
    feature_length_function=150
    def calculateFeaturesFunction(function):
        code_tokens = tokenizer.tokenize(function)[:feature_length_function]
        source_tokens = [tokenizer.cls_token]
        source_tokens+=code_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        context_embeddings=model(torch.tensor(source_ids)[None,:])[0]
        return context_embeddings.sum(dim=1)[0].detach().numpy()


    # %%
    def dataSetToDataFrameFunctionLevel(originalDataset):
        originalDataset['features']=originalDataset.apply(lambda x:calculateFeaturesFunction(x['code'].toDevice(device)),axis=1)
        originalDataset['vulnerable']=originalDataset.apply(lambda x: x['vul'] if  not math.isnan(x['vul'].toDevice(device)) else 1,axis=1).astype('bool') #always define the Ilm dataset as vulnerable as it contains only vulnerable functions
        return originalDataset

# %% [markdown]
#     # # Big-Vul dataset

    # %%
    #only execute below if we want to use this dataset
    if bigvul!=0:
        # %%
        logging.info("preparing bigvul dataset")
        df_bigVul=json.load(open("Big-Vul-dataset/data.json"))
        if bigvul!=-1:
            df_bigVul=random.sample(df_bigVul, bigvul)#reduce dataset size, for testing only!
            logging.info("bigvul dataset randomly sampled")
        df_bigVul=pd.DataFrame(df_bigVul)
        df_bigVul.drop(columns=['bigvul_id'])#remove columns that are not needed
        logging.info("bigvul dataset loaded")

# %% [markdown]
#     # # Ilm-vul dataset

# %% [markdown]
#     # This code adds the code of the original method. The dataset has additional files with partial transformations. Maybe, they are more useful for us, feel free to modify this.
    # %%
    if ilmvul!=0:
        # %%
        logging.info("preparing Ilm dataset")
        df_Ilm=pd.DataFrame(os.listdir("llm-vul-main\llm-vul-main\VJBench-trans"),columns=['project'])
        df_Ilm['code']=df_Ilm.apply(lambda x: open(os.path.join("llm-vul-main\llm-vul-main\VJBench-trans",x['project']+"\\"+x['project']+"_original_method.java"),'r').read(), axis=1)

# %% [markdown]
#         # This adds the location of the bug in the original method. It is a list with 2 elements (start and end line); usually the same.

        # %%
        df_Ilm['location_original_method']=df_Ilm.apply(lambda x:json.load(open(os.path.join("llm-vul-main\llm-vul-main\VJBench-trans",x['project']+"\\"+"buggyline_location.json")))['original'],axis=1)

        # %%
        df_Ilm['flaw_line_no']=df_Ilm.apply(lambda row:list(range(row['location_original_method'][0][0],row['location_original_method'][0][1]+1)),axis=1)  #convert the beginning and end location to a list containing all vulnerable lines. Assumption: there is only one vulnerable location.
        df_Ilm=df_Ilm.drop(columns=['location_original_method','project' ])#delete the column with start and end of the vulnerable location as this is no longer needed as well as the project column
        if ilmvul!=-1:
            df_Ilm=df_Ilm.sample(ilmvul)#reduce dataset size, for testing only!
            logging.info("Ilm dataset randomly sampled")
        df_Ilm #nice output for notebook-people
        logging.info("Ilm dataset loaded")

# %% [markdown]
#     # # data preprocessing for both datasets

# %% [markdown]
#     # combine both datasets into one as they should have the same structure now. This dataset now contains rows with the entire functions and the line location of the vulnerable line(s). Please note that the data format for flaw_line_no is slightly different.

    # %%
    if bigvul!=0 and ilmvul!=0:
        #combine both datasets
        # %%
        df_complete=pd.concat([df_bigVul,df_Ilm])
    # %%
    if bigvul!=0 and ilmvul==0: 
        #use only bigvul
        # %%
        df_complete=df_bigVul
    # %%
    if bigvul==0 and ilmvul!=0: 
        # %%
        #use only ilmvul
        df_complete=df_Ilm    
    # %%
    if bigvul==0 and ilmvul==0:
        # what to do
        logging.critical("no dataset selected. Use -i or -b options")     

# %% [markdown]
#     # Split the dataset into training and testing based on functions, so that all lines from a function are either entirely test or training

    # %%
    logging.info("preparing data for training")
    train, test = train_test_split(df_complete, test_size=0.2, random_state=42)

    # %%
    if args.level==0:
        # %%
        logging.info("using line level classification")
        train=dataSetToDataFrameLineLevel(train)
        test=dataSetToDataFrameLineLevel(test)
    # %%
    if args.level==1:
        # %%
        logging.info("using function level classification")
        train=dataSetToDataFrameFunctionLevel(train)
        test=dataSetToDataFrameFunctionLevel(test)


# %% [markdown]
#     # # classification

    # %%
    logging.info("training random forest started")
    rf = RandomForestClassifier(max_depth=50, n_estimators=15, max_features=5, random_state=42)
    rf.fit(list(train['features']),list(train['vulnerable']))
    score=rf.score(list(test['features']),list(test['vulnerable'])) 
    logging.info("score: "+str(score))




# %% [markdown]
#     # # k-fold cross validation

    # %%
    logging.info("k-fold validation")
    train_test=dataSetToDataFrameLineLevel(df_complete)

    # %%
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    rf = RandomForestClassifier(max_depth=50, n_estimators=15, max_features=5, random_state=42)
    scores = cross_val_score(rf, train_test['features'], train_test['vulnerable'], scoring='neg_mean_squared_error',
                             cv=cv, n_jobs=-1)

    # %%
    logging.info("k-fold score: "+str(sqrt(mean(absolute(scores)))))

