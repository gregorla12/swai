import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import PrecisionRecallDisplay
import numpy as np
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
#from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline
import pandas as pd
from tqdm import tqdm
import logging
import torch
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from linevul_model import Model
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, PrecisionRecallDisplay

logger = logging.getLogger(__name__)
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label

def convert_examples_to_features(func, label, tokenizer):
    block_size=512
    # source
    code_tokens = tokenizer.tokenize(str(func))[:block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_type="train"):

        self.examples = []
        df=pd.DataFrame(os.listdir("llm-vul-main\llm-vul-main\VJBench-trans"),columns=['project'])
        df['processed_func']=df.apply(lambda x: open(os.path.join("llm-vul-main\llm-vul-main\VJBench-trans",x['project']+"\\"+x['project']+"_original_method.java"),'r').read(), axis=1)

        #labels = df["target"].tolist()
        train, test = train_test_split(df, test_size=0.2)
        if file_type == "train":
            df=train
        elif file_type == "test":
            df=test
        #df = pd.read_csv(file_path)
        funcs = df["processed_func"].tolist()
        for i in tqdm(range(len(funcs))):
            self.examples.append(convert_examples_to_features(funcs[i], 1, tokenizer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)

def evaluate(model, tokenizer, eval_dataset, eval_when_training=False):
    #build dataloader
    eval_batch_size=4
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=eval_batch_size,num_workers=0)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids, labels)=[x for x in batch]
        #(inputs_ids, labels)=[x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    logits = np.concatenate(logits,0)
    y_trues = np.concatenate(y_trues,0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:,1]>best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold":best_threshold,
    }

    PrecisionRecallDisplay.from_predictions(y_trues, logits[:, 1], name='LineVul')
    plt.savefig(f'eval_precision_recall_{"model_name"}.pdf')

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result






def train(train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_batch_size=16
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size, num_workers=0)
    learning_rate= 2e-5

    epochs= 10
    max_steps = epochs * len(train_dataloader)
    # evaluate the model per epoch
    save_steps = len(train_dataloader)
    adam_epsilon=1e-8

    #model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1,
                                                num_training_steps=max_steps)



    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", epochs)
    logger.info("  Total optimization steps = %d", max_steps)

    global_step=0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1=0

    model.zero_grad()

    for idx in range(epochs):
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            #(inputs_ids, labels) = [x.to(args.device) for x in batch]
            (inputs_ids, labels) = [x for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))

            if (step + 1) % 1 == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % save_steps == 0:
                    results = evaluate( model, tokenizer, eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join("output_dir", '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format("model_name"))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


if __name__ == "__main__":
    config = RobertaConfig.from_pretrained("./model")
    config.num_labels = 1
    config.num_attention_heads = 12
    tokenizer = RobertaTokenizer.from_pretrained("./model")
    model = RobertaForSequenceClassification.from_pretrained("./model", config=config, ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer)
    train_dataset = TextDataset(tokenizer,  file_type='train')
    eval_dataset = TextDataset(tokenizer,  file_type='eval')
    train(train_dataset, model, tokenizer, eval_dataset)


