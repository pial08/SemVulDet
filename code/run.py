# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import *

from sklearn.model_selection import train_test_split
from torchmetrics import F1Score, Recall, Precision, ConfusionMatrix, Accuracy
from torchmetrics.classification import MulticlassAccuracy

from tree_sitter import Language, Parser
import sctokenizer


import sys
print(sys.getrecursionlimit())
sys.setrecursionlimit(3000)


CPP_LANGUAGE = Language('../build/my-languages.so', 'cpp')

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class InputFeatures(object):
    def __init__(self,
                 input_tokens,
                 input_ids,
                 ast,
                 #idx,
                 label,
                 #tokenTypes

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.ast = ast
        #self.idx=str(idx)
        self.label=label
        #self.tokenTypes = tokenTypes




parser = Parser()
parser.set_language(CPP_LANGUAGE)
count_ast_nodes = 0
def traverse(root, lines, ast_list, word_pair, tokenizer):
    global count_ast_nodes
    count_ast_nodes += 1
    
    if len(root.children) == 0:
        line = lines[root.start_point[0]]
        
        token = line[root.start_point[1] : root.end_point[1]]
        ast_list.append(token.split("_")[0]) 
        return
    ast_list.append(root.type.split("_")[0])

    for child in root.children:
        word_p_id = root.type
        if len(child.children) == 0:
            line = lines[child.start_point[0]]
            word_q_id = line[child.start_point[1] : child.end_point[1]].split("_")[0]
        
        else:
            word_q_id = child.type.split("_")[0]

        x = word_p_id.split("_")[0] + " " + word_q_id.split("_")[0]
        
        word_pair_key = tokenizer.tokenize(x)
        word_pair_key = tokenizer.convert_tokens_to_ids(word_pair_key)
        word_pair_key = [word_pair_key[0], word_pair_key[-1]]
        word_pair += word_pair_key
        traverse(child, lines, ast_list, word_pair, tokenizer)


def convert_examples_to_features(js,tokenizer,args):
    
    code=' '.join(js['code'].split())

    """
    Tokenizing Properly
    """

    f = open("temp_1.cpp", "w")
    f.write(code)
    f.close()
    #print(js["code"])
    newTokens = []
    tokenTypes = []
    tokens = sctokenizer.tokenize_file(filepath='temp_1.cpp', lang='cpp')

    for token in tokens:
        out = (str(token)[1:-1])
        tup = tuple(map(str, out.split(', ')))
        #print(tup[0])
        newTokens.append(tup[0])
        tokenTypes.append(tup[1])
        #break
    #print(newTokens)
    # Convert to string and add a break
    code = " ".join(newTokens)
    tokenTypes = tokenTypes[:args.block_size]
    typePad_len = args.block_size - len(tokenTypes)
    tokenTypes += ['UNK'] * typePad_len

    


    """
    End
    """

    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length


    #print("Printing Tokentypes ...", tokenTypes)
    #print("Printing tokenTypes shape ...", len(tokenTypes))
    #print("Printing source tokens ...", type(source_tokens))
    #print("Printing source tokens id len ...", len(source_ids))
    
    return InputFeatures(source_tokens,source_ids,1, js['target'])

def convert_examples_to_features_ast(js,tokenizer,args):
    
    ast_list = []   # flattened ast
    word_pair = []  # graph structure stays
    global count_ast_nodes
    lines = js["code"].split("\n")
    tree = parser.parse(bytes(js["code"], "utf8"))

    
    traverse(tree.root_node, lines, ast_list, word_pair, tokenizer)
    count_ast_nodes = 0
    code = ' '.join(ast_list)
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]

    source_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length

    

    #resize word pair count
    padding_length = args.block_size - len(word_pair)
    if padding_length < 0:
        word_pair = word_pair[: args.block_size]
    elif padding_length > 0:
        word_pair += [0] * padding_length
        
    return InputFeatures(source_tokens, source_ids, word_pair, js['target'], [])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None, sample_percent=1.):
        self.examples = []
        print("Filepath --------------------------xxxxxxxxxxxxxxxx----------------------", file_path)
        
        parser = Parser()
        parser.set_language(CPP_LANGUAGE)

        with open(file_path) as f:
            for line in tqdm(f):
                js=json.loads(line.strip())
                
                
                if args.graph == "SVG":
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))
                elif args.graph == "AST":
                    self.examples.append(convert_examples_to_features_ast(js, tokenizer, args))

        total_len = len(self.examples)
        num_keep = int(sample_percent * total_len)



        if num_keep < total_len:
            np.random.seed(10)
            np.random.shuffle(self.examples)
            self.examples = self.examples[:num_keep]

        if 'train' in file_path:
            logger.info("*** Total ~~~~~~~~~~~~~ Sample ***")
            logger.info("\tTotal: {}\tselected: {}\tpercent: {}\t".format(total_len, num_keep, sample_percent))
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Sample ***")
                    logger.info("Total sample".format(idx))
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        
        #return  torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label) #self.examples[i].ast )
        #print("Inside getitem ...", len(self.examples[i].input_ids))
        #print("Inside getitem ...", len(self.examples[i].tokenTypes))
        return  torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label),  torch.tensor(self.examples[i].ast)#, torch.tensor(self.examples[i].tokenTypes)  

    

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ Train the model """ 
    print("Printing the model... ....", model)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size,num_workers=4,pin_memory=True)
    args.max_steps=args.epoch*len( train_dataloader)
    args.save_steps=len( train_dataloader)
    args.warmup_steps=len( train_dataloader)
    args.logging_steps=len( train_dataloader)
    args.num_train_epochs=args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_mrr=0.0
    best_acc=0.0
    model.zero_grad()
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        tr_num=0
        train_loss=0
        #print("CK-1")
        
        for step, batch in enumerate(train_dataloader):
            #print("CK-2")
            inputs = batch[0][:400].to(args.device)
            labels=batch[1][:400].to(args.device)
            ast = batch[2].to(args.device)

            model.train()
            loss,logits = model(ast, inputs,labels)



            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
            avg_loss=round(train_loss/tr_num,5)

            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb=global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, eval_dataset, preds = evaluate(args, model, tokenizer,eval_when_training=True)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        
                    if results['eval_acc']>best_acc:
                        best_acc=results['eval_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best acc:%s",round(best_acc,4))
                        logger.info("  "+"*"*20)                          
                        
                        with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
                            counter = 1
                            for example,pred in zip(eval_dataset.examples,preds):
                                if pred:
                                    f.write(str(counter) + '\t1\n')
                                    #pass
                                else:
                                    f.write(str(counter) + '\t0\n')
                                    #pass
                                counter += 1

                        checkpoint_prefix = 'checkpoint-best-acc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
        avg_loss = round(train_loss / tr_num, 5)
        logger.info("epoch {} loss {}".format(idx, avg_loss))
                        

def evaluate(args, model, tokenizer,eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args,args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4,pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[] 
    labels=[]
    for batch in eval_dataloader:
        
        inputs = batch[0][:400].to(args.device)
        label=batch[1][:400].to(args.device)

        # Adding AST structure 
        ast = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss,logit = model(ast, inputs,label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits=np.concatenate(logits,0)
    logits = torch.tensor(logits)

    np.set_printoptions(threshold=sys.maxsize)
     
    labels=np.concatenate(labels,0)
    print("Printing logits ... ", logits.shape)
    #preds=logits[:,0]>0.25
    max_elements, preds = torch.max(logits, dim=1)
    print("Printing preds... ", preds.shape)
    print("Printing labels ... ", labels.shape)
    eval_acc=np.mean(labels==preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    
    # Calculate F1 Precision and Recall Score
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)


    accuracy = Accuracy(task="multiclass", average='macro', num_classes=args.num_classes, subset_accuracy=True)
    f1 = F1Score(task="multiclass", average='macro', num_classes=args.num_classes)
    recall = Recall(task="multiclass", average='macro', num_classes=args.num_classes)
    precision = Precision(task="multiclass", average='macro', num_classes=args.num_classes)
    confmat = ConfusionMatrix(task="multiclass", num_classes=args.num_classes)

    
    
    conf_matrix = confmat(preds, labels)
    f1_score = f1(preds, labels)
    re = recall(preds, labels)
    pre = precision(preds, labels)
    acc = accuracy(preds, labels)

    print("Accuracy F1 Precision and Recall ",acc, f1_score, pre, re)
    print("Printing Confusion Matrix", conf_matrix)
    #print("Printing Confusion Matrix", conf_matrix.cpu().detach().numpy())

    mca = MulticlassAccuracy(num_classes=args.num_classes, average=None)
    print("Accuracy Per Class ... ",mca(preds, labels))
    print("---------   X    ----------")
    
    """
    np_matrix = np.array(conf_matrix)
    accuracy_per_class = np.max(np_matrix, axis=1) /np.sum(np_matrix, axis=1) #* 100
    print("Accuracy per class", accuracy_per_class)
    """

    

    
    result = {
        "eval_loss": float(perplexity),
        "eval_acc":round(eval_acc,4),
    }
    return result, eval_dataset, preds

def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]   
    labels=[]
    # for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
    for batch in eval_dataloader:
        #inputs = batch[0].to(args.device)
        #label=batch[1].to(args.device) 

        inputs = batch[0][:400].to(args.device)
        label=batch[1][:400].to(args.device)

        # Adding AST structure 
        ast = batch[2].to(args.device)

        # send ast info also

        with torch.no_grad():
            logit = model(ast, inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits=np.concatenate(logits,0)
    labels=np.concatenate(labels,0)
    preds=logits[:,0]>0.5

    test_acc=np.mean(labels==preds)

    # Calculate F1 Precision and Recall Score
    labels = torch.tensor(labels)
    preds = torch.tensor(preds)
    
    f1 = F1Score(average='macro', num_classes=args.num_classes)
    recall = Recall(average='macro', num_classes=args.num_classes)
    precision = Precision(average='macro', num_classes=args.num_classes)
    
    f1_score = f1(preds, labels)
    re = recall(preds, labels)
    pre = precision(preds, labels)
    print("F1 Precision and Recall ", f1_score, pre, re)


    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,preds):
            if pred:
                #f.write(example.idx+'\t1\n')
                pass
            else:
               # f.write(example.idx+'\t0\n')
               pass

    result = {
        "test_acc": round(test_acc, 4),
    }
    return result
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default="../dataset/train.jsonl", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default="./saved_models", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default="../dataset/valid.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../dataset/test.jsonl", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")


    parser.add_argument("--model", default="GNNs", type=str,help="")
    parser.add_argument("--hidden_size", default=256, type=int,
                        help="hidden size.")
    parser.add_argument("--feature_dim_size", default=768, type=int,
                        help="feature dim size.")
    parser.add_argument("--num_GNN_layers", default=2, type=int,
                        help="num GNN layers.")
    parser.add_argument("--num_classes", default=2, type=int,
                        help="num classes.")
    parser.add_argument("--gnn", default="ReGCN", type=str, help="ReGCN or ReGGNN")
    
    parser.add_argument("--loss", default="focal", type=str, help="focal or weight")
    parser.add_argument("--alpha", default=0.1, type=float, help="value of alpha should between 0 to 1")
    parser.add_argument("--gamma", default=2.0, type=float, help="To ignore the effect of gamma set the value to 0")
    parser.add_argument("--graph", default="SVG", type=str, help="Use SVG or AST")

    parser.add_argument("--format", default="uni", type=str, help="idx for index-focused method, uni for unique token-focused method")
    parser.add_argument("--window_size", default=3, type=int, help="window_size to build graph")
    parser.add_argument("--remove_residual", default=False, action='store_true', help="remove_residual")
    parser.add_argument("--att_op", default='mul', type=str,
                        help="using attention operation for attention: mul, sum, concat")
    parser.add_argument("--training_percent", default=1., type=float, help="percet of training sample")
    parser.add_argument("--alpha_weight", default=1., type=float, help="percet of training sample")


    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size=args.train_batch_size//max(args.n_gpu,1)
    args.per_gpu_eval_batch_size=args.eval_batch_size//max(args.n_gpu,1)
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)



    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    print("----------------Checkpoint last ---------------", checkpoint_last)
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)    
    else:
        model = model_class(config)

    
    if args.model == "devign":
        model = DevignModel(model, config, tokenizer, args)
    else: #GNNs
        model = GNNReGVD(model, config, tokenizer, args)

    print(model)
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file, args.training_percent)
        #print("Printing Train Dataset ... ", train_dataset.__getitem__(0))
        if args.local_rank == 0:
            torch.distributed.barrier()

        train(args, train_dataset, model, tokenizer)



    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))      
            model.to(args.device)
            result=evaluate(args, model, tokenizer)
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(round(result[key],4)))
            
    if args.do_test and args.local_rank in [-1, 0]:
            checkpoint_prefix = 'checkpoint-best-acc/model.bin'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
            model.load_state_dict(torch.load(output_dir))                  
            model.to(args.device)
            test_result = test(args, model, tokenizer)

            logger.info("***** Test results *****")
            for key in sorted(test_result.keys()):
                logger.info("  %s = %s", key, str(round(test_result[key],4)))

    return results


if __name__ == "__main__":
    main()


