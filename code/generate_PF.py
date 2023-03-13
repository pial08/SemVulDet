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

import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing



logger = logging.getLogger(__name__)


import sctokenizer

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

asst_operators = ["=", "+=", "-=", "*=", "/=", "<<=", ">>="]

def readData(file_path):
    with open(file_path) as f:
        for line in tqdm(f):
            #print(line)
            js=json.loads(line.strip())

            code = js["code"]
            #print(code)
            
            #f = open("temp_1.cpp", "w")
            #f.write(code)
            #f.close()
            
            #print(js["code"])
            
            newTokens = []
            tokenTypes = []
            tokens = sctokenizer.tokenize_file(filepath='temp_1.cpp', lang='cpp')

            #print("Length of tokens ", len(tokens))
            

            #print("Adj matrix", adj_matrix)

            for token in tokens:
                out = (str(token)[1:-1])
                tup = tuple(map(str, out.split(', ')))
                #print(tup[0])
                newTokens.append(tup[0])
                tokenTypes.append(tup[1])
                #break
            #print(newTokens)
            # Convert to string and add a break
            codeStr = " ".join(newTokens)
            #print(codeStr)
            
            return codeStr, tokenTypes
            break



def getNextToken(tokens, tokenType,  index):
    for i in range(index + 1, len(tokens)):
        #print(tokens[i], tokenType[i])
        if tokenType[i] == 'TokenType.IDENTIFIER' or tokenType[i] == 'TokenType.KEYWORD':
            return tokens[i], i



def getPrevToken(tokens, tokenType, index):
    #for i in range(0, index - 1):
    i = index - 1
    while i >= 0:
        if tokenType[i] == 'TokenType.IDENTIFIER' or tokenType[i] == 'TokenType.KEYWORD':
            return tokens[i], i
        i -= 1

def isFunctionDecl(tokens, tokenType, index):
    #print(tokens[index], tokens[index + 1])
    if tokenType[index] == 'TokenType.IDENTIFIER' and tokens[index + 1] == "(":
        prevToken, prevIndex = getPrevToken(tokens, tokenType, index)
        #if tokenType[prevIndex] == 'TokenType.KEYWORD':
        if tokens[prevIndex] in ["void", "float", "int", "doble", "char"]:
            return True
        else:
            return False
    return False

def isAPICall(tokens, tokenType, index):
    #print("....",tokens[index], tokens[index + 1])
    if tokenType[index] == 'TokenType.IDENTIFIER' and tokens[index + 1] == "(":
        prevToken, prevIndex = getPrevToken(tokens, tokenType, index)
        #print(prevToken, tokenType[prevIndex])
        if tokens[prevIndex] not in ["void", "float", "int", "doble", "char"]:
            return True
        else:
            return False
    return False

def getFuncParams(tokens, tokenType, index):
    i = index + 1
    params = []
    if tokens[i] == "(" and tokens[i + 1] == ")":
        return params
    while True:
        nextToken, nextIndex = getNextToken(tokens, tokenType, i)
        params.append((nextToken, nextIndex))
        i = nextIndex
        if tokens[i + 1] == ")":
            break
    return params


"""
code, tokenType = readData("../FQ_validating.jsonl")
print(code)
print(tokenType)
print("-----------------------------------------------------------------")
print("Next token ...", getNextToken(code.split(" "), tokenType, 9))
print("Previous token ...", getPrevToken(code.split(" "), tokenType, 1))
print("Is FunctionDecl", isFunctionDecl(code.split(" "), tokenType, 3))
#isAPICall(code.split(" "), tokenType, 33)
print("Is isAPICall", isAPICall(code.split(" "), tokenType, 33))    # 33, 40

print("Is getFuncParams", getFuncParams(code.split(" "), tokenType, 40))
"""

def generatePFEdges(tokens, tokenType):
    asst_operators = ["=", "+=", "-=", "*=", "/=", "<<=", ">>="]
    execute_apis = ["exec", "system"]
    rows = cols = len(tokens)
    adj_matrix = [[0]*cols]*rows

    scope = {}
    token_pair = []

    try:

        for i in range(len(tokens)):
            if tokens[i] in asst_operators:
                left, leftIndex = getPrevToken(tokens, tokenType, i)
                right, rightIndex = getNextToken(tokens, tokenType, i)

                adj_matrix[leftIndex][rightIndex] = 1
                token_pair.append((leftIndex, rightIndex))
            elif isAPICall(tokens, tokenType, i):
                params = getFuncParams(tokens, tokenType, i)
                for item in params:
                    adj_matrix[i][item[1]]
                    token_pair.append((i, item[1]))

            elif tokens[i] in execute_apis:
                params = getFuncParams(tokens, tokenType, i)
                for item in params:
                    adj_matrix[i][item[1]]
                    token_pair.append((i, item[1]))

            elif tokens[i] == "free":
                indexFree = tokens.index("free")
                adj_matrix[i][indexFree] = 1
                token_pair.append((i, indexFree))
    except:
        pass
    return token_pair #,adj_matrix

        






"""
newTokens = []
tokenTypes = []
tokens = sctokenizer.tokenize_file(filepath='temp.cpp', lang='cpp')
for token in tokens:
    out = (str(token)[1:-1])
    tup = tuple(map(str, out.split(', ')))
    #print(tup[0])
    newTokens.append(tup[0])
    tokenTypes.append(tup[1])
    #break

print(newTokens)
print("------------------------------------------------")
print(tokenTypes)
"""
