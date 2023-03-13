from math import gamma
from multiprocessing import reduction
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from modelGNN_updates import *
from utils import preprocess_features, preprocess_adj
from utils import *

import torchvision



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None):
        
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        logits=outputs
        prob=F.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

class PredictionClassification(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, args, input_size=None):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = args.hidden_size
        self.dense = nn.Linear(input_size, args.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, features):  #
        x = features
        x = self.dropout(x)
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class GNNReGVD(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.w_embeddings = self.encoder.roberta.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()
        self.tokenizer = tokenizer
        if args.gnn == "ReGGNN":
            self.gnn = ReGGNN(feature_dim_size=args.feature_dim_size,
                                hidden_size=args.hidden_size,
                                num_GNN_layers=args.num_GNN_layers,
                                dropout=config.hidden_dropout_prob,
                                residual=not args.remove_residual,
                                att_op=args.att_op)
        else:
            self.gnn = ReGCN(feature_dim_size=args.feature_dim_size,
                               hidden_size=args.hidden_size,
                               num_GNN_layers=args.num_GNN_layers,
                               dropout=config.hidden_dropout_prob,
                               residual=not args.remove_residual,
                               att_op=args.att_op)
        gnn_out_dim = self.gnn.out_dim
        self.classifier = PredictionClassification(config, args, input_size=gnn_out_dim)

    def forward(self, ast, input_ids=None, labels=None):
        # construct graph
        
        adj, x_feature = build_graph(input_ids.cpu().detach().numpy(), self.w_embeddings, self.tokenizer, window_size=self.args.window_size)
        #adj, x_feature = build_ast(ast.cpu().detach().numpy(), self.w_embeddings, self.tokenizer)
        
        # initilizatioin
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)

    
        # run over GNNs
        outputs = self.gnn(adj_feature.to(device).double(), adj.to(device).double(), adj_mask.to(device).double())
        class_weights = torch.FloatTensor([1.0, 10.0]).to(device) 
        logits = self.classifier(outputs)
        prob = F.sigmoid(logits)
        if labels is not None:
            
            if self.args.loss == "focal":
                ce = CrossEntropyLoss()
                loss1 = ce(logits, labels)
                
                # Focal Loss:
                y = torch.zeros(logits.shape[0], self.args.num_classes) 
                y[range(y.shape[0]), labels] = 1
                loss2 = torchvision.ops.sigmoid_focal_loss(logits, y.to(device), alpha=self.args.alpha, gamma=self.args.gamma, reduction="mean")
                loss = loss1 + loss2
                loss = loss.mean()
                
            else:
                ce = CrossEntropyLoss()
                loss1 = ce(logits, labels)          
                y = torch.zeros(logits.shape[0], self.args.num_classes) 
                y[range(y.shape[0]), labels]=1
                loss2 = torchvision.ops.sigmoid_focal_loss(logits, y.to(device), alpha=0.1, gamma=0.0, reduction="mean")
                loss = loss1 + loss2
                loss = loss.mean()

            return loss , prob
        else:
            return prob

