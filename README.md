

# An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph

This code provides the implementation of RoBERTa-PFGCN as described in out paper, a method to generate Graph of 
Program dubbed SVG with our novel Poacher Flow Edges. We use RoBERTa to generate embeddings and GCN for vulnerability detection and classification.


Graph construction            |  Graph neural networks with residual connection
:-------------------------:|:-------------------------:
![](https://github.com/pial08/SemVulDet/blob/main/graph.pdf)  |  ![](https://github.com/pial08/SemVulDet/blob/main/arch.pdf)


## Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).

### Training and Evaluation

```shell
cd code
python run.py --output_dir=./saved_models/regcn_l2_hs128_uni_ws5_lr5e4 --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=../dataset/train.jsonl --eval_data_file=../dataset/valid.jsonl --test_data_file=../dataset/test.jsonl \
	--block_size 400 --train_batch_size 128 --eval_batch_size 128 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-5 --epoch 20 --hidden_size 128 --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt
```

#### Requirements
- Python 	3.7
- Pytorch 	1.9 
- Transformer 	4.4




## License
As a free open-source implementation, ReGVD is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
