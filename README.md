

# An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph

This code provides the implementation of *RoBERTa-PFGCN* as described in out paper, a method to generate Graph of 
Program dubbed SVG with our novel Poacher Flow Edges. We use RoBERTa to generate embeddings and GCN for vulnerability detection and classification.


Graph construction            |  Graph neural networks with residual connection
:-------------------------:|:-------------------------:
![](https://github.com/pial08/SemVulDet/blob/main/figures/graph.png)  |  ![](https://github.com/pial08/SemVulDet/blob/main/figures/arch.png)


#### Requirements
- Python 	3.7
- Pytorch 	1.9 
- Transformer 	4.4
- torchmetrics 0.11.4
- tree-sitter 0.20.1
- sctokenizer 0.0.8

Moreover the above libraries can be installed by the commands from *requirements.txt* file. It is assumed that the installation will be done in a Linux system with a GPU. If GPU does not exist please remove the first command from the *requirements.txt*  file and replace it with 

`conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 -c pytorch` for OSX

or 


`conda install pytorch==1.9.0 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch` for Linux and Windows with no GPU.

Instructions to install libraries using *requirements.txt* file.

```shell
cd code 
pip install -r requirements.txt
```


### Usage
The repository is partially based on [CodeXGLUE](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection).


### Training and Evaluation
The following command should be used for training, testing and evaluation. Please set the ```--output_dir``` to the address where the model will be saved. We have also compiled a shell file with the same command for ease of use for the practitioners. Please put the location/address of train, evaluation and test file directory for the parameters
```--train_data_file```, ```--eval_data_file``` and ```--test_data_file```. 


Please run the following commands:

```shell
cd code

./run.sh

or,

python run.py --output_dir=<output_directory> --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=<training_data_directory> --eval_data_file=<eval_data_directory> --test_data_file=<test_data_directory> \
	--block_size 400 --train_batch_size 512 --eval_batch_size 512 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 512 --num_classes 2 --model_checkpoint <saved_directory> --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt

```

### Shell file parameters explaination
Here we explain some of the important parameters we used for our application. 

| Parameters | Default Values | Values | Description |
| :---:    | :---:   |:---:                | :---: |
| `--loss` | `focal` | *focal* or *weight* | Change parameters based on the usage of focal loss or weighted loss |
| `--graph`| `SVG`   | *SVG* or *AST* | Change parameters based on the graph generation method |
| `--alpha`| `0.1`   | 0-1 | The number should be a floating point |
| `--gamma`| `2.0`   | 0-INF | Floating value ranging from 0 to infinity. If the value is 0, effect os gamma is ignored |


### Datasets
- Please download our [VulF](https://drive.google.com/drive/folders/1d00kfEX6k1MhpxJtuFv5JqtlQTJfg03N?usp=sharing) dataset VulF directory.

- Our N-day and zero-day samples are also available in the previous link under *Testing* directory.
- After downloading VulF dataset, please put it under the directory *data*.

### Reproducibility
In order to use our pre-trained model, please download our model from [here](https://drive.google.com/drive/folders/1d00kfEX6k1MhpxJtuFv5JqtlQTJfg03N?usp=sharing) under the Saved Model directory. After downloading, please set the value of the parameter `--model_checkpoint` to local directory you saved the pre-trained model.



## License
As a free open-source implementation, our repository is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.
