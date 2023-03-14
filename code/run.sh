python run.py --output_dir=<output_directory> --model_type=roberta --tokenizer_name=microsoft/graphcodebert-base --model_name_or_path=microsoft/graphcodebert-base \
	--do_eval --do_test --do_train --train_data_file=<training_data_directory> --eval_data_file=<eval_data_directory> --test_data_file=<test_data_directory> \
	--block_size 400 --train_batch_size 512 --eval_batch_size 512 --max_grad_norm 1.0 --evaluate_during_training \
	--gnn ReGCN --learning_rate 5e-4 --epoch 100 --hidden_size 512 --num_classes 2 --model_checkpoint <saved_directory> --num_GNN_layers 2 --format uni --window_size 5 \
	--seed 123456 2>&1 | tee $logp/training_log.txt

