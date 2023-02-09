
# Discovering Language-neutral Sub-networks in Multilingual Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code for the paper: "[Discovering Language-neutral Sub-networks in Multilingual Language Models](https://arxiv.org/abs/2205.12672)" [EMNLP 2022]

The following figure shows an overview of our approach. We discover sub-networks in the original multilingual language model that are good foundations for learning various tasks and languages (a). Then, we investigate to what extent these sub-networks are similar by transferring them across other task-language pairs (b). In this example, the **blue** and **red** lines show sub-networks found for French and Urdu, respectively, and **purple** connections are shared in both sub-networks. Dashed lines show the weights that are removed in the pruning phase.

![alt text](./figs/model_overview.png?raw=true)


## Requirements
We recommend using a conda environment for running the scripts.
You can run the following commands to create the conda environment (assuming CUDA11.3):
```bash
conda create -n mbert_tickets python=3.6.10
conda activate mbert_tickets
pip install -r requirements.txt
conda install faiss-cpu -c pytorch
```


## Usage
In all the following experiments, you can either pass the arguments directly to the python script or specify the arguments in a `JSON` file and pass the file path to the script. You can find examples of such a `JSON` file in the configs folder. 

### Extract Sub-networks

#### MLM task:
To prepare the data for each language you need to perform the following steps:

1. Download the [wikipedia dump](https://dumps.wikimedia.org/) of the language, e.g. https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2 for English.
2. Install the [Wikiextractor](https://github.com/attardi/wikiextractor) to extract and clean the Wikipedia data.
3. Process and clean the data using the `Wikiextractor` script and generate a `JSON` file.

	```python -m wikiextractor.WikiExtractor <Wikipedia dump file (xml file)> --output en.data.json --bytes 100G --json --quiet```
4. Divide the data (`en.data.json`) into train and validation partitions and create `en.train.json` and `en.valid.json`. 
Alternatively, you can use the whole data as the train partition, and specify the validation ratio by `val_data_ratio` argument.

```shell
python LT_pretrain.py \
	--output_dir LT_pretrain_model \
	--model_type mbert \
	--model_name_or_path bert-base-multilingual-cased \
	--train_file pretrain_data/en.train.json \
	--do_train \
	--validation_file pretrain_data/en.valid.json \
	--do_eval \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--evaluate_during_training \
	--num_train_epochs 1 \
	--logging_steps 10000 \
	--save_steps 10000 \
	--mlm \
	--overwrite_output_dir \
	--seed 57
```

#### NER task:

```shell
python LT_ner.py \
	--output_dir tmp/ner \
	--data_language fr \
	--task_name ner \
	--dataset_name wikiann \
	--model_name_or_path bert-base-multilingual-cased \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 30 \
	--overwrite_output_dir \
	--evaluate_during_training \
	--logging_steps 1875 \
	--save_steps 1875 \
	--pad_to_max_length \
	--seed 57
```

#### XNLI task:
To run experiments for the mBERT model use `LT_xnli.py`, and to run experiments for the mT5 model use `LT_mt5_xnli.py`.

```shell
python LT_xnli.py \
	--output_dir tmp/xnli \
	--data_language fr \
	--task_name xnli \
	--dataset_name xnli \
	--model_name_or_path bert-base-multilingual-cased \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--learning_rate 5e-5 \
	--num_train_epochs 30  \
	--overwrite_output_dir \
	--evaluate_during_training \
	--logging_steps 36813 \
	--save_steps 36813 \
	--pad_to_max_length \
	--seed 57
```


### Fine-tuning Sub-networks

#### MLM task:
To prepare the data for the MLM task follow the steps mentioned [here](README.md#mlm-task).

```shell
python pretrain_trans.py \
	--mask_dir tmp/dif_mask/pretrain_mask.pt \
	--output_dir tmp/fine-tuning/pre \
	--model_type mbert \
	--mlm \
	--model_name_or_path bert-base-multilingual-cased \
	--train_file pretrain_data/en.train.json \
	--do_train \
	--validation_file pretrain_data/en.valid.json \
	--do_eval \
	--per_device_train_batch_size 16 \
	--per_device_eval_batch_size 16 \
	--evaluate_during_training \
	--num_train_epochs 1 \
	--logging_steps 2000 \
	--save_steps 0 \
	--max_steps 20000 \
	--overwrite_output_dir \
	--seed 57 \
	--weight_init pre  #[using random weight or official pretrain weight]
```

#### NER task:

```shell
python ner_fine_tune.py \
	--mask_dir tmp/ner_mask/fr_mask.pt \
	--output_dir tmp/fine-tuning/ner/fr \
	--task_name ner \
	--dataset_name wikiann \
	--data_language fr \
	--model_name_or_path bert-base-multilingual-cased \
	--do_train \
	--do_eval \
	--max_seq_length 128 \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--learning_rate 2e-5 \
	--num_train_epochs 3 \
	--overwrite_output_dir \
	--evaluate_during_training \
	--pad_to_max_length \
	--logging_steps 1875 \
	--save_steps 0 \
	--seed 5 \
	--weight_init pre  #[using random weight or official pretrain weight]
```

#### XNLI task (mBERT model):

```shell
python xnli_fine_tune.py \
	--mask_dir tmp/xnli_mask/fr_mask.pt \
	--output_dir tmp/fine-tuning/xnli/fr \
	--model_name_or_path bert-base-multilingual-cased \
	--dataset_name xnli \
	--data_language fr \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--learning_rate 5e-5 \
	--num_train_epochs 3 \
	--max_seq_length 128 \
	--evaluate_during_training \
	--overwrite_output_dir \
	--logging_steps 1227 \
	--save_steps 0 \
	--seed 57 \
	--weight_init pre  #[using random weight or official pretrain weight]
```
#### XNLI task (mT5 model):
```shell
python mt5_xnli_fine_tune.py \
	--mask_dir tmp/xnli_mask/fr_mask.pt \
	--output_dir tmp/fine-tuning/xnli/fr \
	--model_name_or_path google/mt5-base \
	--dataset_name xnli \
	--data_language fr \
	--do_train \
	--do_eval \
	--per_device_train_batch_size 32 \
	--per_device_eval_batch_size 32 \
	--learning_rate 2e-4 \
	--num_train_epochs 3 \
	--max_seq_length 256 \
	--max_source_length 128 \
	--max_target_length 16 \
	--val_max_target_length 5 \
	--evaluate_during_training \
	--overwrite_output_dir \
	--pad_to_max_length \
	--logging_steps 1227 \
	--save_steps 0 \
	--seed 57 \
	--weight_init pre  #[using random weight or official pretrain weight]
```

## Citation

If you use this code for your research, please cite our paper:

``` bib
@article{foroutan2022discovering,
  title={Discovering Language-neutral Sub-networks in Multilingual Language Models},
  author={Foroutan, Negar and Banaei, Mohammadreza and Lebret, Remi and Bosselut, Antoine and Aberer, Karl},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  url={https://arxiv.org/pdf/2205.12672.pdf},
  year={2022}
}
```
