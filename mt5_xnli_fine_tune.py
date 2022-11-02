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
""" Finetuning the library models for sequence classification on XNLI (mT5 model)."""

import json
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from helpers.pruning_utils import see_mt5_weight_rate, pruning_mt5_model_custom, pruning_mt5_model, random_pruning_mt5_model
from helpers.utils import set_seed
from datasets import load_dataset

from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AdamW,
    Adafactor,
    AutoConfig,
    MT5Tokenizer, 
    MT5ForConditionalGeneration,
    HfArgumentParser,
    default_data_collator,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    get_constant_schedule
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LTHTrainingArguments:
    """ Arguments for Lottery Ticket Hypothesis training. """
    
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    num_beams: int = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be "
                          "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."}
    )
    val_max_target_length: int = field(
        default=5,
        metadata={"help": "The maximum total sequence length for validation "
                          "target text after tokenization.Sequences longer than this will be truncated, sequences "
                          "shorter will be padded. Will default to `max_target_length`.This argument is also used "
                          "to override the  ``max_length`` param of ``model.generate``, which is used "
                          "during ``evaluate`` and ``predict``."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )
    max_source_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. "
                          "Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: int = field(
        default=16,
        metadata={"help": "The maximum total target sequence length after tokenization.Sequences longer than this will"
                          " be truncated, sequences shorter will be padded."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Output directory path."}
    )
    log_dir: str = field(
        default=None, metadata={"help": "Log directory path."}
    )
    mask_dir: str = field(
        default=None, metadata={"help": "LTH pretrained mask path."}
    )
    overwrite_output_dir: bool = field(
        default=True, metadata={"help": "Whether overwrite the output dir."}
    )
    data_language: str = field(
        default=None, metadata={"help": "Data language."}
    )
    pruning_type: str = field(
        default="lth", metadata={"help": "Pruning type (random, oneshot, lth)."}
    )
    save_steps: int = field(
        default=36813, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=36813, metadata={"help": "Log every X updates steps."}
    )
    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Whether to evaluate during training or not."}
    )
    weight_init: str = field(
        default="pre", metadata={"help": "Initial weights."}
    )
    rand_seed: bool = field(
        default=False,  metadata={"help": "Whether set a seed or not."}
    )
    do_train: bool = field(
        default=False, metadata={"help": "Whether to run training."}
    )
    do_eval: bool = field(
        default=True, metadata={"help": "Whether to run eval on the dev set."}
    )
    do_predict: bool = field(
        default=False, metadata={"help": "Whether to run predictions on the test set."}
    )
    per_device_train_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=32, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    optim: str = field(
        default="Adam", metadata={"help": "Optimizer type."}
    )
    scheduler: str = field(
        default="linear", metadata={"help": "Scheduler type."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=3.0, metadata={"help": "Total number of training epochs to perform."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    adam_beta1: float = field(
        default=0.9, metadata={"help": "Beta1 for AdamW optimizer"}
    )
    adam_beta2: float = field(
        default=0.999, metadata={"help": "Beta2 for AdamW optimizer"}
    )
    adam_epsilon: float = field(
        default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."}
    )
    local_rank: int = field(
        default=-1, metadata={"help": "For distributed training: local_rank"}
    )
    no_cuda: bool = field(
        default=False, metadata={"help": "Do not use CUDA even when it is available"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    model_name_or_path: str = field(
        default="google/mt5-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    task_name: Optional[str] = field(
        default="xnli",
        metadata={"help": "The name of the task (ner, pos...)."}
    )
    dataset_name: Optional[str] = field(
        default="xnli",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. If set, sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={"help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                          "See details at https://nvidia.github.io/apex/amp.html"}
    )
    server_ip: str = field(
        default="", metadata={ "help": "For distant debugging."}
    )
    server_port: str = field(
        default="", metadata={ "help": "For distant debugging."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


def train(args, train_dataset, eval_dataset, model, data_collator, tokenizer):
    """ Train the model """
    record_result = []

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  batch_size=args.train_batch_size,
                                  sampler=train_sampler)

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.optim == "Adam":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        if args.scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
            )
        elif args.scheduler == "constant":
            scheduler = get_constant_schedule(optimizer)
        
    elif args.optim == "Adafactor":
        print()
        optimizer_kwargs = {"lr": args.learning_rate}
        optimizer_kwargs.update({"scale_parameter": False, "relative_step": False})
        optimizer = Adafactor(model.parameters(), **optimizer_kwargs)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_device_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    # pruning_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            inputs = {t: batch[t].to(args.device) for t in batch}

            outputs = model(**inputs)
            loss = outputs.loss  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.detach().item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                    args.gradient_accumulation_steps >= len(epoch_iterator) == (step + 1)
            ):
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 or global_step == args.save_steps:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, eval_dataset, model, data_collator, tokenizer)
                        torch.cuda.empty_cache() 
                        record_result.append(results)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    zero_rate = see_mt5_weight_rate(model)
                    print(f'model zero rate: {zero_rate}')
                    tb_writer.add_scalar('zero rate', zero_rate, global_step)

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    print(json.dumps({**logs, **{"step": global_step}}))

                if (args.local_rank in [-1, 0] and args.save_steps > 0 and
                        global_step % args.save_steps == 0) or (global_step == t_total):
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
             
                    torch.save(model,os.path.join(output_dir, "model.pt"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            torch.cuda.empty_cache()
            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    torch.save(record_result, os.path.join(args.output_dir, "result.pt"))

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, data_collator, tokenizer, prefix=""):

    results = {}
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    nb_eval_steps = 0
    if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

    gen_kwargs = {
        "max_length": args.val_max_target_length,
        "num_beams": args.num_beams,
    }
    preds = []
    true_labels = []

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        with torch.no_grad():
            inputs = {t: batch[t].to(args.device) for t in batch}

            generated_tokens = model.module.generate(input_ids=inputs['input_ids'],
                                                     attention_mask=inputs['attention_mask'],
                                                     **gen_kwargs,).detach().cpu().numpy()

            labels = inputs["labels"].detach().cpu().numpy()
            if args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            preds += decoded_preds
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            true_labels += decoded_labels

        nb_eval_steps += 1

    acc = sum(1 for x,y in zip(preds, true_labels) if x == y) / len(preds)
    results["accuracy"] =  acc
    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results


def main():
    parser = HfArgumentParser(LTHTrainingArguments)

    if sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    logger.info("**************************************************************")
    logger.info("**** XNLI fine-tuning on {} data.****\n".format(args.data_language))
    logger.info(f"* Pruning type: {args.pruning_type}\n")
    logger.info(f"* output dir: {args.output_dir}\n")
    logger.info(f"* mask dir: {args.mask_dir}")
    logger.info("**************************************************************")


    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
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
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab


    if args.do_train:
        train_dataset = load_dataset(args.dataset_name, args.data_language, split="train", cache_dir=args.cache_dir)
        label_list = train_dataset.features["label"].names

    if args.do_eval:
        eval_dataset = load_dataset(args.dataset_name, args.data_language, split="validation", cache_dir=args.cache_dir)
        label_list = eval_dataset.features["label"].names

    if args.do_predict:
        predict_dataset = load_dataset(args.dataset_name, args.data_language, split="test", cache_dir=args.cache_dir)
        label_list = predict_dataset.features["label"].names

    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name
    )
    
    tokenizer_name_or_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    tokenizer = MT5Tokenizer.from_pretrained(
            tokenizer_name_or_path,
            use_fast=True
        )

    if os.path.exists(args.model_name_or_path):
        model = torch.load(os.path.join(args.model_name_or_path, "model.pt"))
        model = torch.nn.DataParallel(model)
        model.to(args.device)
    else:
        if args.weight_init == 'pre':
            model = MT5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config
        )

        model = torch.nn.DataParallel(model)
        model.to(args.device)
        
        if args.pruning_type == "random":
            logger.info("Random pruning.....")
            random_pruning_mt5_model(model, px=args.sparsity * 0.01)
            zero_rate = see_mt5_weight_rate(model)
            logger.info(f"Model zero rate: {zero_rate}\n")
        
        elif args.pruning_type == "oneshot":
            logger.info("Oneshot pruning.....")
            pruning_mt5_model(model, px=args.sparsity * 0.01)
            zero_rate = see_mt5_weight_rate(model)
            logger.info(f"Model zero rate: {zero_rate}\n")
            
        elif args.pruning_type == "lth":
            if args.mask_dir:
                logger.info("\nPruning using a mask...")
                mask = torch.load(args.mask_dir, map_location=args.device)
                pruning_mt5_model_custom(model, mask)
                zero_rate = see_mt5_weight_rate(model)
                logger.info(f"Model zero rate: {zero_rate}\n")
            else:
                raise ValueError("Need a trained mask!")

    if args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
    train_dataset = train_dataset.map(lambda example: {'input_seq': "xnli premise: " + str(example["premise"]) +
                                                                    " hypothesis:  " + str(example["hypothesis"]),
                                                       'target_seq': label_map[example['label']]})
    eval_dataset = eval_dataset.map(lambda example: {'input_seq': "xnli premise: " + str(example["premise"]) +
                                                                  " hypothesis:  " + str(example["hypothesis"]),
                                                     'target_seq': label_map[example['label']]})

    def preprocess_function(examples):
        # Tokenize the texts
        model_inputs = tokenizer(examples["input_seq"], max_length=args.max_source_length, padding=padding,
                                 truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["target_seq"], max_length=args.max_target_length, padding=padding,
                               truncation=True)
        
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    if args.do_train:
        if args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if args.do_eval:
        if args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if args.do_predict:
        if args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if args.pad_to_max_length:
        data_collator = default_data_collator
    elif args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
        
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        global_step, tr_loss = train(args, train_dataset, eval_dataset, model, data_collator, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        results = evaluate(args, eval_dataset, model, data_collator, tokenizer)
        print(f"Final eval result: {results}")
    elif args.do_eval:
        results = evaluate(args, eval_dataset, model, data_collator, tokenizer)
        print(f"Evaluation result: {results}")


if __name__ == "__main__":
    main()
