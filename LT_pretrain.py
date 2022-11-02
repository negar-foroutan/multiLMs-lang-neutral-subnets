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
""" Lottery Ticket Hypothesis (LTH) for the MLM pretraining """


import logging
import os
import random
import sys
from typing import Optional
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from helpers.utils import _sorted_checkpoints, _rotate_checkpoints, set_seed
from helpers.pruning_utils import rewind, pruning_model, see_weight_rate

import helpers.data_processor as data_processor


from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    get_linear_schedule_with_warmup,
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

    model_type: str = field(
        default="mbert", metadata={"help": "Model type"}
    )
    mlm: bool = field(
        default=True,  metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    save_model: bool = field(
        default=False, metadata={"help": "Whether to save the model or not."}
    )
    should_continue: bool = field(
        default=False, metadata={"help": "Whether to continue from latest checkpoint in output_dir"}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    max_grad_norm: float = field(
        default=1.0, metadata={"help": "Max gradient norm."}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Output directory path."}
    )
    log_dir: str = field(
        default=None, metadata={"help": "Log directory path."}
    )
    overwrite_output_dir: bool = field(
        default=True, metadata={"help": "Whether overwrite the output dir."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    data_language: str = field(
        default=None, metadata={"help": "Data language."}
    )
    save_steps: int = field(
        default=10000, metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_steps: int = field(
        default=10000, metadata={"help": "Log every X updates steps."}
    )
    evaluate_during_training: bool = field(
        default=True, metadata={"help": "Whether to evaluate during training or not."}
    )
    warmup_steps: int = field(
        default=0, metadata={"help": "Linear warmup over warmup_steps."}
    )
    rand_seed: bool = field(
        default=False,  metadata={"help": "Whether set a random seed or not."}
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
    learning_rate: float = field(
        default=2e-5, metadata={"help": "The initial learning rate for AdamW."}
    )
    num_train_epochs: float = field(
        default=1, metadata={"help": "Total number of training epochs to perform."}
    )
    block_size: int = field(
        default=512, metadata={"help": "Optional input sequence length after tokenization."}
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
        default=65, metadata={"help": "Random seed that will be set at the beginning of training."}
        )
    model_name_or_path: str = field(
        default="bert-base-multilingual-cased",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
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
        default="mlm",
        metadata={"help": "The name of the task (ner, pos...)."}
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
    val_data_ratio: Optional[float] = field(
        default=0.01,
        metadata={"help": "The ratio of validation data from the training data."},
    )
    max_seq_length: int = field(
        default=128,
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
    fp16: bool = field(
        default=False,
        metadata={"help":"Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"}
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={"help": "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                          "See details at https://nvidia.github.io/apex/amp.html"}
    )
    server_ip: str = field(
        default="",
        metadata={"help": "For distant debugging."}
    )
    server_port: str = field(
        default="",
        metadata={"help": "For distant debugging."}
    )


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. "
            "Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    '''We sample a few tokens in each sequence for masked-LM training (with probability
    args.mlm_probability defaults to 0.15 in Bert/RoBERTa)'''
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer, eval_dataset, orig):
    """ Train the model """
    record_result = []

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_device_train_batch_size * max(1, args.n_gpu)

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

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
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
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
    pruning_step = 0
    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            outputs = model(inputs, labels=labels)
            loss = outputs.loss

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
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        
                        rate_weight_equal_zero = see_weight_rate(model)
                        logger.info('zero_rate = ', rate_weight_equal_zero)

                        results = evaluate(args, model, tokenizer, eval_dataset)
                        logger.info(results)

                        record_result.append(results)

                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)

                    logger.info('starting pruning')
                    logger.info('saving model for ', 100-pruning_step*10)
                    pruning_model(model, 1/(10-pruning_step))
                    rate_weight_equal_zero = see_weight_rate(model)
                    pruning_step += 1
                    logger.info('zero_rate = ', rate_weight_equal_zero)

                    if args.save_model:
                        if hasattr(model, "module"):
                                torch.save(model.module, os.path.join(output_dir, "model.pt"))
                        else:
                            torch.save(model, os.path.join(output_dir, "model.pt"))

                        logger.info("Saving model checkpoint to %s", output_dir)
                        _rotate_checkpoints(args, checkpoint_prefix)

                    logger.info('rewinding')
                    model_dict = model.state_dict()
                    model_dict.update(orig)
                    model.load_state_dict(model_dict)

                    mask_dict = {}
                    for key in model_dict.keys():
                        if 'mask' in key:
                            mask_dict[key] = model_dict[key]   

                    torch.save(mask_dict, os.path.join(output_dir, "mask.pt"))

                    logger.info('optimizer rewinding')
                    no_decay = ["bias", "LayerNorm.weight"]
                    optimizer_grouped_parameters = [
                        {
                            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                            "weight_decay": args.weight_decay,
                        },
                        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                         "weight_decay": 0.0},
                    ]
                    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
                    )

            if pruning_step == 10:
                epoch_iterator.close()
                break

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if pruning_step == 10:
            epoch_iterator.close()
            break        
        
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    torch.save(record_result, os.path.join(args.output_dir, "result.pt"))

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_dataset, prefix=""):

    eval_output_dir = args.output_dir

    top1 = AverageMeter()

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_device_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs.loss
            prediction_scores = outputs.logits

            vocab_size = prediction_scores.size(-1)
            acc = accuracy(prediction_scores.view(-1, vocab_size).data, labels.view(-1))[0]
            top1.update(acc.item(), labels.view(-1).size(0))

            eval_loss += lm_loss.detach().mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, 'acc': top1.avg}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def accuracy(output_orig, label, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    index = torch.nonzero(label+100)
    target = label[index].view(-1)
    output = output_orig[index].squeeze(1)

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():

    parser = HfArgumentParser(LTHTrainingArguments)

    if sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]

    logger.info("****** LTH for MLM task *****")
    logger.info(f"args.rand_seed :{args.rand_seed}\n")

    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

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

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Set seed for reproducibility
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir
    )

    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            cache_dir=args.cache_dir
    )

    if args.block_size <= 0:
        args.block_size = tokenizer.model_max_length
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.model_max_length)

    if args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM(config=config)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    origin_model_dict = rewind(model.state_dict())
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        
        data = data_processor.JsonTextDataset(tokenizer, args, args.train_file)
        if args.validation_file is not None:
            eval_dataset = data_processor.JsonTextDataset(tokenizer, args, args.validation_file)
            train_dataset = data
        else:
            data.expand_dataset(10 * 160000)
            dataset_len = len(data)
            val_data_size = max(20000, int(args.val_data_ratio * dataset_len))
            train_data_size = dataset_len - val_data_size
            train_dataset, eval_dataset = torch.utils.data.random_split(data, [train_data_size, val_data_size],
                                                                        generator=torch.Generator().manual_seed(42))

        logger.info("Train data size: {}".format(len(train_dataset)))
        logger.info("Test data size: {}".format(len(eval_dataset)))

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, eval_dataset, origin_model_dict)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if __name__ == "__main__":
    main()
