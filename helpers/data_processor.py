import bz2
import json
import math 
import os
import time
import pickle
import logging
import ast
import json

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


class JsonTextDataset(Dataset):
    """ To represent a Wikipedia dataset. """
    def __init__(self, tokenizer, args, file_path):
        
        if file_path.endswith(".pkl") and not args.overwrite_cache:
            if os.path.exists(file_path):
                logger.info("Loading features from cached file %s", file_path)
                with open(file_path, "rb") as handle:
                    self.examples = pickle.load(handle)
            else:
                raise ValueError("The cache file does not exist!")
        else:
            logger.info("Creating features from dataset file at %s", file_path)
            
            block_size = args.block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
            directory, filename = os.path.split(file_path)
            cached_features_file = os.path.join(
                    directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename + ".pkl"
            )

            t0 = time.time()
            input_file = open(file_path, 'r')
            log_counter = 0
            self.examples = []

            for line in input_file:
                doc = json.loads(line)
                text = doc["text"].strip()
                if text == "":
                    continue
                text = text.replace("\n", " ")
                text = text.replace("[...]", "")
                if "src=" in text: 
                    continue

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                num_examples = int(len(tokenized_text)/ block_size)

                for i in range(0, num_examples, block_size):  # Truncate in block of block_size
                    self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
                
                if len(tokenized_text) > num_examples * block_size:
                    last_example = tokenizer.build_inputs_with_special_tokens(tokenized_text[num_examples * block_size:])
                    pad_size = block_size - len(last_example) + 2
                    for _ in range(pad_size):
                        last_example.append(tokenizer.pad_token_id)
                    self.examples.append(last_example)
                
                log_counter += 1
                if log_counter % 20000 == 0:
                    logger.info("{} documents are processed so far in {:.2f} seconds.".format(log_counter, time.time() - t0))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should look for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.
            logger.info("Number of samples: {}, Time: {:.2f} seconds.".format(len(self.examples), time.time() - t0))
            logger.info("Saving features into cached file %s", cached_features_file)
            
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)
    
    def expand_dataset(self, data_size=9*160000):
        if len(self.examples) < data_size:
            print("Expanding the dataset.....")
            initial_data = self.examples.copy()
            n = math.ceil(data_size / len(initial_data))
            for _ in range(n):
                self.examples += initial_data


def preprocess_compressed_json_file(in_file_path, out_folder=None):
    """" Reads compressed json files (bz2), removes the records with empty text,a nd saves the result as a new file.
    Args:
        in_file_path (string): Input file path.
        out_folder (string): Output folder (default:None)
    """
    if not in_file_path.endswith(".bz2"):
        raise ValueError("The input file needs to be in bz2 format!")

    out_file_name = in_file_path.split("/")[-2] + "_" + in_file_path.split("/")[-1].split(".")[0] + "_processed.json"
    if out_folder is not None:
        out_file_path = os.path.join(out_folder, out_file_name)
    else:
        out_file_path = "/".join(in_file_path.split("/")[0:-1]) + "/" + out_file_name

    out = []
    try:
        with bz2.open(in_file_path, "r") as fin:
            documents = fin.readlines()
            for doc in documents:
                doc_str = doc.decode("UTF-8")
                doc_dict = ast.literal_eval(doc_str)
                text = doc_dict["text"].strip()
                if text == "":
                    continue
                new_doc = {"id": doc_dict["id"], "text": text}
                out.append(new_doc)

        with open(out_file_path, 'w') as fout:
            json.dump(out , fout)
    except EOFError as e:
        print("{}: {}".format(in_file_path, e))
        

def process_list_of_compressed_json_files(dir_path, out_folder=None):
    """" Processed all the compressed files existing in the given directory.
    Args:
        dir_path (string): Input directory path.
        out_folder (string): Output folder (default:None)
    """
    print("Directory: {}".format(dir_path))
    files_list = [v for v in os.listdir(dir_path) if v.endswith(".bz2")]
    for file_name in files_list:
        file_path = os.path.join(dir_path, file_name)
        preprocess_compressed_json_file(file_path, out_folder)
    
    print("{} files are processed\n".format(len(files_list)))


def process_all_compressed_json_files(dir_path, out_folder=None):
    """" Processed all the dataset directories.
    Args:
        dir_path (string): Data directory path.
        out_folder (string): Output directory (default:None)
    """
    list_dir = os.listdir(dir_path)
    print(list_dir)
    for dir_name in list_dir:
        child_dir_path = os.path.join(dir_path, dir_name)
        if not os.path.isdir(child_dir_path):
            continue
        process_list_of_compressed_json_files(child_dir_path, out_folder)
