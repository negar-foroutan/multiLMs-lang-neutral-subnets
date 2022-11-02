import time
import logging

import torch
import faiss
import numpy as np

from utils import knn

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)


def score(x, y, fwd_mean, bwd_mean, margin):
    return margin(x.dot(y), (fwd_mean + bwd_mean) / 2)


def score_candidates(x, y, candidate_inds, fwd_mean, bwd_mean, margin, verbose=False):
    if verbose:
        print(' - scoring {:d} candidates'.format(x.shape[0]))
    scores = np.zeros(candidate_inds.shape)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            k = candidate_inds[i, j]
            scores[i, j] = score(x[i], y[k], fwd_mean[i], bwd_mean[k], margin)
    return scores


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def sentence_retrieval_by_layer(dataset, src_model, target_model, tokenizer, src, target,
                                agr_func, margin_func="ratio", neighborhood=5, gpu=False):
    device_ = torch.device("cuda" if gpu else "cpu")
    accuracy_per_layer = []
    src_all_embeddings = {layer_num: [] for layer_num in range(13)}
    tgt_all_embeddings = {layer_num: [] for layer_num in range(13)}

    for i in range(len(dataset)):
        pair = dataset[i]['translation']
        encoded_src_input = tokenizer([pair[src]], padding="max_length", truncation=True,
                                      return_tensors='pt').to(device_)
        encoded_tgt_input = tokenizer([pair[target]], padding="max_length", truncation=True,
                                      return_tensors='pt').to(device_)

        with torch.no_grad():
            model_src_output = src_model(**encoded_src_input, output_hidden_states=True)
            model_tgt_output = target_model(**encoded_tgt_input, output_hidden_states=True)
            for layer_num in range(13):
                if agr_func == "mean":
                    sent1_embd = mean_pooling(model_src_output["hidden_states"][layer_num],
                                              encoded_src_input['attention_mask']).cpu()[0].numpy()
                    sent2_embd = mean_pooling(model_tgt_output["hidden_states"][layer_num],
                                              encoded_tgt_input['attention_mask']).cpu()[0].numpy()
                else:
                    sent1_embd = model_src_output["hidden_states"][layer_num][:, 0, :].flatten().cpu().numpy()
                    sent2_embd = model_tgt_output["hidden_states"][layer_num][:, 0, :].flatten().cpu().numpy()

                src_all_embeddings[layer_num].append(sent1_embd)
                tgt_all_embeddings[layer_num].append(sent2_embd)

    for layer in range(13):
        x = src_all_embeddings[layer]
        y = tgt_all_embeddings[layer]
        t1 = time.time()
        # logger.info("************ Layer number: {} ************".format(layer))
    
        x = np.array(x)
        y = np.array(y)

        faiss.normalize_L2(x)
        faiss.normalize_L2(y)

        x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], neighborhood), gpu)
        x2y_mean = x2y_sim.mean(axis=1)

        y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], neighborhood), gpu)
        y2x_mean = y2x_sim.mean(axis=1)

        # margin function
        if margin_func == 'absolute':
            margin = lambda a, b: a
        elif margin_func == 'distance':
            margin = lambda a, b: a - b
        else:  # args.margin == 'ratio':
            margin = lambda a, b: a / b

        scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
        best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

        nbex = x.shape[0]
        ref = np.linspace(0, nbex-1, nbex).astype(int)  # [0, nbex)
        err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
        acc = 100 - (100*err/nbex)
        accuracy_per_layer.append(acc)
    return accuracy_per_layer


def sentence_retrieval(dataset, src_model, target_model, tokenizer, src, target,
                       margin_func="ratio", neighborhood=5,  gpu=False):
    """ Sentence retrieval using bert representation and margin score knn """
    
    x = []
    y = []

    for i in range(len(dataset)):
        pair = dataset[i]['translation']
        encoded_src_input = tokenizer([pair[src]], padding="max_length", truncation=True,
                                      return_tensors='pt').to("cuda:0")
        encoded_tgt_input = tokenizer([pair[target]], padding="max_length", truncation=True,
                                      return_tensors='pt').to("cuda:0")
        with torch.no_grad():
            model_src_output = src_model(**encoded_src_input)
            sentence_embeddings = mean_pooling(model_src_output[0], encoded_src_input['attention_mask']).cpu()
            x.append(sentence_embeddings[0].numpy())

            model_tgt_output = target_model(**encoded_tgt_input)
            sentence_embeddings = mean_pooling(model_tgt_output[0], encoded_tgt_input['attention_mask']).cpu()
            y.append(sentence_embeddings[0].numpy())
    
    x = np.array(x)
    y = np.array(y)

    faiss.normalize_L2(x)
    faiss.normalize_L2(y)

    x2y_sim, x2y_ind = knn(x, y, min(y.shape[0], neighborhood), gpu)
    x2y_mean = x2y_sim.mean(axis=1)

    y2x_sim, y2x_ind = knn(y, x, min(x.shape[0], neighborhood), gpu)
    y2x_mean = y2x_sim.mean(axis=1)

    # margin function
    if margin_func == 'absolute':
        margin = lambda a, b: a
    elif margin_func == 'distance':
        margin = lambda a, b: a - b
    else:  # args.margin == 'ratio':
        margin = lambda a, b: a / b

    scores = score_candidates(x, y, x2y_ind, x2y_mean, y2x_mean, margin)
    best = x2y_ind[np.arange(x.shape[0]), scores.argmax(axis=1)]

    nbex = x.shape[0]
    ref = np.linspace(0, nbex-1, nbex).astype(int)
    err = nbex - np.equal(best.reshape(nbex), ref).astype(int).sum()
    print(' - errors: {:d}={:.2f}%'.format(err, 100*err/nbex))