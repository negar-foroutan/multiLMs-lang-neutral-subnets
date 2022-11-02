from asyncio.log import logger
import torch


def get_mask_union(mask_list):
    """ Computes number of active neurons of two pruned masks. """
    union_mask = {}
    keys = list(mask_list[1].keys())
    for key in keys:
        data_list = []
        for mask_dict in mask_list:
            data_list.append(mask_dict[key])
        result = torch.stack(data_list, dim=0).sum(dim=0)
        result[result != 0] = 1
        union_mask[key] = result
    return union_mask


def get_mt5_mask_ones_overlap(mask_list):
    """ Computes number of active neurons of two pruned masks. """
    total_similarity = 0
    total_size = 0
    encoder_similarity = []
    decoder_similarity = []
    
    encoder_components = ["layer.0.SelfAttention.q.weight_mask", "layer.0.SelfAttention.k.weight_mask",
                          "layer.0.SelfAttention.v.weight_mask", "layer.0.SelfAttention.o.weight_mask",
                          "layer.1.DenseReluDense.wi_0.weight_mask", "layer.1.DenseReluDense.wi_1.weight_mask",
                          "layer.1.DenseReluDense.wo.weight_mask"]
    decoder_components = ["layer.0.SelfAttention.q.weight_mask", "layer.0.SelfAttention.k.weight_mask",
                          "layer.0.SelfAttention.v.weight_mask", "layer.0.SelfAttention.o.weight_mask",
                          "layer.1.EncDecAttention.q.weight_mask", "layer.1.EncDecAttention.k.weight_mask",
                          "layer.1.EncDecAttention.v.weight_mask", "layer.1.EncDecAttention.o.weight_mask",
                          "layer.2.DenseReluDense.wi_0.weight_mask", "layer.2.DenseReluDense.wi_1.weight_mask",
                          "layer.2.DenseReluDense.wo.weight_mask"]

    for ii in range(12):
        for comp in encoder_components:
            layer_similarity = 0
            layer_size = 0

            data_list = []
            for mask_dict in mask_list:
                data_list.append(mask_dict[f"module.encoder.block.{ii}.{comp}"])
            result = torch.stack(data_list, dim=0).sum(dim=0)
            layer_size += len(result[result != 0])
            layer_similarity += len(result[result == len(mask_list)])

            total_similarity += layer_similarity
            total_size += layer_size
        similarity = float(layer_similarity / layer_size)
        encoder_similarity.append(similarity)
    
    for ii in range(12):
        for comp in decoder_components:
            layer_similarity = 0
            layer_size = 0

            data_list = []
            for mask_dict in mask_list:
                data_list.append(mask_dict[f"module.decoder.block.{ii}.{comp}"])
            result = torch.stack(data_list, dim=0).sum(dim=0)
            layer_size += len(result[result != 0])
            layer_similarity += len(result[result == len(mask_list)])

            total_similarity += layer_similarity
            total_size += layer_size
        similarity = float(layer_similarity / layer_size)
        decoder_similarity.append(similarity)

    print("Total similarity rate: {:.4f}".format(float(total_similarity / total_size)))
    return encoder_similarity, decoder_similarity


def get_mbert_mask_ones_overlap(mask_list):
    """ Computes number of active neurons of two pruned masks. """
    total_similarity = 0
    total_size = 0
    all_layers_similarity = []
    for ii in range(12):
        layer_similarity = 0
        layer_size = 0

        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            data_list = []
            for mask_dict in mask_list:
                data_list.append(mask_dict[f'bert.encoder.layer.{ii}.{module_name}.weight_mask'])
            result = torch.stack(data_list, dim=0).sum(dim=0)
            layer_size += len(result[result != 0])
            layer_similarity += len(result[result == len(mask_list)])

        total_similarity += layer_similarity
        total_size += layer_size
        similarity = float(layer_similarity / layer_size)
        # print("Layer: {}, similarity rate: {:.4f}".format(ii, similarity))
        all_layers_similarity.append(similarity)

    if "bert.pooler.dense.weight_mask" in mask_list[0]:
        data_list = []
        for mask_dict in mask_list:
            data_list.append(mask_dict['bert.pooler.dense.weight_mask'])
        result = torch.stack(data_list, dim=0).sum(dim=0)
        layer_size += len(result[result != 0])
        layer_similarity += len(result[result == len(mask_list)])

        total_similarity += layer_similarity
        total_size += layer_size

    print("Total similarity rate: {:.4f}".format(float(total_similarity / total_size)))
    return all_layers_similarity


def see_mask_zero_rate(mask_dict):
    """ Compute the percenrage of the pruned neurons using the given mask (mBERT). """
    
    total_size = 0.0
    zero_size = 0.0
    for key in list(mask_dict.keys()):
        total_size += float(mask_dict[key].nelement())
        zero_size += float(torch.sum(mask_dict[key] == 0))

    zero_rate = (100 * zero_size) / total_size
    return zero_rate


def see_mask_zero_rate_per_layer(mask_dict):
    """ Compute the percenrage of the pruned neurons for each layer given the pruned mask (mBERT). """

    layers_zero_rate = []
    for idx in range(12):
        sum_list = 0
        zero_sum = 0
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            key = f"bert.encoder.layer.{idx}.{module_name}.weight_mask"
            sum_list = sum_list + float(mask_dict[key].nelement())
            zero_sum = zero_sum + float(torch.sum(mask_dict[key] == 0))

        layers_zero_rate.append(100 * zero_sum / sum_list)
    return layers_zero_rate


def find_intersection_ones(mask_list):
    """ Find the intersection of the ones in the mask list (mBERT) """
    final_mask_dict = {}
    zero_rate = 0
    
    for idx in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            data_list = []
            key = f"bert.encoder.layer.{idx}.{module_name}.weight_mask"
            for mask_dict in mask_list:
                data_list.append(mask_dict[key])
                
            result = torch.stack(data_list, dim=0).sum(dim=0)
            result[result != len(mask_list)] = 0
            result[result == len(mask_list)] = 1
            final_mask_dict[f'bert.encoder.layer.{idx}.attention.self.query.weight_mask'] = result
            zero_rate += len(result == 0)
    
    logger.info(f"Intersection zero rate: {zero_rate}")
    return final_mask_dict


def mask_sparsity_overlap(mask_list):
    """ Computes mask sparsity overlap across the mask list """
    total_similarity = 0
    total_size = 0
    all_layers_similarity = []
    for idx in range(12):
        data_list = []
        layer_similarity = 0
        layer_size = 0
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            data_list = []
            key = f"bert.encoder.layer.{idx}.{module_name}.weight_mask"
            for mask_dict in mask_list:
                data_list.append(mask_dict[key])
            result = torch.stack(data_list, dim=0).sum(dim=0)
            layer_similarity += torch.sum((result == len(mask_list))) + torch.sum((result == 0))
            layer_size += data_list[0].nelement()

        total_similarity += layer_similarity
        total_size += layer_size
        similarity = float(layer_similarity / layer_size)
        all_layers_similarity.append(similarity)

    if "bert.pooler.dense.weight_mask" in mask_list[0]:
        data_list = []
        for mask_dict in mask_list:
            data_list.append(mask_dict['bert.pooler.dense.weight_mask'])
        result = torch.stack(data_list, dim=0).sum(dim=0)
        layer_similarity += torch.sum((result==len(mask_list))) + torch.sum((result==0))
        layer_size += data_list[0].nelement()

        total_similarity += layer_similarity
        total_size += layer_size

    print("Total similarity rate: {:.4f}".format(float(total_similarity / total_size)))
    return all_layers_similarity


def see_mask_zero_rate_per_component(mask_dict):
    """ Computes sparsity rate for every component of each layer. """

    components = ["attention.self.query", "attention.self.key", "attention.self.value",
                  "attention.output.dense", "intermediate.dense", "output.dense"]
    all_zero_rate = {c:[] for c in components}

    for comp in components:
        avg = 0
        all_sum = 0
        for ii in range(12):
            sum_list = float(mask_dict['bert.encoder.layer.'+str(ii)+'.' + comp + '.weight_mask'].nelement())
            zero_sum = float(torch.sum(mask_dict['bert.encoder.layer.'+str(ii)+'.' + comp + '.weight_mask'] == 0))
            zero_rate = zero_sum / sum_list
            all_zero_rate[comp].append(zero_rate)
            avg += zero_sum
            all_sum += sum_list
            print(f"Layer: {ii}, component: {comp}, zero rate: {zero_rate:.3f}")
        print(f"Average zero rate: {avg / all_sum}")
        print("\n")
    return all_zero_rate


def mask_sparsity_overlap_per_component(mask_dict1, mask_dict2):
    """ Computes sparsity pattern similarity between every component of two pruned masks. """

    components = ["attention.self.query", "attention.self.key", "attention.self.value",
                  "attention.output.dense", "intermediate.dense", "output.dense"]
    comp_sparsity_overlap = {}

    for comp in components:
        for idx in range(12):
            key = f'bert.encoder.layer.{idx}.' + comp + '.weight_mask'
            comp_size = mask_dict1[key].nelement()
            not_xor = torch.logical_not(torch.logical_xor(mask_dict1[key], mask_dict2[key]))
            comp_similarity = torch.count_nonzero(not_xor)
            rate = comp_similarity * 1.0 / comp_size
            comp_sparsity_overlap[key] = rate

    return comp_sparsity_overlap
