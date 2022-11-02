import torch.nn.utils.prune as prune
import torch


def get_mt5_params_to_prune(model):
    """ Returns the parameters we want to prune for mT5 model """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    parameters_to_prune = {}
    encoder_params = []
    decoder_params = []
    # parameters_to_prune.append((model.lm_head, 'weight'))
    for ii in range(len(model.encoder.block)):
        # Self attention
        encoder_params.append((model.encoder.block[ii].layer[0].SelfAttention.q, 'weight'))
        encoder_params.append((model.encoder.block[ii].layer[0].SelfAttention.k, 'weight'))
        encoder_params.append((model.encoder.block[ii].layer[0].SelfAttention.v, 'weight'))
        encoder_params.append((model.encoder.block[ii].layer[0].SelfAttention.o, 'weight'))
        # FF layer
        encoder_params.append((model.encoder.block[ii].layer[1].DenseReluDense.wi_0, 'weight'))
        encoder_params.append((model.encoder.block[ii].layer[1].DenseReluDense.wi_1, 'weight'))
        encoder_params.append((model.encoder.block[ii].layer[1].DenseReluDense.wo, 'weight'))

        # Self attention
        decoder_params.append((model.decoder.block[ii].layer[0].SelfAttention.q, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[0].SelfAttention.k, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[0].SelfAttention.v, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[0].SelfAttention.o, 'weight'))
        # cross attention
        decoder_params.append((model.decoder.block[ii].layer[1].EncDecAttention.q, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[1].EncDecAttention.k, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[1].EncDecAttention.v, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[1].EncDecAttention.o, 'weight'))
        # FF layer
        decoder_params.append((model.decoder.block[ii].layer[2].DenseReluDense.wi_0, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[2].DenseReluDense.wi_1, 'weight'))
        decoder_params.append((model.decoder.block[ii].layer[2].DenseReluDense.wo, 'weight'))


    parameters_to_prune["encoder"] = encoder_params
    parameters_to_prune["decoder"] = decoder_params
    return parameters_to_prune

def pruning_mt5_model(model, px, glb=False):
    """ Performs magnitude pruning for mT5 model
    Args:
        model: mT5 model
        px (float): Pruning rate (a number between 0 and 1)
        glb (bool): Wether prune the encoder and decoder together or separately. 
    """
    parameters_to_prune = get_mt5_params_to_prune(model)
    encoder_params = parameters_to_prune["encoder"]
    decoder_params = parameters_to_prune["decoder"]
    if glb:
        parameters_to_prune = tuple(encoder_params + decoder_params)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
    else:
        parameters_to_prune = tuple(encoder_params)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )
        parameters_to_prune = tuple(decoder_params)
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=px,
        )


def get_params_to_prune(model_to_prune):
    """ Returns the parameters we want to prune for mBERT model """
    parameters_to_prune = []
    if isinstance(model_to_prune, torch.nn.DataParallel):
        model = model_to_prune.module
    else:
        model = model_to_prune
   
    for layer_num in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            parameters_to_prune.append((eval(f"model.bert.encoder.layer[{layer_num}].{module_name}"), 'weight'))

    if model.bert.pooler is not None:
        parameters_to_prune.append((model.bert.pooler.dense, 'weight'))
        
    parameters_to_prune = tuple(parameters_to_prune)
    return parameters_to_prune


def component_wise_pruning_model(model, px):
    """ Performs component-wise magnitude pruning for mBERT model 
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    for param in parameters_to_prune:
        prune.l1_unstructured(param[0], name="weight", amount=px)


def pruning_model(model, px):
    """ Performs magnitude pruning for mBERT model
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )


def random_pruning_mt5_model(model, px):
    """ Performs random pruning for mT5 model 
    Args:
        model: mT5 model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_mt5_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune["encoder"],
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )
    prune.global_unstructured(
        parameters_to_prune["decoder"],
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def random_pruning_model(model, px):
    """ Performs random pruning for mBERT model
    Args:
        model: mBERT model
        px (float): Pruning rate (a number between 0 and 1)
    """
    parameters_to_prune = get_params_to_prune(model)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )


def pruning_model_custom(model, mask_dict):
    """ Performs custom pruning given a pruning mask for mBERT model """
    parameters_to_prune = []
    mask_list = []
    for layer_num in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            parameters_to_prune.append(eval(f"model.bert.encoder.layer[{layer_num}].{module_name}"))
            mask_list.append(mask_dict[f'bert.encoder.layer.{layer_num}.{module_name}.weight_mask'])

    if model.bert.pooler is not None and 'bert.pooler.dense.weight_mask' in mask_dict:
        parameters_to_prune.append(model.bert.pooler.dense)
        mask_list.append(mask_dict['bert.pooler.dense.weight_mask'])

    for idx in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[idx], 'weight', mask=mask_list[idx])


def pruning_mt5_model_custom(model, mask_dict):
    """ Performs custom pruning given a pruning mask for mT5 model """

    prefix = ""
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
        prefix = "module."
    parameters_to_prune =[]
    mask_list = []
    if prefix + "lm_head.weight_mask" in mask_dict:
        parameters_to_prune.append(model.lm_head)
        mask_list.append(mask_dict[prefix + "lm_head.weight_mask"])

    for ii in range(len(model.encoder.block)):
        # Self attention
        parameters_to_prune.append(model.encoder.block[ii].layer[0].SelfAttention.q)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.0.SelfAttention.q.weight_mask"])
        parameters_to_prune.append(model.encoder.block[ii].layer[0].SelfAttention.k)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.0.SelfAttention.k.weight_mask"])
        parameters_to_prune.append(model.encoder.block[ii].layer[0].SelfAttention.v)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.0.SelfAttention.v.weight_mask"])
        parameters_to_prune.append(model.encoder.block[ii].layer[0].SelfAttention.o)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.0.SelfAttention.o.weight_mask"])
        # FF layer
        parameters_to_prune.append(model.encoder.block[ii].layer[1].DenseReluDense.wi_0)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wi_0.weight_mask"])
        parameters_to_prune.append(model.encoder.block[ii].layer[1].DenseReluDense.wi_1)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wi_1.weight_mask"])
        parameters_to_prune.append(model.encoder.block[ii].layer[1].DenseReluDense.wo)
        mask_list.append(mask_dict[prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wo.weight_mask"])

        # Self attention
        parameters_to_prune.append(model.decoder.block[ii].layer[0].SelfAttention.q)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.0.SelfAttention.q.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[0].SelfAttention.k)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.0.SelfAttention.k.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[0].SelfAttention.v)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.0.SelfAttention.v.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[0].SelfAttention.o)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.0.SelfAttention.o.weight_mask"])
        # cross attention
        parameters_to_prune.append(model.decoder.block[ii].layer[1].EncDecAttention.q)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.q.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[1].EncDecAttention.k)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.k.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[1].EncDecAttention.v)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.v.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[1].EncDecAttention.o)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.o.weight_mask"])
        # FF layer
        parameters_to_prune.append(model.decoder.block[ii].layer[2].DenseReluDense.wi_0)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wi_0.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[2].DenseReluDense.wi_1)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wi_1.weight_mask"])
        parameters_to_prune.append(model.decoder.block[ii].layer[2].DenseReluDense.wo)
        mask_list.append(mask_dict[prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wo.weight_mask"])

    for ii in range(len(parameters_to_prune)):
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list[ii])


def see_weight_rate(model): 
    """ Computes the sparsity level of the given mBERT model """
    sum_list = 0
    zero_sum = 0
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for idx in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            sum_list = sum_list + float(eval(f"model.bert.encoder.layer[{idx}].{module_name}.weight.nelement()"))
            zero_sum = zero_sum +\
                float(torch.sum(eval(f"model.bert.encoder.layer[{idx}].{module_name}.weight") == 0))

    if model.bert.pooler is not None:
        sum_list = sum_list + float(model.bert.pooler.dense.weight.nelement())
        zero_sum = zero_sum + float(torch.sum(model.bert.pooler.dense.weight == 0))
 
    return 100.0 * zero_sum / sum_list


def see_mt5_weight_rate(model):
    """ Computes the sparsity level of the given mT5 model """
    sum_list = 0
    zero_sum = 0
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    encoder = model.encoder
    decoder = model.decoder

    for ii in range(len(encoder.block)):
        # Encoder self attention
        sum_list += float(encoder.block[ii].layer[0].SelfAttention.q.weight.nelement())
        zero_sum += torch.sum(encoder.block[ii].layer[0].SelfAttention.q.weight == 0)

        sum_list += encoder.block[ii].layer[0].SelfAttention.k.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[0].SelfAttention.k.weight == 0)

        sum_list += encoder.block[ii].layer[0].SelfAttention.v.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[0].SelfAttention.v.weight == 0)

        sum_list += encoder.block[ii].layer[0].SelfAttention.o.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[0].SelfAttention.o.weight == 0)

        # Encoder FF layer
        sum_list += encoder.block[ii].layer[1].DenseReluDense.wi_0.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[1].DenseReluDense.wi_0.weight == 0)

        sum_list += encoder.block[ii].layer[1].DenseReluDense.wi_1.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[1].DenseReluDense.wi_1.weight == 0)

        sum_list += encoder.block[ii].layer[1].DenseReluDense.wo.weight.nelement()
        zero_sum += torch.sum(encoder.block[ii].layer[1].DenseReluDense.wo.weight == 0)

        # Self attention
        sum_list += decoder.block[ii].layer[0].SelfAttention.q.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[0].SelfAttention.q.weight == 0)

        sum_list += decoder.block[ii].layer[0].SelfAttention.k.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[0].SelfAttention.k.weight == 0)

        sum_list += decoder.block[ii].layer[0].SelfAttention.v.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[0].SelfAttention.v.weight == 0)

        sum_list += decoder.block[ii].layer[0].SelfAttention.o.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[0].SelfAttention.o.weight == 0)

        # cross attention
        sum_list += decoder.block[ii].layer[1].EncDecAttention.q.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[1].EncDecAttention.q.weight == 0)

        sum_list += decoder.block[ii].layer[1].EncDecAttention.k.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[1].EncDecAttention.k.weight == 0)

        sum_list += decoder.block[ii].layer[1].EncDecAttention.v.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[1].EncDecAttention.v.weight == 0)

        sum_list += decoder.block[ii].layer[1].EncDecAttention.o.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[1].EncDecAttention.o.weight == 0)

        # Decoder FF layer
        sum_list += decoder.block[ii].layer[2].DenseReluDense.wi_0.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[2].DenseReluDense.wi_0.weight == 0)

        sum_list += decoder.block[ii].layer[2].DenseReluDense.wi_1.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[2].DenseReluDense.wi_1.weight == 0)

        sum_list += decoder.block[ii].layer[2].DenseReluDense.wo.weight.nelement()
        zero_sum += torch.sum(decoder.block[ii].layer[2].DenseReluDense.wo.weight == 0)
   
    return 100.0 * zero_sum / sum_list


def rewind_mt5_model(pre_weight):
    """ Rewinds to the given weights (pre-trained weights) for mT5 model """
    recover_dict = {}
    name_list = []
    pre_weight_keys = pre_weight.keys()
    prefix = ""

    for key in pre_weight_keys:
        if key.startswith("module"):
            prefix = "module."
            break
        
    for ii in range(12):
        name_list.append(prefix + f"encoder.block.{ii}.layer.0.SelfAttention.q.weight")
        name_list.append(prefix + f"encoder.block.{ii}.layer.0.SelfAttention.k.weight")
        name_list.append(prefix + f"encoder.block.{ii}.layer.0.SelfAttention.v.weight")
        name_list.append(prefix + f"encoder.block.{ii}.layer.0.SelfAttention.o.weight")
        # FF layer
        name_list.append(prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wi_0.weight")
        name_list.append(prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wi_1.weight")
        name_list.append(prefix + f"encoder.block.{ii}.layer.1.DenseReluDense.wo.weight")

        # Self attention
        name_list.append(prefix + f"decoder.block.{ii}.layer.0.SelfAttention.q.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.0.SelfAttention.k.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.0.SelfAttention.v.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.0.SelfAttention.o.weight")
        # cross attention

        name_list.append(prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.q.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.k.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.v.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.1.EncDecAttention.o.weight")
        # FF layer
        name_list.append(prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wi_0.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wi_1.weight")
        name_list.append(prefix + f"decoder.block.{ii}.layer.2.DenseReluDense.wo.weight")

    for key in pre_weight.keys():

        if key in name_list:
            new_key = key + '_orig'
        else:
            new_key = key

        recover_dict[new_key] = pre_weight[key]

    return recover_dict


def rewind(pre_weight):
    """ Rewinds to the given weights (pre-trained weights) for mBERT model """
    recover_dict = {}
    name_list = []
    pre_weight_keys = pre_weight.keys()
    prefix = ""

    for key in pre_weight_keys:
        if key.startswith("module"):
            prefix = "module."
            break
    
    for layer_num in range(12):
        for module_name in ["attention.self.query", "attention.self.key", "attention.self.value",
                            "attention.output.dense", "intermediate.dense", "output.dense"]:
            name_list.append(prefix + f'bert.encoder.layer.{layer_num}.{module_name}.weight')

    name_list.append(prefix + 'bert.pooler.dense.weight')

    for key in pre_weight.keys():

        if 'bert' in key:
            if key in name_list:
                new_key = key + '_orig'
            else:
                new_key = key

            recover_dict[new_key] = pre_weight[key]

    return recover_dict
