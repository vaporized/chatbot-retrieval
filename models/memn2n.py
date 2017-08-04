import numpy as np
import tensorflow as tf
import models.memn2n_preprocessing as mp
from models.memn2n_2d_layerwise import memn2n_model_2d_layerwise as m2dl
from models.memn2n_2d_adjacent import memn2n_model_2d_adjacent as m2da
from models.memn2n_3d_layerwise import memn2n_model_3d_layerwise as m3dl
from models.memn2n_3d_adjacent import memn2n_model_3d_adjacent as m3da


FLAGS = tf.flags.FLAGS

def memn2n_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):
    """the memn2n model with 3 general training modes.

    Args:
        hparams: Hyper-parameters of the model, including training scheme.
        mode: Indicator of training or validation
        context: Tensor of shape (batch_size, context_len)
        context_len: Tensor of shape (batch_size,)
        utterance: Tensor of shape (batch_size, utterance_len)
        utterance_len: Tensor of shape (batch_size, )
        targets: Tensor of shape (batch_size, 1)

    Returns:
        (probs, mean_loss) for training mode, or (probs, None) for validation

    """


    #extract parameters

    #model type: adjacent or layer_wise
    model_type = hparams.model_type

    #training mode
    preprocessing_mode = hparams.preprocessing_mode

    #dictionary of type model correspondence
    call_model = {'2d': {"adjacent": m2da, "layer_wise": m2dl},
                  '3d': {"adjacent": m3da, "layer_wise": m3dl}

    #call different models accordingly
    if preprocessing_mode == None:
        return call_model['2d'][model_type](hparams, mode, context, utterance, targets)

    elif preprocessing_mode == "eou":
        max_s = hparams.max_sentence_len_split_eou
        mem = hparams.memory_size_split_eou
        context_3d = mp.split_tensor_batch(context, max_s, mem, 1)

    elif preprocessing_mode == "eot":
        max_s = hparams.max_sentence_len_split_eot
        mem = hparams.memory_size_split_eot
        context_3d = mp.split_tensor_batch(context, max_s, mem, 2)

    elif preprocessing_mode == "split":
        max_s = hparams.max_sentence_len_split
        mem = hparams.memory_size_split
        context_3d = mp.split_overlap_batch(context, max_s, mem)

    elif preprocessing_mode == "split_overlap":
        max_s = hparams.max_sentence_len_overlap
        mem = hparams.memory_size_overlap
        overlap_s = hparams.overlapping_size
        context_3d = mp.split_overlap_batch(context, max_s, mem, overlap=overlap_s)

    else:
        raise NotImplementedError()

    return call_model['3d'][model_type](hparams, mode, context_3d, utterance, targets)
