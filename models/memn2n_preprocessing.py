import tensorflow as tf

#functions for mode "eou" and "eot"

def pad_right_with_num(input_tensor, pad_length, padding = 0):
    """pad a (?, ) tensor to (? + pad_length, ) tensor with pad_num.

    Args:
        input_tensor: The vector to pad.
        pad_length: Number of elements to add on the right.
        padding: (optional) The element to pad.

    Returns:
        Padded tensor.
    """
    input_length = tf.shape(input_tensor)[0]
    pad_nums = tf.fill([pad_length], padding)
    complement_pad_op = tf.cast(tf.pad(pad_nums, [[input_length, 0]]), dtype=tf.int64)
    pad_zero_op = tf.pad(input_tensor, [[0, pad_length]])
    return tf.add(pad_zero_op, complement_pad_op)

def to_fixed_length(input_tensor, target_length, padding = 0):
    """convert a (?,) tensor to a fixed length tensor (target_length, ) with padding.

    This function checks the length of the input vector. If it is larger than
    target_length, a slice is performed to take (0, target_length); otherwise
    the vector is padded to target_length with padding.

    Args:
        input_tensor: The vector to process.
        target_length: The length of the output vector.
        padding: (optional) The number to fill if necessary.

    Returns:
        Padded or sliced tensor.
    """
    input_length = tf.shape(input_tensor)[0]
    pad_op = pad_right_with_num(input_tensor,
                                tf.maximum(target_length - input_length, 0),
                                padding)
    #pad_op = tf.pad(input_tensor, [[0, tf.maximum(target_length - input_length, 0)]])
    chop_op = tf.strided_slice(input_tensor, [0], [target_length])
    select_op = tf.less(input_length, target_length)
    core_op = tf.cond(select_op, lambda: pad_op, lambda: chop_op)
    return tf.reshape(core_op, [target_length])

def get_position(input_tensor, target_num):
    """find all positions of target_num in input_tensor.

    Args:
        input_tensor: The vector to be looked up.
        target_num: The number to look for.

    Returns:
        A tensor of position indices.
    """
    return tf.reshape(tf.where(tf.equal(input_tensor, target_num)), [-1])

def create_indices(position_tensor, num_indices, max_length, include_target = True):
    """create a list of complete indices to slice a tensor.

    This function converts a vector of split points to pairs of start and end
    indices. It outputs fixed number of indices, (max_length, max_length) is
    padded if the position tensor is insufficient, otherwise the remaining positions
    are discarded.

    Args:
        position_tensor: The vector of positions.
        num_indices: The exact pair of indices to generate.
        max_length: The maximum length of original vector.
        include_target: (optional) Whether to include the splitting elements.

    Returns:
        A list of tensors containing pairs of indices.
    """
    processed_indices = to_fixed_length(position_tensor, num_indices, max_length)
    if include_target:
        end_indices = tf.add(processed_indices, tf.ones_like(processed_indices))
    else:
        end_indices = processed_indices

    #drop last, prepend 0, add 1
    start_indices = tf.add(tf.concat([[-1],tf.strided_slice(processed_indices, [0], [-1])], axis = 0),
                           tf.ones_like(processed_indices))
    indices = tf.transpose(tf.stack([start_indices, end_indices], axis = 0))
    return tf.unstack(indices)

def extract_and_pad(input_tensor, index, target_length, padding = 0):
    """take the input_tensor and a pair of indices, and output the sliced vector
    with fixed length.

    Args:
        input_tensor: The vector to slice.
        index: The tensor of (start_index, end_index).
        target_length: The length of the output vector.
        padding: (optional) The number to pad.

    Returns:
        Sliced and padded tensor.
    """
    sliced = tf.strided_slice(input_tensor, [index[0]], [index[1]])
    return to_fixed_length(sliced, target_length, padding)

def split_tensor(input_tensor, max_sentence_len, memory_size, target_num, include_target = True):
    """extract sentences split by specific element, and convert it into specific
    format.

    This is the main function of preprocessing mode "eou"(1) and "eot"(2). The
    input tensor contains a sequence of sentences (maybe incomplete) of varying
    size. The function picks out each sentence and creates a new tensor of
    (number of sentences, number of words per sentence).

    Args:
        input_tensor: The vector of sentences (in words).
        max_sentence_len: The exact length of each sentence in the output.
        memory_size: The number of sentences in the output.
        target_num: The deliminator (eou: 1, eot: 2).
        include_target: (optional) Whether to include the deliminator.

    Returns:
        A tensor of shape (max_sentence_len, memory_size)
    """
    target_pos = get_position(input_tensor, target_num)
    indices = create_indices(target_pos, memory_size, tf.shape(input_tensor)[0], include_target)
    return tf.stack([extract_and_pad(input_tensor, index, max_sentence_len) for index in indices])

#functions for "split" and "split_overlap"

def split_overlap(input_tensor, max_sentence_len, memory_size, overlap = None):
    """create tensor from overlapping slices of the input tensor.

    This function generates memory_size of vectors sliced from input_tensor. When
    overlap is None (or equivalently, max_sentence_len), it reduces to reshape op.
    otherwise, the next sentence starts `overlap` position of the current.

    Args:
        input_tensor: The vector of sentences (in words).
        max_sentence_len: The exact length of each sentence in the output.
        memory_size: The number of sentences in the output.
        overlap: (optional) The offset of overlapping

    Returns:
        A tensor of shape (max_sentence_len, memory_size)

    """
    input_length = tf.shape(input_tensor)[0]
    if overlap == None:
        overlap = max_sentence_len
    slice_op = tf.stack([tf.strided_slice(input_tensor,
                                          [i * overlap],
                                          [i * overlap + max_sentence_len])
                                          for i in range(memory_size)])
    return slice_op

#functions for batch preprocessing

def split_tensor_batch(input_tensor,
                       max_sentence_len,
                       memory_size,
                       target_num,
                       include_target = True):
    """split_tensor for a batch of tensors.
    """
    pure_fn = lambda x: split_tensor(x, max_sentence_len, memory_size, target_num, include_target)
    return tf.map_fn(pure_fn, input_tensor)

def split_overlap_batch(input_tensor,
                        max_sentence_len,
                        memory_size,
                        overlap = None):
    """split_overlap for a batch of tensors.
    """
    pure_fn = lambda x: split_overlap(x, max_sentence_len, memory_size, overlap)
    return tf.map_fn(pure_fn, input_tensor)
