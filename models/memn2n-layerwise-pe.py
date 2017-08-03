import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

#from memn2n, babi implementation
def position_encoding(sentence_size, embedding_size):
    """
    Position Encoding described in section 4.1 [1]
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

def memn2n_model(
    hparams,
    mode,
    context,
    context_len,
    utterance,
    utterance_len,
    targets):

    #the function has to take the same params as the existing model

    #the following tf tensors are int64
    #context: (batch_size, context_len)
    #context_len: (batch_size,)
    #utterance: (batch_size, utterance_len)
    #utterance_len: (batch_size, )
    #targets: (batch_size, 1)


    #get parameters

    #number of distinct words, will not be counted programmatically since given
    vocab_size = hparams.vocab_size

    #standard deviation of random values for initialization
    init_std = hparams.init_std

    #the number of layers(hops)
    nhop = hparams.nhop

    #embedding dimension
    edim = hparams.embedding_dim

    pe_c = position_encoding(160, edim)
    pe_u = position_encoding(160, edim)

    #memory size
    #since we only have one sentence in the dataset, this parameter is ignored

    #build memory

    #A is the embedding matrix for in memory representation, will be looked up for context
    A = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='A')

    #C is the embedding matrix for out memory representation, will be looked up for context
    C = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='C')

    #alternatively, use multiple embedding matrix C, used by bAbi implementation
    # C = []
    # for idx in nhop:
    #    with tf.variable_scope('hop_{}'.format(idx)) as vs:
    #        C.append(tf.Variable(C, name="C"))

    #B is the embedding matrix for utterance
    B = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='B')

    #final output matrix W, (embedding_size, output_dim)
    W = tf.Variable(tf.random_normal([edim, 1], stddev=init_std), name="W")

    #log those variables
    tf.summary.histogram('A',A)
    tf.summary.histogram('B',B)
    tf.summary.histogram('C',C)
    tf.summary.histogram('W',W)


    #hidden_state
    hidden_state = []

    with tf.variable_scope('memn2n_model'):

        #look up context in embedding matrix A, (batch_size, context_len, embedding_size) and C
        context_embedded_A = tf.nn.embedding_lookup(A, context, name='context_embedded_A')
        context_embedded_C = tf.nn.embedding_lookup(C, context, name='context_embedded_C')

        #look up utterance in embedding matrix B, (batch_size, utterance_len, embedding_size)
        utterance_embedded = tf.nn.embedding_lookup(A, utterance, name='utterance_embedded')

        #sum up every word in utterance, (batch_size, embedding_size)
        reduced_utterance = tf.reduce_sum(utterance_embedded * pe_u, 1, name='reduced_utterance')

        #reduced utterance is the first internal state, [(batch_size, embedding_size)]
        hidden_state.append(reduced_utterance)

        for n in range(nhop):

            with tf.variable_scope('hop_{}'.format(n)) as vs:

                #convert (batch_size, embedding_size) to (batch_size, 1, embedding_size)
                hidden_state_3d = tf.expand_dims(hidden_state[-1], 1, name='hidden_state_3d')

                tf.summary.histogram('hidden_state_3d', hidden_state_3d)

                #convert (batch_size, context_len, embedding_size) to (batch_size, embedding_size, context_len)
                context_embedded_A_transposed = tf.transpose(context_embedded_A * pe_c, perm=(0,2,1), name='context_embedded_A_transposed')

                tf.summary.histogram('context_embedded_A_transposed', context_embedded_A_transposed)

                #inner product of internal state and embedded context
                #(batch_size, 1, embedding_size) * (batch_size, embedding_size, context_len)
                #    => (batch_size, 1, context_len)
                m_i_3d = tf.matmul(hidden_state_3d, context_embedded_A_transposed, name='m_i_3d')

                #convert (batch_size, 1, context_len) to (batch_size, context_len)
                m_i = tf.squeeze(m_i_3d, name='m_i')

                tf.summary.histogram('m_i', m_i)

                #apply softmax layer, get p_i, (batch_size, context_len)
                p_i = tf.nn.softmax(m_i, name='p_i')

                tf.summary.histogram('p_i', p_i)

                #convert p_i (batch_size, context_len) to (batch_size, 1, context_len)
                p_i_3d = tf.expand_dims(p_i, 1, name='p_i_3d')

                #inner product of p_i 3d and embedded C
                #calculates weighted sum of c_i with weight p_i
                #(batch_size, 1, context_len) * (batch_size, context_len, embedding_size)
                #    => (batch_size, 1, embedding_size)
                o_3d = tf.matmul(p_i_3d, context_embedded_C * pe_c, name='o_3d')

                #convert (batch_size, 1, embedding_size) to (batch_size, embedding_size)
                o = tf.squeeze(o_3d, name='o')

                tf.summary.histogram('o', o)

                ##calculates the sum of o and hidden_state
                #both tensors are (batch_size, embedding_size)
                new_state = tf.add(hidden_state[-1] , o, name='new_state')

                tf.summary.histogram('new_state', new_state)

                #add to hidden_state
                hidden_state.append(new_state)

        #output after the last layer, one-hot category
        #(batch_size, embedding_size) * (embedding_size, 2) => (batch_size, 2)
        logits = tf.matmul(hidden_state[-1], W, name='logits')

        tf.summary.histogram('logits', logits)

        #to probabilities
        probs = tf.sigmoid(logits, name='final_probs')

        if mode == tf.contrib.learn.ModeKeys.INFER:
            return probs, None


        #calculate loss
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(targets), logits=logits, name='loss')

        tf.summary.histogram('losses', losses)

        mean_loss = tf.reduce_mean(losses, name="mean_loss")

        return probs, mean_loss
