import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def memn2n_model_3d_adjacent(
    hparams,
    mode,
    context,
    utterance,
    targets):
    """the 3d memn2n model in adjacent mode.

    Args:
        hparams: Hyper-parameters of the model, including training scheme.
        mode: Indicator of training or validation
        context: Tensor of shape (batch_size, memory_size, sentence_len)
        utterance: Tensor of shape (batch_size, utterance_len)
        targets: Tensor of shape (batch_size, 1)

    Returns:
        (probs, mean_loss) for training mode, or (probs, None) for validation

    """


    #extract parameters

    #number of distinct words, will not be counted programmatically since given
    #the number - word correspondence is in data/vocabulary.txt
    vocab_size = hparams.vocab_size

    #standard deviation of random values for initialization
    init_std = hparams.init_std

    #the number of layers(hops)
    nhop = hparams.nhop

    #embedding dimension
    edim = hparams.embedding_dim


    #build memory

    #A is the embedding matrix for in memory representation, will be looked up for context
    A = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='A')

    #C is the embedding matrix for out memory representation, will be looked up for context
    #C = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='C')

    #alternatively, use multiple embedding matrix C, used by bAbi implementation
    C = []
    for idx in range(nhop):
       with tf.variable_scope('hop_{}'.format(idx)) as vs:
           C.append(tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name="C"))

    #B is the embedding matrix for utterance
    #B = tf.Variable(tf.random_normal([vocab_size, edim], stddev=init_std), name='B')

    #final output matrix W, (embedding_size, output_dim)
    W = tf.Variable(tf.random_normal([edim, 1], stddev=init_std), name="W")

    #log those variables
    tf.summary.histogram('A',A)
    #tf.summary.histogram('B',B)
    tf.summary.histogram('C',C)
    tf.summary.histogram('W',W)


    #hidden_state
    hidden_state = []

    with tf.variable_scope('memn2n_model'):

        #look up utterance in embedding matrix B, (batch_size, utterance_len, embedding_size)
        utterance_embedded = tf.nn.embedding_lookup(A, utterance, name='utterance_embedded')

        #sum up every word in utterance, (batch_size, embedding_size)
        reduced_utterance = tf.reduce_sum(utterance_embedded, 1, name='reduced_utterance')

        #reduced utterance is the first internal state, [(batch_size, embedding_size)]
        hidden_state.append(reduced_utterance)

        for n in range(nhop):

            with tf.variable_scope('hop_{}'.format(n)) as vs:

                if n==0:

                    #(batch_size, memory_size, sentence_len, embedding_size)
                    embedded_context_A = tf.nn.embedding_lookup(A, context, name='embedded_context_A')
                    #(batch_size, memory_size, embedding_size)
                    reduced_context_A = tf.reduce_sum(embedded_context_A, 2, name='reduced_context_A')

                else:
                    with tf.variable_scope('hop_{}'.format(n - 1)):
                        embedded_context_A = tf.nn.embedding_lookup(C[n-1], context, name='embedded_context_A')
                        reduced_context_A = tf.reduce_sum(embedded_context_A, 2, name='reduced_context_A')

                #convert (batch_size, embedding_size) to (batch_size, 1, embedding_size)
                hidden_state_3d = tf.expand_dims(hidden_state[-1], 1, name='hidden_state_3d')

                tf.summary.histogram('hidden_state_3d', hidden_state_3d)

                #convert (batch_size, memory_size, embedding_size) to (batch_size, embedding_size, memory_size)
                reduced_context_A_transposed = tf.transpose(reduced_context_A, perm=(0,2,1), name='reduced_context_A_transposed')

                tf.summary.histogram('reduced_context_A_transposed', reduced_context_A_transposed)

                #inner product of internal state and embedded context
                #(batch_size, 1, embedding_size) * (batch_size, embedding_size, memory_size)
                #    => (batch_size, 1, memory_size)
                m_i_3d = tf.matmul(hidden_state_3d, reduced_context_A_transposed, name='m_i_3d')

                #convert (batch_size, 1, memory_size) to (batch_size, memory_size)
                m_i = tf.squeeze(m_i_3d, name='m_i')

                tf.summary.histogram('m_i', m_i)

                #apply softmax layer, get p_i, (batch_size, memory_size)
                p_i = tf.nn.softmax(m_i, name='p_i')

                tf.summary.histogram('p_i', p_i)

                #convert p_i (batch_size, memory_size) to (batch_size, 1, memory_size)
                p_i_3d = tf.expand_dims(p_i, 1, name='p_i_3d')

                with tf.variable_scope('hop_{}'.format(n)):
                    embedded_context_C = tf.nn.embedding_lookup(C[n], context, name='embedded_context_C')
                    reduced_context_C = tf.reduce_sum(embedded_context_C, 2, name='reduced_context_C')

                #inner product of p_i 3d and embedded C
                #calculates weighted sum of c_i with weight p_i
                #(batch_size, 1, memory_size) * (batch_size, memory_size, embedding_size)
                #    => (batch_size, 1, embedding_size)
                o_3d = tf.matmul(p_i_3d, reduced_context_C, name='o_3d')

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
