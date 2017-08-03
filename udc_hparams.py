import tensorflow as tf
from collections import namedtuple

# Model Parameters
tf.flags.DEFINE_integer(
  "vocab_size",
  91620,
  "The size of the vocabulary. Only change this if you changed the preprocessing")

# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer("rnn_dim", 128, "Dimensionality of the RNN cell")
tf.flags.DEFINE_integer("max_context_len", 160, "Truncate contexts to this length")
tf.flags.DEFINE_integer("max_utterance_len", 80, "Truncate utterance to this length")

#----------------New Experiments

#memn2n model params
tf.flags.DEFINE_integer("nhop", 6, "number of hops [6]")
tf.flags.DEFINE_float("init_std", 0.05, "weight initialization std [0.05]")

#two types of the model according to the paper
#Possible options: "adjacent" or "layer_wise"
tf.flags.DEFINE_string("model_type", "adjacent", "type of the model")
#tf.flags.DEFINE_string("model_type", "layer_wise", "type of the model")

#preprocessing mode, done at RUN TIME
#Possible options: None, "eou", "eot", "split", "split_overlap"
tf.flags.DEFINE_string("preprocessing_mode", "eot", "mode of preprocessing")

#TODO: Add params for max_sentense_len



#----------------End



# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", None, "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", None, "Path to vocabulary.txt file")

# Training Parameters
tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
tf.flags.DEFINE_integer("batch_size", 128, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "Batch size during evaluation")
tf.flags.DEFINE_string("optimizer", "Adam", "Optimizer Name (Adam, Adagrad, etc)")

FLAGS = tf.flags.FLAGS

#TODO: register new hparams
HParams = namedtuple(
  "HParams",
  [
    "batch_size",
    "embedding_dim",
    "eval_batch_size",
    "learning_rate",
    "max_context_len",
    "max_utterance_len",
    "optimizer",
    "rnn_dim",
    "vocab_size",
    "glove_path",
    "vocab_path",
    "init_std",
    "nhop"
  ])

def create_hparams():
  return HParams(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    vocab_size=FLAGS.vocab_size,
    optimizer=FLAGS.optimizer,
    learning_rate=FLAGS.learning_rate,
    embedding_dim=FLAGS.embedding_dim,
    max_context_len=FLAGS.max_context_len,
    max_utterance_len=FLAGS.max_utterance_len,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    rnn_dim=FLAGS.rnn_dim,
    init_std=FLAGS.init_std,
    nhop=FLAGS.nhop)
