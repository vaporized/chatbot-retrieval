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

#params for the word separated models
#calculated by scripts.memn2n_hparams_estimate.find_split

#data for "eot" mode
tf.flags.DEFINE_integer("max_sentence_len_split_eot", 40, "max sentence len in split mode")
tf.flags.DEFINE_integer("memory_size_split_eot", 4, "num of sentence")

#data for "eou" mode
tf.flags.DEFINE_integer("max_sentence_len_split_eou", 39, "max sentence len in split mode")
tf.flags.DEFINE_integer("memory_size_split_eou", 6, "num of sentence")

#params for the length separated models

#plain non-overlapping
tf.flags.DEFINE_integer("max_sentence_len_split", 20, "max sentence len in split mode")
tf.flags.DEFINE_integer("memory_size_split", 8, "num of sentence")

#with overlapping
tf.flags.DEFINE_integer("max_sentence_len_overlap", 20, "max sentence len in split mode")
tf.flags.DEFINE_integer("overlapping_size", 10, "size of overlapping")
tf.flags.DEFINE_integer("memory_size_overlap", 15, "num of sentence")




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
    "nhop",
    "model_type",
    "preprocessing_mode",
    "max_sentence_len_split_eot",
    "memory_size_split_eot",
    "max_sentence_len_split_eou",
    "memory_size_split_eou",
    "max_sentence_len_split",
    "memory_size_split",
    "max_sentence_len_overlap",
    "overlapping_size",
    "memory_size_overlap"
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
    nhop=FLAGS.nhop,
    model_type=FLAGS.model_type,
    preprocessing_mode=FLAGS.preprocessing_mode,
    max_sentence_len_split_eot=FLAGS.max_sentence_len_split_eot,
    memory_size_split_eot=FLAGS.memory_size_split_eot,
    max_sentence_len_split_eou=FLAGS.max_sentence_len_split_eou,
    memory_size_split_eou=FLAGS.memory_size_split_eou,
    max_sentence_len_split=FLAGS.max_sentence_len_split,
    memory_size_split=FLAGS.memory_size_split,
    max_sentence_len_overlap=FLAGS.max_sentence_len_overlap,
    overlapping_size=FLAGS.overlapping_size,
    memory_size_overlap=FLAGS.memory_size_overlap
    )
