import os
import time
import itertools
import sys
import numpy as np
import tensorflow as tf
import udc_model
import udc_hparams
import udc_metrics
import udc_inputs
from models.memn2n import memn2n_model
from models.helpers import load_vocab

tf.flags.DEFINE_string("model_dir", None, "Directory to load model checkpoints from")
tf.flags.DEFINE_string("vocab_processor_file", "./data/vocab_processor.bin", "Saved vocabulary processor file")
tf.flags.DEFINE_string("vocab", "./data/vocabulary.txt", "Vocabulary file")
FLAGS = tf.flags.FLAGS

if not FLAGS.model_dir:
  print("You must specify a model directory")
  sys.exit(1)

def tokenizer_fn(iterator):
  return (x.split(" ") for x in iterator)

# Create vocabulary ourselves or load saved one
if not FLAGS.vocab_processor_file:
  vp = tf.contrib.learn.preprocessing.VocabularyProcessor(100000)
  vp.fit(open(FLAGS.vocab_processor_file))
  vp.save('./data/vocab_processor.bin')
else:
  vp = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
    FLAGS.vocab_processor_file)

# Load your own data here
INPUT_CONTEXT = "hi"
POTENTIAL_RESPONSES = ["hello", "goodbye", "maybe"]

def get_features(context, utterance):
  context_matrix = np.array(list(vp.transform([context])))
  utterance_matrix = np.array(list(vp.transform([utterance])))
  context_len = len(context.split(" "))
  utterance_len = len(utterance.split(" "))
  features = {
    "context": tf.convert_to_tensor(context_matrix, dtype=tf.int64),
    "context_len": tf.constant(context_len, shape=[1,1], dtype=tf.int64),
    "utterance": tf.convert_to_tensor(utterance_matrix, dtype=tf.int64),
    "utterance_len": tf.constant(utterance_len, shape=[1,1], dtype=tf.int64),
  }
  return features, None

if __name__ == "__main__":
  hparams = udc_hparams.create_hparams()
  model_fn = udc_model.create_model_fn(hparams, model_impl=memn2n_model)
  estimator = tf.contrib.learn.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir)

  print("Context: {}".format(INPUT_CONTEXT))
  for r in POTENTIAL_RESPONSES:
    prob = estimator.predict(input_fn=lambda: get_features(INPUT_CONTEXT, r))
    print("{}: {}".format(r, prob.next()[0]))
