# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

MODEL_BASE = "models/syntaxnet/syntaxnet/models/parsey_mcparseface/"
TASK_CONTEXT = MODEL_BASE + "context.pbtxt"

"""A program to annotate a conll file with a tensorflow neural net parser."""

from pprint import pprint

import os
import os.path
import time
import tempfile
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet import sentence_pb2
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

def Tag(sess, text):
  """Tags the text"""
  task_context = TASK_CONTEXT
  feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
      gen_parser_ops.feature_size(task_context=task_context,
                                  arg_prefix="brain_tagger"))
  t = time.time()
  hidden_layer_sizes = [64]

  parser = structured_graph_builder.StructuredGraphBuilder(
    num_actions,
    feature_sizes,
    domain_sizes,
    embedding_dims,
    hidden_layer_sizes,
    gate_gradients=True,
    arg_prefix="brain_tagger",
    beam_size=8,
    max_steps=1000)

  parser.AddEvaluation(task_context,
                   1024,
                   corpus_name="direct",
                   value=text,
                   evaluation_max_steps=1000)

  parser.AddSaver(True)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, MODEL_BASE + "tagger-params")

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.variable_sink(sink_documents,
                                      corpus_name="stdout-conll",
                                      task_context=task_context)
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  num_documents = 0
  while True:
    tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents'],
    ])
    logging.info("TF DOCUMENTS: %s" % tf_documents)
    if len(tf_documents):
      num_documents += len(tf_documents)
      result = sess.run(sink, feed_dict={sink_documents: tf_documents})
      return result

    num_tokens += tf_eval_metrics[0]
    num_correct += tf_eval_metrics[1]
    if num_epochs is None:
      num_epochs = tf_eval_epochs
    elif num_epochs < tf_eval_epochs:
      break

def Parse(sess, text):
  """Parses the text"""
  task_context = TASK_CONTEXT
  feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
      gen_parser_ops.feature_size(task_context=task_context,
                                  arg_prefix="brain_parser"))
  t = time.time()
  hidden_layer_sizes = [512,512]

  parser = structured_graph_builder.StructuredGraphBuilder(
    num_actions,
    feature_sizes,
    domain_sizes,
    embedding_dims,
    hidden_layer_sizes,
    gate_gradients=True,
    arg_prefix="brain_parser",
    beam_size=8,
    max_steps=1000)

  parser.AddEvaluation(task_context,
                   1024,
                   corpus_name="direct-conll",
                   value=text,
                   evaluation_max_steps=1000)

  parser.AddSaver(True)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, MODEL_BASE + "parser-params")

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.variable_sink(sink_documents,
                                      corpus_name="stdout-conll",
                                      task_context=task_context)
  t = time.time()
  num_epochs = None
  num_tokens = 0
  num_correct = 0
  num_documents = 0
  while True:
    tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents'],
    ])
    logging.info("TF DOCUMENTS: %s" % tf_documents)
    if len(tf_documents):
      num_documents += len(tf_documents)
      result = sess.run(sink, feed_dict={sink_documents: tf_documents})
      return result

    num_tokens += tf_eval_metrics[0]
    num_correct += tf_eval_metrics[1]
    if num_epochs is None:
      num_epochs = tf_eval_epochs
    elif num_epochs < tf_eval_epochs:
      break

def Eval(text):
    tf.reset_default_graph()
    with tf.Session() as sess:
        tagged = Tag(sess, text)[0]
    tf.reset_default_graph()
    with tf.Session() as sess:
        parsed = Parse(sess, tagged)[0]

    results = []
    arr = map(lambda x: x.split("\t"), parsed.split("\n"))
    for a in arr:
        result = {}
        if len(a) == 10:
            result['id'] = a[0]
            result['form'] = a[1]
            result['lemma'] = a[2]
            result['cpostag'] = a[3]
            result['postag'] = a[4]
            result['feats'] = a[5]
            result['head'] = a[6]
            result['deprel'] = a[7]
            result['phead'] = a[8]
            result['pdeprel'] = a[9]
            results.append(result)
    return results

def main(unused_argv):
  pprint(Eval("Bills on ports and immigration were submitted by Senator Brownback, Republican of Kansas."))

if __name__ == '__main__':
  tf.app.run()
