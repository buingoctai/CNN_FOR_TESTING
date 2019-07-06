#! /usr/bin/env python
from __future__ import absolute_import
from model_cnn import data_helpers, text_cnn
import csv
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
sys.path.insert(0, '/model_cnn')
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("find_data_file", "./data_evaluate_cnn/Data.findRestaurantsByCity",
                       "Data source for the findRestaurantsByCity data.")
tf.flags.DEFINE_string(
    "greet_data_file", "./data_evaluate_cnn/Data.greet", "Data source for the greet data.")
tf.flags.DEFINE_string(
    "bye_data_file", "./data_evaluate_cnn/Data.bye", "Data source for the bye data.")
tf.flags.DEFINE_string("affirmative_data_file", "./data_evaluate_cnn/Data.affirmative",
                       "Data source for the affirmative data.")
tf.flags.DEFINE_string("negative_data_file", "./data_evaluate_cnn/Data.negative",
                       "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "",
                       "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(
        FLAGS.find_data_file, FLAGS.greet_data_file, FLAGS.bye_data_file, FLAGS.affirmative_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name(
            "dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name(
            "output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        batches = data_helpers.batch_iter(
            list(x_test), FLAGS.batch_size, 1, shuffle=False)
        for x_test_batch in batches:
            batch_scores = sess.run(
                scores, {input_x: x_test_batch, dropout_keep_prob: 1.0})

# Save score (get bacth_scores[3]) to text
out_scores = os.path.join(FLAGS.checkpoint_dir, "..", "scores.txt")
os.remove(out_scores)
for value in batch_scores[3]:
     with open(out_scores, 'a') as f:
        f.write(str(value)+' ')

# Determine confidence for each kind of intent
label_dic = {'greet': '4.0', 'findRestaurantsByCity': '3.0','bye': '2.0', 'affirmative': '1.0', 'negative': '0.0'}
import numpy as np
f13 = open("run_cnn\model\scores.txt", "r")
scores_str = (f13.read().split('\n')[0].split(' '))
scores_str.pop(5)
scores_nu = []
for score in scores_str:
    scores_nu.append(float(score))

#Apply softmax for scores_nu array: scores_nu -> scores_dic
e_Z = np.exp(scores_nu)
scores_nu = e_Z / e_Z.sum(axis=0)
score_dic = []

#Append label for every element in score_dic and sort it rely on decrease confidence
for i in range(0, 5):
    if(i == 0):
        result_str = "negative"
    elif(i == 1):
        result_str = "affirmative"
    elif(i == 2):
        result_str = "bye"
    elif(i == 3):
        result_str = "findRestaurantsByCity"
    elif(i == 4):
        result_str = "greet"
    score_dic.append({'confidence': scores_nu[i], 'key': i, 'intent': result_str})
import operator
score_dic.sort(key=operator.itemgetter('confidence'), reverse=True)
print()
print("---------------------------Print Result of evaluating--------------------------------")
intent={"name":score_dic[0]['intent'],"confidence":score_dic[0]['confidence'] }
intent_ranking=[{'name': score_dic[0]['intent'], 'confidence':score_dic[0]['confidence']}, {'name': score_dic[1]['intent'], 'confidence': score_dic[1]['confidence']}, {'name': score_dic[2]['intent'], 'confidence': score_dic[2]['confidence']}, {'name':score_dic[3]['intent'], 'confidence': score_dic[3]['confidence']}, {'name':score_dic[4]['intent'], 'confidence': score_dic[4]['confidence']}]
Message={"intent":{},"intent_ranking":[],"text":""}
Message['intent']=intent
Message['intent_ranking']=intent_ranking

f=open('data_evaluate_cnn/Data.findRestaurantsByCity','r')
Message['text']=f.read()
f.close()
print(Message)
print("")