#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import tensorflow.python.platform

import numpy as np
from six.moves import xrange
import tensorflow as tf

import data_utils as data_utils
import seq2seq_model
from tensorflow.python.platform import gfile
import MeCabFuji as mf


tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 20000, "input vocabulary size.")
tf.app.flags.DEFINE_integer("out_vocab_size", 24000, "output vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "datas", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "datas", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("self_test", False,
                            "Run a self-test if this is set to True.")

FLAGS = tf.app.flags.FLAGS

_buckets = [(10, 20), (15, 30), (30, 60), (40, 80)]


def read_data(read_from_index, read_to_index, source_path, target_path, max_size=None):
  print("resdStart " + source_path)
  data_set = [[] for _ in _buckets]
  source_file = open(source_path,"r")
  target_file = open(target_path,"r")

  source, target = source_file.readline(), target_file.readline()
  counter = 0
  counter2 = 0
  print("read range is " + str(read_from_index) + " - " + str(read_to_index))
  while source and target and (not max_size or counter < max_size):
    counter += 1
    # カウンターがfromより小さいまたは、カウンターがtoより大きい -> 範囲外
    if counter < read_from_index:
      source, target = source_file.readline(), target_file.readline()
      continue
    if read_to_index < counter:
      break

    if counter % 50 == 0:
      print("  reading data line %d" % counter)
      sys.stdout.flush()

    source_ids = [int(x) for x in source.split()]
    target_ids = [int(x) for x in target.split()]
    target_ids.append(data_utils.EOS_ID)
    isHit = False
    for bucket_id, (source_size, target_size) in enumerate(_buckets):
      if len(source_ids) < source_size and len(target_ids) < target_size:
        data_set[bucket_id].append([source_ids, target_ids])
        isHit = True
        break
    if not isHit:
      print("noHit : " + str(len(source_ids)) + " / " + str(len(target_ids)))
      counter2 += 1
    source, target = source_file.readline(), target_file.readline()
  print("noHitCount is " + str(counter2))
  for i, bucket in enumerate(data_set):
    print(str(i) + " size is " + str(len(bucket)))

  return data_set

def create_model(session, forward_only):
  print("create_model")
  model = seq2seq_model.Seq2SeqModel(
      FLAGS.in_vocab_size, FLAGS.out_vocab_size, _buckets,
      FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
      FLAGS.learning_rate, FLAGS.learning_rate_decay_factor,
      forward_only=forward_only)
  print("created")
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train(read_from_index, read_to_index):
  print("train arv")
  print("Preparing data in %s" % FLAGS.data_dir)
  # in_train, out_train, in_dev, out_dev, _, _ = data_utils.prepare_wmt_data(
  #     FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
  in_train = os.path.join('datas', 'train_data_ids_in.txt')
  out_train = os.path.join('datas', 'train_data_ids_out.txt')
  in_dev = os.path.join('datas', 'test_data_ids_in.txt')
  out_dev = os.path.join('datas', 'test_data_ids_out.txt')
  _ = os.path.join('datas', 'vocab_in.txt')
  _ = os.path.join('datas', 'vocab_out.txt')


  with tf.Session() as sess:


    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    model = create_model(sess, False)

    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(1, 200000, in_dev, out_dev)
    train_set = read_data(read_from_index, read_to_index, in_train, out_train, FLAGS.max_train_data_size)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))


    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    
    random_number_01 = np.random.random_sample()
    bucket_id = min([i for i in xrange(len(train_buckets_scale))
                     if train_buckets_scale[i] > random_number_01])

    start_time = time.time()
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        train_set, bucket_id)

    _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                 target_weights, bucket_id, False)
    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
    loss += step_loss / FLAGS.steps_per_checkpoint
    current_step += 1


    if current_step % FLAGS.steps_per_checkpoint == 0:

        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
       #   if model.learning_rate * FLAGS.learning_rate_decay_factor > 0.001:
            sess.run(model.learning_rate_decay_op)

        previous_losses.append(loss)

        checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        for bucket_id in xrange(len(_buckets)):
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
        if perplexity < 2:
          print("perplexity < 2 break")
          
        if model.learning_rate.eval() < 0.01:
          sess.run(model.learning_rate_revart_op)


def decode():
  with tf.Session() as sess:
    print ("Hello!!")
    model = create_model(sess, True)
    model.batch_size = 1

    in_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_in.txt")
    out_vocab_path = os.path.join(FLAGS.data_dir,
                                 "vocab_out.txt" )

    in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
    _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)


    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:

      # print("【DEBUG】 sentence : " + sentence)
      text = mf.mecab(sentence)
      print("【DEBUG】 mecabed_sentence : " + text)

      token_ids = data_utils.sentence_to_token_ids(text, in_vocab)
      print("【DEBUG】 token_ids : " + str(token_ids) + " / len : " + str(len(token_ids)))

      bucket_id = min([b for b in xrange(len(_buckets))
                   if _buckets[b][0] > len(token_ids)])
      print("【DEBUG】 use_bucket : " + str(bucket_id))

      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)

      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)

      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]

      print(" ".join([rev_out_vocab[output] for output in outputs]))
      print("> ", end="")
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():

  with tf.Session() as sess:
    print("Self-test for neural translation model.")

    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())


    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  print("defMain")
  if FLAGS.self_test:
    self_test()
  elif FLAGS.decode:
    decode()
  else:
    # メモリエラーになるので、件数を分割して読み込ませる
    count = len(open(os.path.join('datas', 'train_data_ids_in.txt'), 'r').readlines())
    loop_count = 1
    read_row = 200000
    # 一回当たりの読み込み行数から、ループ回数を求める。
    if count % read_row == 0:
        # ぴったり割り切れた場合はそのまま
        loop_count = count / read_row
    else:
        # 小数点以下切り捨てなので、 + 1して全行カバーできるループ回数にしておく
        loop_count = (count / read_row) + 1

    counter = 0
    while counter < loop_count:
        # from : (カウンター * 読込行数) +1, to: カウンター+1 * 読込行数
        # 例 読込件数:10000件の場合
        #    1回目 : from=0*10000+1=1, to=(0+1)*10000=10000 -> 1～10000
        #    2回目 : from=1*10000+1=10001, to=(1+1)*10000=20000 -> 10001～20000
        print("go train")
        train((counter * read_row) + 1, ((counter + 1) * read_row))
        counter += 1


dec_session = None
dec_model = None
in_vocab = None
rev_out_vocab = None

def decode_main(sentence):
  print("decode main")
  
  print ("Hello!!")

  text = mf.mecab(sentence)
  print("【DEBUG】 mecabed_sentence : " + text)

  token_ids = data_utils.sentence_to_token_ids(text, in_vocab)
  print("【DEBUG】 token_ids : " + str(token_ids) + " / len : " + str(len(token_ids)))

  bucket_id = min([b for b in xrange(len(_buckets))
               if _buckets[b][0] > len(token_ids)])
  print("【DEBUG】 use_bucket : " + str(bucket_id))

  encoder_inputs, decoder_inputs, target_weights = dec_model.get_batch(
      {bucket_id: [(token_ids, [])]}, bucket_id)

  _, _, output_logits = dec_model.step(dec_session, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, True)

  outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

  if data_utils.EOS_ID in outputs:
    outputs = outputs[:outputs.index(data_utils.EOS_ID)]

  print(" ".join([rev_out_vocab[output] for output in outputs]))
  return " ".join([rev_out_vocab[output] for output in outputs])

def init_main():
  print("decode main")
  global dec_model
  global in_vocab
  global rev_out_vocab
  global dec_session
  dec_session = tf.Session()
  print ("Hello!!")
  print("init aaa")
  dec_model = create_model(dec_session, True)
  dec_model.batch_size = 1
  in_vocab_path = os.path.join(FLAGS.data_dir,
                               "vocab_in.txt")
  out_vocab_path = os.path.join(FLAGS.data_dir,
                               "vocab_out.txt" )
  print(in_vocab_path)
  in_vocab, _ = data_utils.initialize_vocabulary(in_vocab_path)
  _, rev_out_vocab = data_utils.initialize_vocabulary(out_vocab_path)
  print("init end ")

if __name__ == "__main__":
  print("run main")
  tf.app.run()
