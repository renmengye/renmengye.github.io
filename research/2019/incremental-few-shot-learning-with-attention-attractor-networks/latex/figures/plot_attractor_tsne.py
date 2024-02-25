# Modified from run_transfer_exp.py
"""Transfer + finetuning baselines.

Usage:
./run_transfer_exp.py                                                     \
                             --aug              [AUGMENT 90 DEGREE]       \
                             --shuffle_episode  [SHUFFLE EPISODE]         \
                             --nclasses_train   [NUM CLASSES TRAIN]       \
                             --nclasses_val     [NUM CLASSES VAL]         \
                             --nclasses_test    [NUM CLASSES TEST]        \
                             --nshot            [NUM SHOT]                \
                             --num_eval_episode [NUM EVAL EPISODE]        \
                             --ntest            [NUM TEST]                \
                             --num_unlabel      [NUM UNLABEL]             \
                             --seed             [RANDOM SEED]             \
                             --dataset          [DATASET NAME]

Flags:

"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import horovod.tensorflow as hvd
import numpy as np
import os
import six
import sys
import tensorflow as tf

from tqdm import tqdm
from google.protobuf.text_format import Merge, MessageToString
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

from fewshot.configs.experiment_config_pb2 import ExperimentConfig
from fewshot.data.batch_iter import BatchIterator
from fewshot.data.data_factory import get_concurrent_iterator
from fewshot.data.data_factory import get_dataset
from fewshot.data.mini_imagenet import MiniImageNetDataset
from fewshot.data.mini_imagenet2 import MiniImageNetDataset2
from fewshot.data.omniglot import OmniglotDataset
from fewshot.data.tiered_imagenet import TieredImageNetDataset
from fewshot.data.lowshot_imagenet import LowshotImageNetDataset
from fewshot.models.multi_task_model import MultiTaskModel
from fewshot.models.imprint_model import ImprintModel
from fewshot.models.transfer_model import TransferModel
from fewshot.models.transfer_model_one import TransferModelOne
from fewshot.utils import logger
from train_lib import get_metadata

log = logger.get()

flags = tf.flags
flags.DEFINE_bool("eval", False, "Whether run evaluation only")
flags.DEFINE_bool("restore_rbp", False, "Restore from RBP experiments")
flags.DEFINE_bool("shuffle_episode", False, "Shuffle the sequence order")
flags.DEFINE_bool("val", True, "Whether to run val")
flags.DEFINE_bool("test", True, "Whether to run test")
flags.DEFINE_integer("nclasses_val", 5, "Number of classes for validation")
flags.DEFINE_integer("nclasses_test", 5, "Number of classes for testing")
flags.DEFINE_integer("nclasses_train", 5, "Number of classes for training")
flags.DEFINE_integer("nclasses_a", -1, "Number of classes for pretraining")
flags.DEFINE_integer("nepisode", 600, "Number of evaluation episodes")
flags.DEFINE_integer("nshot", 1, "nshot")
flags.DEFINE_integer("ntest", 15, "Number of test images per episode")
flags.DEFINE_string("dataset", "omniglot", "Dataset name")
flags.DEFINE_string("pretrain", None, "Restore checkpoint name")
flags.DEFINE_string("results", "./results", "Save folder")
flags.DEFINE_string("tag", None, "Experiment tag")
flags.DEFINE_string("config", None, "Experiment config file")
flags.DEFINE_bool("load_pytorch_weights", False,
                  "Whether to load pytorch weights")
flags.DEFINE_bool("retest", False, "Reload everything")
flags.DEFINE_bool("dropout_classes", False, "Dropout base classes")
flags.DEFINE_bool("random_shots", False, "Random shots 1 2 3 5 10")
FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.ERROR)


def get_saver(log_folder):
  saver = tf.train.Saver()

  class Saver():

    def get_session(self, sess):
      session = sess
      while type(session).__name__ != 'Session':
        #pylint: disable=W0212
        session = session._sess
      return session

    def save(self, sess, tag):
      saver.save(
          self.get_session(sess), os.path.join(log_folder,
                                               'ckpt-{}'.format(tag)))

  return Saver()


def loginfo(message):
  if hvd.local_rank == 0:
    log.info(message)


def logerror(message):
  if hvd.local_rank == 0:
    log.error(message)


def get_iter(size,
             get_fn,
             batch_size,
             cycle=True,
             shuffle=True,
             max_queue_size=50,
             num_threads=5,
             seed=0):
  b = BatchIterator(
      size,
      batch_size=batch_size,
      cycle=True,
      shuffle=True,
      get_fn=get_fn,
      log_epoch=-1,
      seed=seed)
  if num_threads > 1:
    return get_concurrent_iterator(
        b, max_queue_size=max_queue_size, num_threads=num_threads)
  else:
    return b


def evaluate_b(sess,
               model,
               task_it,
               num_steps,
               old_and_new,
               is_chief,
               nclasses_dropout=-1,
               random_shots=False):
  """Evaluate the model on task A."""
  it = tqdm(six.moves.xrange(num_steps), ncols=0)
  feature_dict = {}
  for tt, task_data in zip(it, task_it):
    ystr = task_data.y_train_str  # [5]
    prediction_b, labels_b = model.eval_step_b(sess, task_data)
    fetches = [model.gamma, model.attended_h, model.h_b]
    fetches_val = model.eval_step_b_custom_fetch(sess, fetches, task_data)
    gamma = fetches_val[0]  # [1, D+1]
    attended_h = fetches_val[1]  # [5, D+1]
    h_b = fetches_val[2]  # [5, D]
    print('gamma', gamma.shape)
    print('attended_h', attended_h.shape)
    print('h_b', h_b.shape)
    h_b_gamma = h_b * np.sqrt(gamma[:-1, 0])  # [5, D]
    attend_gamma = attended_h[:, :-1] * np.sqrt(gamma[:-1, 0])  # [5, D]
    for ii in range(h_b_gamma.shape[0]):
      if ystr[ii] not in feature_dict:
        feature_dict[ystr[ii]] = []
      if attend_gamma.shape[0] > 1:
        # attn
        feature_dict[ystr[ii]].append((h_b_gamma[ii], attend_gamma[ii]))
      else:
        # static
        feature_dict[ystr[ii]].append((h_b_gamma[ii], attend_gamma[0]))
  print('Summary')
  for k in sorted(feature_dict.keys()):
    print(k, ':', len(feature_dict[k]))
  import pickle as pkl
  pkl.dump(feature_dict, open('attractor.pkl', 'wb'))
  # plt.get_cmap('Paired')


def visualize_attractor(data, static_data=None):
  cls = sorted(data.keys())
  import sklearn.manifold
  tsne = sklearn.manifold.TSNE(n_components=2, verbose=1, random_state=1234)
  pca = sklearn.decomposition.PCA(n_components=2)
  # [2 * C, 100, D]
  C = 5
  N = 20
  X = np.zeros([2 * C, N, data[cls[0]][0][0].shape[0]])
  print('X', X.shape)
  for idx, c in enumerate(cls[1:C + 1]):
    assert len(data[c]) >= N
    for ii in range(N):
      X[idx, ii] = data[c][ii][0]  # First one is the example feature.
      X[idx + C, ii] = data[c][ii][1]  # Second one is the attractor.
  # [2 * C * 100, D]
  X = X.reshape([2 * C * N, -1])
  if static_data is not None:
    sa = static_data[c][0][1]  # anything should be the same
    X = np.concatenate([X, np.expand_dims(sa, 0)], axis=0)
  from matplotlib import pyplot as plt
  Z = tsne.fit_transform(X)
  y1 = np.tile(np.expand_dims(2 * np.arange(C), 1), [1, N])
  y2 = np.tile(np.expand_dims(2 * np.arange(C) + 1, 1), [1, N])
  y = np.concatenate([y1, y2], axis=0).reshape([-1])
  fig = plt.figure()
  half = C * N
  cmap = 'viridis'
  plt.scatter(Z[:half, 0], Z[:half, 1], c=y[:half], cmap=cmap, alpha=1.0, s=10)
  if static_data is None:
    plt.scatter(
        Z[half:, 0], Z[half:, 1], c=y[half:], cmap=cmap, alpha=0.5, s=80)
    plt.legend(['features', 'attention attractors'])
  else:
    plt.scatter(
        Z[half:-1, 0], Z[half:-1, 1], c=y[half:], cmap=cmap, alpha=0.5, s=80)
    plt.scatter(Z[-2:-1, 0], Z[-2:-1, 0], c='red', alpha=0.5, s=80)
    plt.legend(['features', 'attention attractors', 'static attractor'])
  plt.xlim(-20, 12)
  plt.ylim(-18, 16)
  plt.savefig('attractor.pdf')


def preprocess_old_and_new(num_classes_a, task_a_it, task_b_it):
  """Combining two iterators into a single iterator. A regular B few-shot"""

  class Iterator():

    def next(self, nclasses_dropout=-1, random_shots=False):
      # Increment class indices.
      if FLAGS.random_shots:
        task_b_data = task_b_it.next(random_shots=random_shots)
      else:
        task_b_data = task_b_it.next()
      task_b_data._y_train += num_classes_a
      task_b_data._y_test += num_classes_a

      task_a_data_test = task_a_it.next(forbid=task_b_data.y_sel)
      x_test_a, y_test_a = task_a_data_test

      # Combine old and new in the validation.
      task_b_data._x_test = np.concatenate([x_test_a, task_b_data.x_test],
                                           axis=0)
      task_b_data._y_test = np.concatenate([y_test_a, task_b_data.y_test],
                                           axis=0)
      return task_b_data

    def stop(self):
      task_b_it.stop()

  return Iterator()


def get_config(config_file):
  """Reads configuration."""
  config = ExperimentConfig()
  Merge(open(config_file).read(), config)
  return config


def get_model(config,
              num_classes_a,
              nclasses_train,
              nclasses_val,
              nclasses_test,
              is_eval=False,
              load_pytorch_weights=False):
  """Builds model."""
  assert config.backbone_class == 'resnet_backbone', 'Only support ResNet'
  bb_config = config.resnet_config

  # ------------------------------------------------------------------------
  # Placeholders
  x_a = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_a')
  y_a = tf.placeholder(tf.int64, [None], name='y_a')
  x_b = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_b')
  y_b = tf.placeholder(tf.int64, [None], name='y_b')
  y_sel = tf.placeholder(tf.int64, [None], name='y_sel')

  x_b_v = tf.placeholder(
      tf.float32,
      [None, bb_config.height, bb_config.width, bb_config.num_channel],
      name='x_b_v')
  y_b_v = tf.placeholder(tf.int64, [None], name='y_b_v')

  # ------------------------------------------------------------------------
  # Model classes
  if config.model_class == "multitask":
    model_class = MultiTaskModel
  elif config.model_class == "imprint":
    model_class = ImprintModel
  elif config.model_class == "transfer":
    model_class = TransferModel
  elif config.model_class == "transfer-one":
    model_class = TransferModelOne
  else:
    raise ValueError("Unknown model")

  # ------------------------------------------------------------------------
  # Load external weights in NPZ
  if load_pytorch_weights:
    ext_wts = dict(np.load('weights.npz'))
    for k in ext_wts.keys():
      if k.startswith('w') and len(ext_wts[k].shape) == 4:
        ext_wts[k] = np.transpose(ext_wts[k], [2, 3, 1, 0])
  else:
    ext_wts = None
  with tf.name_scope('Test'):
    with tf.variable_scope('Model', reuse=tf.AUTO_REUSE):
      modelt = model_class(
          config,
          x_a,
          y_a,
          x_b,
          y_b,
          x_b_v,
          y_b_v,
          num_classes_a,
          nclasses_test,
          is_training=False,
          ext_wts=ext_wts,
          y_sel=y_sel)

  return {'test': modelt}


def get_datasets(dataset, metadata, nshot, num_test, batch_size, num_gpu,
                 nclasses_a, nclasses_train, nclasses_val, nclasses_test,
                 old_and_new, aug_90, seed, is_eval):
  """Builds datasets"""
  # ------------------------------------------------------------------------
  # Datasets
  traintest_dataset_a = get_dataset(
      dataset,
      metadata['trainsplit_a_test'],
      nclasses_train,
      nshot,
      label_ratio=metadata['label_ratio'],
      num_test=num_test // num_gpu,
      aug_90=aug_90,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed,
      image_split_file=metadata['image_split_file_a_test'],
      nclasses=nclasses_a,
      local_rank=hvd.local_rank(),
      total_rank=hvd.size())
  test_dataset = get_dataset(
      dataset,
      "test",
      nclasses_test,
      nshot,
      num_test=num_test // num_gpu,
      label_ratio=1.0,
      aug_90=aug_90,
      num_unlabel=0,
      shuffle_episode=False,
      seed=seed,
      local_rank=hvd.local_rank(),
      total_rank=hvd.size())

  # ------------------------------------------------------------------------
  # Task B iterators for old and new (wrapper)
  if old_and_new:
    task_a_test_iter_old = get_iter(
        traintest_dataset_a.get_size(),
        traintest_dataset_a.get_batch_idx,
        num_test * nclasses_test // num_gpu,
        cycle=True,
        shuffle=True,
        max_queue_size=10,
        num_threads=1,
        seed=seed + 1)

    if nclasses_a == -1:
      num_classes_a = metadata['num_classes_a']
    else:
      num_classes_a = nclasses_a
    task_b_test_iter = get_concurrent_iterator(
        test_dataset, max_queue_size=2, num_threads=1)
    task_b_test_iter = preprocess_old_and_new(
        num_classes_a, task_a_test_iter_old, task_b_test_iter)

  results = {}
  results['b_test'] = task_b_test_iter
  return results


def get_restore_saver(retest=False,
                      restore_rbp=False,
                      cosine_a=False,
                      reinit_tau=False,
                      compatible=False):
  # Restore from
  var_list = tf.global_variables()
  forbid_list = [
      'ft_step', 'w_class_b', 'b_class_b', 'Adam', 'Optimizer', 'grads_b',
      'Momentum', 'beta1_power', 'beta2_power'
  ]
  loginfo('Forbid restore list: {}'.format(forbid_list))
  condition = lambda x: not any([forbid in x.name for forbid in forbid_list])
  var_list = list(filter(condition, var_list))
  var_keys = [v.name.split(':')[0] for v in var_list]
  var_dict = dict(zip(var_keys, var_list))
  print('Restore map')
  [print('{}: {}'.format(kk, var_dict[kk].name)) for kk in var_keys]
  restore_saver = tf.train.Saver(var_dict)
  return restore_saver


def restore_model(sess,
                  model,
                  modelv,
                  restore_saver,
                  is_eval=False,
                  pretrain=None):
  """Restore model from checkpoint."""
  if pretrain is not None:
    log_folder_restore = pretrain
    loginfo('Restore from {}'.format(log_folder_restore))
    ckpt = tf.train.latest_checkpoint(log_folder_restore)
    print('Checkpoint: {}'.format(ckpt))
    if is_eval:
      modelv.initialize(sess)
    else:
      model.initialize(sess)
    restore_saver.restore(sess, ckpt)
  else:
    modelv.initialize(sess)


def main():
  # ------------------------------------------------------------------------
  # Flags
  nshot = FLAGS.nshot
  dataset = FLAGS.dataset
  nclasses_train = FLAGS.nclasses_train
  nclasses_val = FLAGS.nclasses_val
  nclasses_test = FLAGS.nclasses_test
  nclasses_a = FLAGS.nclasses_a
  num_test = FLAGS.ntest
  is_eval = FLAGS.eval
  nepisode = FLAGS.nepisode
  run_val = FLAGS.val
  run_test = FLAGS.test
  load_pytorch_weights = FLAGS.load_pytorch_weights
  pretrain = FLAGS.pretrain
  restore_rbp = FLAGS.restore_rbp
  retest = FLAGS.retest
  tag = FLAGS.tag

  # ------------------------------------------------------------------------
  # Configuration
  config = get_config(FLAGS.config)
  opt_config = config.optimizer_config
  old_and_new = config.transfer_config.old_and_new

  hvd.init()
  is_chief = False
  if opt_config.num_gpu > 1:
    if hvd.local_rank() == 0:
      is_chief = True
  else:
    is_chief = True

  # ------------------------------------------------------------------------
  # Log folder
  assert tag is not None, 'Please add a name for the experiment'
  log_folder = os.path.join(FLAGS.results, dataset, 'n{}w{}'.format(
      FLAGS.nshot, FLAGS.nclasses_val), tag)
  loginfo('Experiment ID {}'.format(tag))
  if os.path.exists(log_folder) and not FLAGS.eval:
    assert False, 'Folder {} exists. Pick another tag.'.format(log_folder)

  # ------------------------------------------------------------------------
  # Model
  metadata = get_metadata(dataset)
  if nclasses_a == -1:
    num_classes_a = metadata['num_classes_a']
  else:
    num_classes_a = nclasses_a
    log.info('Use total number of classes = {}'.format(num_classes_a))
  with log.verbose_level(2):
    model_dict = get_model(
        config,
        num_classes_a,
        nclasses_train,
        nclasses_val,
        nclasses_test,
        is_eval=is_eval,
        load_pytorch_weights=load_pytorch_weights)
    modelt = model_dict['test']

  # ------------------------------------------------------------------------
  # Dataset
  if opt_config.num_gpu > 1:
    seed = hvd.rank() * 1000
  else:
    seed = 0

  with log.verbose_level(2):
    data = get_datasets(dataset, metadata, nshot, num_test,
                        opt_config.batch_size, opt_config.num_gpu, nclasses_a,
                        nclasses_train, nclasses_val, nclasses_test,
                        old_and_new, False, seed, is_eval)

  # ------------------------------------------------------------------------
  # Log outputs
  restore_saver = get_restore_saver(
      retest=retest,
      restore_rbp=restore_rbp,
      cosine_a=modelt.config.protonet_config.cosine_a,
      reinit_tau=modelt.config.protonet_config.reinit_tau)
  saver = get_saver(log_folder)

  # ------------------------------------------------------------------------
  # Create a TensorFlow session
  sess_config = tf.ConfigProto()
  sess_config.gpu_options.allow_growth = True
  if opt_config.num_gpu > 1:
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]
    sess_config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.train.MonitoredTrainingSession(
        checkpoint_dir=log_folder,
        config=sess_config,
        hooks=hooks,
        # save_checkpoint_steps=config.train_config.steps_per_save,
        save_checkpoint_steps=None,
        save_summaries_secs=None)
  else:
    hooks = []
    sess = tf.Session(config=sess_config)

  # ------------------------------------------------------------------------
  # Initialize model
  restore_model(
      sess, modelt, modelt, restore_saver, is_eval=is_eval, pretrain=pretrain)

  # ------------------------------------------------------------------------
  # Testing
  loginfo('Experiment ID {}'.format(tag))
  import pickle as pkl
  if not os.path.exists('data.pkl'):
    data = [data['b_test'].next() for i in range(nepisode)]
    pkl.dump(data, open('data.pkl', 'wb'))
  else:
    data = pkl.load(open('data.pkl', 'rb'))
  # Populate attractor features.
  if not os.path.exists('attractor.pkl') and not (
      os.path.exists('attn-attractor-2.pkl') and
      os.path.exists('static-attractor-2.pkl')):
    evaluate_b(
        sess,
        modelt,
        data,
        nepisode,
        old_and_new,
        is_chief,
        random_shots=FLAGS.random_shots)

  import pickle as pkl
  attn_attractor = pkl.load(open('attn-attractor-2.pkl', 'rb'))
  static_attractor = pkl.load(open('static-attractor-2.pkl', 'rb'))
  visualize_attractor(attn_attractor, static_attractor)


if __name__ == '__main__':
  main()
