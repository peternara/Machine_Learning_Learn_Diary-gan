# coding=utf-8
import tensorflow as tf
import tensorflow.contrib.slim as slim
import os


def init_fn(FLAGS):
    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def variable_to_train(FLAGS):
    """Returns a list of variables to train.

    Returns:
      A list of variables to train by the optimizer.
    """
    if FLAGS.trainable_scopes is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def get_provider(FLAGS):
    keys_to_features = {
        'image': tf.FixedLenFeature((), tf.string, default_value=''),
        'label': tf.FixedLenFeature(tf.FixedLenFeature((), tf.string, default_value=''))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[FLAGS.width, FLAGS.height, 3], channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[80]),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources='*.tfrecords',
        reader=tf.WholeFileReader,
        decoder=decoder,
        num_samples=FLAGS.num_samples,
        num_classes=FLAGS.num_classes,
        items_to_descriptions={
            'image': '图片',
            'label': 'one_hot编码label'
        },
        labels_to_names=None)

    return slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=FLAGS.num_readers,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=10 * FLAGS.batch_size)
