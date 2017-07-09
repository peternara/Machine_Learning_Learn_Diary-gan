# coding: utf8
import tensorflow as tf
import os

tf.app.flags.DEFINE_string('buckets', './test', '图片文件夹')
tf.app.flags.DEFINE_string("checkpointDir", "./", "模型保存路径")

flags = tf.app.flags.FLAGS
with tf.gfile.FastGFile(os.path.join(flags.checkpointDir, 'output_graph.pb'), 'rb') as f:
    grapf_def = tf.GraphDef()
    grapf_def.ParseFromString(f.read())
    final_result_tensor, jpeg_data_tensor, keep_prop = \
        tf.import_graph_def(grapf_def, name='',
                            return_elements=[
                                'final_result:0',
                                'DecodeJpeg/contents:0',
                                'input/keep_prob:0'
                            ])

sess = tf.Session()

result = tf.arg_max(final_result_tensor, 1)

test_image = tf.gfile.Glob(os.path.join(flags.buckets, '*.jpg'))

output_file = tf.gfile.GFile(os.path.join(flags.checkpointDir, 'result.txt'), 'wb')
output_label = tf.gfile.GFile(os.path.join(flags.checkpointDir, 'output_labels.txt'), 'r')
label = []
for line in output_label:
    label.append(int(line))

for key, filename in enumerate(test_image):
    image_id = os.path.basename(filename).split('.')[0]
    image = tf.gfile.FastGFile(filename, 'rb').read()
    predict = sess.run(result, {
        jpeg_data_tensor: image,
        keep_prop: [1]
    })
    output_name = "{}\t{}\n".format(label[int(predict[0])], image_id)
    print "step: {}, output_name: {}".format(key, output_name)
    output_file.write(output_name)
