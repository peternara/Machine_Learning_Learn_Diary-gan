[X] First_cnn_trial/learn.py
---------------
MNIST_identification with CNN

**accuracy rate = 99.13%**

[ ] tensorlayer || keras || sonnet || tflearn
---
figure out which is the best suit me

**Not Started**

[X] k-means/kmeans.py
---
tensorflow K-means algorithm implement

![](https://github.com/wqj97/Machine_Learning_Learn_Diary/blob/master/image/K-means-base.png)

[X] First_cnn_trial/MNIST_CNN_K-means.py
---
MNIST identification with CNN then using K-means to clustering

**Still can't understand how to use K-means to build reduce fuction**

[x] tensorflow_example: examples/cfair10
---

Rewrite Cifar10 with newest tensorflow API (1.2.0-rc1) and create
a pull request to google, hope they will merge my code

[#10349](https://github.com/tensorflow/tensorflow/pull/10349)

read steps
- [x] tf.Reader
- [x] tf.image
- [x] tf.train.string_input_producer
- [x] tf.get_variable tf.Variable tf.scope
- [x] weight_decay (conceptual)
- [x] weight_decay (coding)

> 1. tf.truncated_normal
> 2. tf.l2_loss
> 3. tf.multiply(step 1, step 2)
> 4. tf.add_to_collection
> 5. use collection

- [x] bias_add
- [x] loss_function
- [x] train

[X] cloth classifier -- My first neuron network design by myself
---

![cloth](https://github.com/wqj97/Machine_Learning_Learn_Diary/blob/master/image/T022.jpg)
is cloth

![Not cloth](https://github.com/wqj97/Machine_Learning_Learn_Diary/blob/master/image/F045.jpg)
not cloth

Accuracy: 0.86

Add a drop out layer

Accuracy: 0.91

Fixed a predict problem

Accuracy: 0.93

Figuring out how to optimize

[X] Batch normalization
---

Use Batch normalization to optimize predict Accuracy


[X] GAN. Generative Adversarial Network
---

Trying to understand
1. [X] gan
2. [X] dc-gan
3. [X] ac-gan( basic-usage )
4. [X] ac-gan with cnn

[ ] Style transfer
---

1. [X] Understanding transfer theory
2. [X] Basic usage
3. [ ] Understanding fast style transfer theory
4. [X] Fast Style Transfer usage

[ ] Baidu classify competition
---

1. [X] train_sets download -> me
2. [] coding -> me
3. [] make test_sets -> me and teammate
4. [] optimize

[x] Cloth Generator with GC-GAN
---
running in
<http://59.67.107.117:3028/>

[X] Aliyun machine learning platform script
1. [X] image resizer
