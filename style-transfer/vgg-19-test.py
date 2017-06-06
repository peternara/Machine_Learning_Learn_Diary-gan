import scipy.io as io

vggnet = io.loadmat('./imagenet-vgg-verydeep-19.mat')

print vggnet['layers']
