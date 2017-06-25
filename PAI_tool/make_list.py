# coding=utf-8
import oss2

auth = oss2.Auth('LTAIbHBeGsbrDmqv', 'KuwKLM7eqJN4BI4bSpoxBg8y4WrVcy')
bucket = oss2.Bucket(auth=auth, endpoint='oss-cn-shanghai.aliyuncs.com', bucket_name='mlearn')

file = open('file_list.txt', 'a')

for key, b in enumerate(oss2.ObjectIterator(bucket, marker='classify/')):
    file.write("{}/{} {}\r\n".format("mlearn", b.key, key))


# for root, path, filenames in os.walk('../../cloth_classifier/classify'):
#     for key, filename in enumerate(filenames):
#
