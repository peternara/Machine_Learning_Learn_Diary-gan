import oss2
import sys

with open('main.py') as file:
    flags = False
    for line in file:
        if line == "tf.flags.DEFINE_boolean('is_train', True, 'Log option used device')\n":
            flags = True
    if not flags:
        sys.stdout.write('FLAGS:is_train set to False, are you sure to upload? y/n\r\n')
        sys.stdout.flush()
        if not raw_input() == 'y':
            exit('script terminate')
        else:
            exit('force put object')

auth = oss2.Auth('LTAIbHBeGsbrDmqv', 'KuwKLM7eqJN4BI4bSpoxBg8y4WrVcy')
bucket = oss2.Bucket(auth, 'oss-cn-shanghai.aliyuncs.com', 'mlearn')

bucket.put_object_from_file('cat/train.zip', 'train.zip')
print "train script uploaded to 'vocal_track/train.zip'"
