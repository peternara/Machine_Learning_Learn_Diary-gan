import pandas as pd
import numpy as np

reader = pd.read_csv('data/train.csv', iterator=True)

data = reader.get_chunk(50).set_index('User_id')
data['is_use'] = data['Date_online'] == 'null'

print data