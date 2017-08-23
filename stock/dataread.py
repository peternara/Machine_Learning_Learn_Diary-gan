# coding=utf-8
import pandas as pd

csv = pd.read_csv('./stock_dataset.csv')
print csv['最高价'][::1]
