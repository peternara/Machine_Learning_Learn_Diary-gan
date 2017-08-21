import pandas as pd

offline = pd.DataFrame(pd.read_csv('data/ccf_offline_stage1_train.csv')).set_index(['User_id', 'Coupon_id'])

online = pd.DataFrame(pd.read_csv('data/ccf_online_stage1_train.csv')).set_index(['User_id', 'Coupon_id'])

merged = pd.merge(offline, online, how='outer', left_index=True, right_index=True, suffixes=('_offline', '_online'))

merged = pd.DataFrame(merged)


open('data/train.csv', 'wb').write(merged.to_csv())
