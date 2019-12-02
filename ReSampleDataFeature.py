import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

cluster = Cluster(contact_points=['10.192.27.110'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='dex')

cql = 'select * from data_feature where sensor_id = %s AND type = %s allow filtering'
result = session.execute(cql, ('S1MTR1001AC135RN', 'DKW')).current_rows
df = pd.DataFrame(result)
df = df[(df['time'] >= 1536903994000) & (df['time'] <= 1536904594000)]
df = df.reset_index(drop=True)

df['time'] = pd.to_datetime(df['time'], unit='ms')

df = df.set_index('time')

df['value'] = df['value'].resample('1S').ffill()
df['end_time'] = df['end_time'].resample('1S').ffill()
df['start_time'] = df['start_time'].resample('1S').ffill()
df['type'] = df['type'].resample('1S').ffill()
df['sensor_id'] = df['sensor_id'].resample('1S').ffill()
print('resample ', df['end_time'])
# df = df[~(df['value'].isnull())]

df.reset_index(level=0, inplace=True)

df['time'] = (df['time'].values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')

df['time'] = df['time'].astype(pd.np.int64)

df['start_time'] = df['start_time'].astype(pd.np.int64)
df['end_time'] = df['end_time'].astype(pd.np.int64)

# for row in range(len(df)):
# print(df.loc[row, 'time'])
# insertSql = 'insert into sandstone_image(category,feature_1,feature_2,feature_3,feature_4,base64) values (%s,%s,' \
#             '%s,%s,%s,%s) '
# session.execute(insertSql,
#                 (df.loc[row, 'Category'], df.loc[row, 'Feature_1'], df.loc[row, 'Feature_2'],
#                  df.loc[row, 'Feature_3'], df.loc[row, 'Feature_4'], df.loc[row, 'Base64']))
# if row % 10 == 0:
#     print('**********************row: ', row)
# print(df)
