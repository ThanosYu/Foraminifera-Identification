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
df = df[(df['time'] >= 1565398800000) & (df['time'] <= 1568270650000)]
df = df.reset_index(drop=True)
df['value'] = df['value'].astype(pd.np.float64)
df['time'] = pd.to_datetime(df['time'], unit='ms')
print('===========start resample')
df = df.resample('1H', on='time').mean().ffill()

df.reset_index(level=0, inplace=True)

df['time'] = (df['time'].values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
df['time'] = df['time'].astype(pd.np.int64)
df['start_time'] = df['start_time'].astype(pd.np.int64)
df['end_time'] = df['end_time'].astype(pd.np.int64)

print('===========start insert')
for row in range(len(df)):
    print(df.loc[row, 'value'])
    print('%.19f' % (df.loc[row, 'value']))
    # insertSql = 'insert into data_feature_resample(sensor_id,type,time,end_time,start_time,value) values (%s,%s,%s,' \
    #             '%s,%s,%s) '
    # session.execute(insertSql,
    #                 ('S1MTR1001AC135RN', 'DKW', df.loc[row, 'time'], df.loc[row, 'end_time'], df.loc[row, 'start_time'],
    #                  df.loc[row, 'value']))
