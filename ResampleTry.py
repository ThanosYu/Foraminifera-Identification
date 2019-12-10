import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

cluster = Cluster(contact_points=['10.192.27.110'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='dex')

sensorId = 'S1MTR1001AC135RN'
type = 'Crest'
startTime = 1565398800000
endTime = 1565401740000

cql_data_feature = 'select * from data_feature where sensor_id = %s AND type = %s'
result = session.execute(cql_data_feature, (sensorId, type)).current_rows
df = pd.DataFrame(result)
df = df[(df['time'] >= startTime) & (df['time'] <= endTime)]
df = df.reset_index(drop=True)

df['time'] = pd.to_datetime(df['time'], unit='ms')
print('=========init', df)

df = df.set_index('time')
print('=========index', df)

df = df.resample('1H').ffill().bfill()
print('reSample', df)

df.reset_index(level=0, inplace=True)

df['time'] = (df['time'].values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
df['time'] = df['time'].astype(pd.np.int64)

print('=========reset', df)
