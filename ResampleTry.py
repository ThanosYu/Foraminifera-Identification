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
startTime = 1565401200000
endTime = 1565401740000

cql_data_feature = 'select * from data_feature where sensor_id = %s AND type = %s and time >= %s and time <= %s allow ' \
                   'filtering '
result = session.execute(cql_data_feature, (sensorId, type, startTime, endTime)).current_rows
df = pd.DataFrame(result)
print(df)

# enlarge the time range
row = [{'time': 1565412540000}]

df = df.append(row, ignore_index=True)
print('=========new', df)

df['time'] = pd.to_datetime(df['time'], unit='ms')
# print('=========init', df)

df = df.set_index('time')
print('=========index', df)

df = df.resample('1S').bfill().ffill()
print('reSample', df)

df.reset_index(level=0, inplace=True)

df['time'] = (df['time'].values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
df['time'] = df['time'].astype(pd.np.int64)

# print('=========reset', df)
