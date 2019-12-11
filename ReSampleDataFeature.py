import pandas as pd
import numpy as np
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

cluster = Cluster(contact_points=['10.192.27.110'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='dex')

cql_feature_type = 'select distinct sensor_id, type from data_feature'
typeResult = session.execute(cql_feature_type).current_rows
typeDf = pd.DataFrame(typeResult)

startTime = 1565398800000
endTime = 1568270649060

for row in range(len(typeDf)):
    sensorId = typeDf.loc[row, 'sensor_id']
    type = typeDf.loc[row, 'type']
    print('======sensorId: ', sensorId)
    print('======type: ', type)

    cql_data_feature = 'select * from data_feature where sensor_id = %s AND type = %s and time >= %s and time <= %s ' \
                       'allow filtering '
    result = session.execute(cql_data_feature, (sensorId, type, startTime, endTime)).current_rows
    df = pd.DataFrame(result)
    print('=========df size', df.shape[0])
    if df.shape[0] == 0:
        continue

    df['time'] = pd.to_datetime(df['time'], unit='ms')
    # print('=========init', df)

    df = df.set_index('time')
    # print('=========index', df)

    df = df.resample('1H').bfill().ffill()
    # print('reSample', df)

    df.reset_index(level=0, inplace=True)
    # print('=========reset', df)
    df['time'] = (df['time'].values - np.datetime64('1970-01-01T08:00:00Z')) / np.timedelta64(1, 'ms')
    df['time'] = df['time'].astype(pd.np.int64)

    if df.shape[0] < 10:
        continue

    for subRow in range(len(df)):
        # print('==========subRow', subRow)
        insertSql = 'insert into data_feature_resample(sensor_id,type,time,end_time,start_time,value) values (%s,%s,' \
                    '%s,%s,%s,%s) '
        session.execute(insertSql,
                        (sensorId, type, df.loc[subRow, 'time'], df.loc[subRow, 'end_time'],
                         df.loc[subRow, 'start_time'],
                         df.loc[subRow, 'value']))
        if subRow % 10 == 0:
            print('**********************row: ', subRow)
