import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy

cluster = Cluster(contact_points=['10.0.0.110'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='dex')

cql = 'select * from data_feature where sensor_id = %s AND type = %s allow filtering'
result = session.execute(cql, ('S1MTR1001AC135RN', 'DKW')).current_rows
df = pd.DataFrame(result)
df = df[(df['time'] >= 1536903994000) & (df['time'] <= 1568270649060)]
df1 = df.value.resample('1S').mean()


print(df)
