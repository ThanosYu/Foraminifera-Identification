from cassandra.cluster import Cluster
from cassandra.policies import RoundRobinPolicy
import pandas
from scipy.fftpack import fft
import numpy

cluster = Cluster(contact_points=['10.192.27.232'],
                  port=9042,
                  load_balancing_policy=RoundRobinPolicy())
session = cluster.connect(keyspace='bbac')

# sql = 'select * from raw_data where sensor_id = %s and time > %s and time < %s'
# result = session.execute(sql, ('S2MTR1002AC90RN', 1536907526000000000, 1536907526001000000)).current_rows
sql = 'select * from raw_data'
result = session.execute(sql).current_rows
result = pandas.DataFrame(result)
x = result['time']
y = result['value']

xf = numpy.arange(len(x))
print(xf)
yf = abs(fft(y))
xf1 = xf[range(int(len(x) / 2))]  # 取一半区间
yf1 = yf[range(int(len(x) / 2))]  # 由于对称性，只取一半区间

insertSql = 'insert into fft_info_test(sensor_id,time,amplitude,data) values (%s,%s,%s,%s)'
session.execute(insertSql,
                ('S2MTR1002AC90RN', 1536907526000000000, ','.join(str(i) for i in yf1), ','.join(str(i) for i in xf1)))

# 关键连接
session.shutdown()
