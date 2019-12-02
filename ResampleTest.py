import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pd.set_option('display.mpl_style', 'default')

range = pd.date_range('2015-01-01', '2015-01-01', freq='15min')
df = pd.DataFrame(index=range)

# Average speed in miles per hour
df['speed'] = np.random.randint(low=0, high=60, size=len(df.index))
# Distance in miles (speed * 0.5 hours)
df['distance'] = df['speed'] * 0.25
# Cumulative distance travelled
df['cumulative_distance'] = df.distance.cumsum()
print(df)
# fig, ax1 = plt.subplots()
#
# ax2 = ax1.twinx()
# ax1.plot(df.index, df['speed'], 'g-')
# ax2.plot(df.index, df['distance'], 'b-')
#
# ax1.set_xlabel('Date')
# ax1.set_ylabel('Speed', color='g')
# ax2.set_ylabel('Distance', color='b')
#
# plt.show()
# plt.rcParams['figure.figsize'] = 12, 5

weekly_summary = pd.DataFrame()
weekly_summary['speed'] = df.speed.resample('W').mean()
weekly_summary['distance'] = df.distance.resample('W').sum()
weekly_summary['cumulative_distance'] = df.cumulative_distance.resample('W').last()

# Select only whole weeks
weekly_summary = weekly_summary.truncate(before='2015-01-05', after='2015-12-27')
weekly_summary.head()

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(weekly_summary.index, weekly_summary['speed'], 'g-')
ax2.plot(weekly_summary.index, weekly_summary['distance'], 'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Speed', color='g')
ax2.set_ylabel('Distance', color='b')

plt.show()
plt.rcParams['figure.figsize'] = 12, 5
