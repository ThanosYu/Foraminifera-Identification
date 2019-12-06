import pandas as pd

index = pd.date_range('1/1/2000', periods=9, freq='T')
series = pd.Series(range(9), index=index)
print('init', series)

series = series.resample('1H').ffill()
print('resample', series)
