import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()
tf.compat.v1.random.set_random_seed(1234)

#/ LOCAL FILES /#
from Linear_Regression import getData

#Get the data from csv file
df = pd.read_csv('GjF_OneYear.csv')


#Get the close price
minmax = MinMaxScaler().fit(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = minmax.transform(df.iloc[:, 4:5].astype('float32')) # Close index
df_log = pd.DataFrame(df_log)
#print(df_log.head())

#Test size -> n number of days
test_size = 30
simulation_size = 10

df_train = df_log.iloc[:-test_size]
df_test = df_log.iloc[-test_size:]

print(df.shape, df_train.shape, df_test.shape)


