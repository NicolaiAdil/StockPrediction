import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from pandas_datareader import data as pdr
import os, uuid

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

GJF = yf.download("GJF.OL", period="max")
GJF_close = GJF['Close']

print(GJF_close)

yf.pdr_override()
data = pdr.get_data_yahoo("GJF.OL", period="max")
close = data.reset_index()['Close']

print(close)
plt.plot(close)
plt.show()

# t = np.arange(0,100,len(hist))

# plt.plot(t,hist)
# plt.show()

