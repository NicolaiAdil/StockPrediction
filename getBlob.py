from azure.storage.blob import BlobServiceClient
import pandas as pd
from Credentials import storageaccounturl, storageaccountkey, containername, blobname

STORAGEACCOUNTURL= storageaccounturl
STORAGEACCOUNTKEY= storageaccountkey
LOCALFILENAME= "localfile.csv"
CONTAINERNAME= containername
BLOBNAME= blobname

#download from blob
# t1=time.time()
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
with open(LOCALFILENAME, "wb") as my_blob:
    blob_data = blob_client_instance.download_blob()
    blob_data.readinto(my_blob)
# t2=time.time()
print("It takes seconds to download "+BLOBNAME)