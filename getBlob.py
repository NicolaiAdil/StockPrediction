from azure.storage.blob import BlobServiceClient
import pandas as pd

STORAGEACCOUNTURL= "https://stockpredictio5203052435.blob.core.windows.net"
STORAGEACCOUNTKEY= "IEZDxzg3RoO8wxbWphz70pwGQbpIk+0UWky0iC3YvBXAvBinrKIZpI3wuzNK3vqzCueCnvKHVzx/+AStF2KCEQ=="
#STORAGEACCOUNTKEY = "sp=r&st=2022-11-11T10:32:12Z&se=2022-11-11T18:32:12Z&spr=https&sv=2021-06-08&sr=c&sig=2t6gPYD8ofMgTUdQDfE7iTsobgiiGF6A2nKbJCetsQ8%3D"
LOCALFILENAME= "localfile.csv"
CONTAINERNAME= "predictionfiles"
BLOBNAME= "ds.csv"

#download from blob
# t1=time.time()
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)
with open(LOCALFILENAME, "wb") as my_blob:
    blob_data = blob_client_instance.download_blob()
    blob_data.readinto(my_blob)
# t2=time.time()
print("It takes seconds to download "+BLOBNAME)