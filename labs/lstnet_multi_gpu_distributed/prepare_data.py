import boto3
import pandas as pd
import os
import numpy as np

bucket = 'eduthie-sagemaker-1'
prefix = 'lstnet'

data_dir = '../data'
data_file = '{}/electricity.txt'.format(data_dir)
splits = 10
train_frac = 0.8

df = pd.read_csv(data_file,header=None)
print(df.describe())
max_columns = df.max().astype(np.float64)
df = df/max_columns # normalize
print(df.describe())

num_time_steps = len(df)
split_index = int(num_time_steps*train_frac)
train = df[0:split_index]
print('Training size {}'.format(len(train)))
test = df[split_index:]
print('Test size {}'.format(len(test)))

train_sets = []
train_len = len(train)
train_size = int(train_len)/splits
for i in range(0,splits):
    start = int(i*train_size)
    end = int((i+1)*train_size)
    print('start {}'.format(start))
    print('end {}'.format(end))
    if end < (train_len-1):
        train_sets.append(train[start:end])
    else:
        train_sets.append(train[start:])


test_file_path = os.path.join(data_dir,'test.csv')
test.to_csv(test_file_path,header=None,index=False)
train_file_path = os.path.join(data_dir,'train.csv')
train.to_csv(train_file_path,header=None,index=False)

client = boto3.client('s3')

for i in range(0,splits):
    file_path = os.path.join(data_dir,'train_{}.csv'.format(i))
    print('Uploading file: {} with {} rows'.format(file_path,len(train_sets[i])))
    train_sets[i].to_csv(file_path,header=None,index=False)
    client.upload_file(file_path, bucket, prefix + '/train/multiple_host/train_{}.csv'.format(i))

client.upload_file(test_file_path, bucket, prefix + '/test/test.csv')
client.upload_file(train_file_path, bucket, prefix + '/train/single_host/train.csv')
