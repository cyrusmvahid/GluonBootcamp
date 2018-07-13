from sagemaker.mxnet import MXNet

bucket = 'eduthie-sagemaker-1'
prefix = 'lstnet'

hyperparameters = {
    'conv_hid' : 100,
    'gru_hid' : 100,
    'skip_gru_hid' : 5,
    'skip' : 24,
    'ar_window' : 24,
    'window' : 24*7,
    'horizon' : 24,
    'learning_rate' : 0.001,
    'clip_gradient' : 10.,
    #'batch_size' : 128,
    'batch_size' : 512,
    'epochs' : 100

}

# lstnet1 = MXNet(entry_point='lstnet_sagemaker.py',
#     source_dir='/Users/duthiee/code/LSTNet/src',
#     role='arn:aws:iam::987551451182:role/service-role/AmazonSageMaker-ExecutionRole-20180423T174055',
#     output_path='s3://{}/{}/output'.format(bucket, prefix),
#     train_instance_count=1,
#     train_instance_type='ml.p2.xlarge',
#     hyperparameters=hyperparameters)
# lstnet1.fit(wait=False,inputs={'train': 's3://{}/{}/train/single_host/'.format(bucket, prefix),'test': 's3://{}/{}/test/'.format(bucket, prefix)})

#
# lstnet2 = MXNet(entry_point='lstnet_sagemaker.py',
#     source_dir='/Users/duthiee/code/LSTNet/src',
#     role='arn:aws:iam::987551451182:role/service-role/AmazonSageMaker-ExecutionRole-20180423T174055',
#     output_path='s3://{}/{}/output'.format(bucket, prefix),
#     train_instance_count=1,
#     train_instance_type='ml.p3.2xlarge',
#     hyperparameters=hyperparameters)
# lstnet2.fit(wait=False,inputs={'train': 's3://{}/{}/train/single_host/'.format(bucket, prefix),'test': 's3://{}/{}/test/'.format(bucket, prefix)})
#
# lstnet3 = MXNet(entry_point='lstnet_sagemaker.py',
#     source_dir='/Users/duthiee/code/LSTNet/src',
#     role='arn:aws:iam::987551451182:role/service-role/AmazonSageMaker-ExecutionRole-20180423T174055',
#     output_path='s3://{}/{}/output'.format(bucket, prefix),
#     train_instance_count=5,
#     train_instance_type='ml.p3.2xlarge',
#     hyperparameters=hyperparameters)
# lstnet3.fit(wait=False,inputs={'train': 's3://{}/{}/train/multiple_host'.format(bucket, prefix),'test': 's3://{}/{}/test/'.format(bucket, prefix)})
#
# lstnet4 = MXNet(entry_point='lstnet_sagemaker.py',
#     source_dir='/Users/duthiee/code/LSTNet/src',
#     role='arn:aws:iam::987551451182:role/service-role/AmazonSageMaker-ExecutionRole-20180423T174055',
#     output_path='s3://{}/{}/output'.format(bucket, prefix),
#     train_instance_count=1,
#     train_instance_type='ml.p3.8xlarge',
#     hyperparameters=hyperparameters)
# lstnet4.fit(wait=False,inputs={'train': 's3://{}/{}/train/single_host/'.format(bucket, prefix),'test': 's3://{}/{}/test/'.format(bucket, prefix)})

# ##The winner, this takes 7 minutes in total with 158 seconds of training time
# lstnet5 = MXNet(entry_point='lstnet_sagemaker.py',
#     source_dir='/Users/duthiee/code/LSTNet/src',
#     role='arn:aws:iam::987551451182:role/service-role/AmazonSageMaker-ExecutionRole-20180423T174055',
#     output_path='s3://{}/{}/output'.format(bucket, prefix),
#     train_instance_count=5,
#     train_instance_type='ml.p3.8xlarge',
#     hyperparameters=hyperparameters)
# lstnet5.fit(wait=False,inputs={'train': 's3://{}/{}/train/multiple_host'.format(bucket, prefix),'test': 's3://{}/{}/test/'.format(bucket, prefix)})
