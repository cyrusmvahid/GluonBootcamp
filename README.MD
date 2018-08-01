# MXNet/Gluon Bootcamp

## Setup

Since this repository contains submodules use
the following command to clone:

`git clone --recurse-submodules
https://github.com/cyrusmvahid/GluonBootcamp.git`

## Schedule

### Day 1
|Title | Description | Duration | Presenter| 
|:---  |:---         |:---      |:---      |
|MXNet & Gluon Overview|An overview of MXNet architecture and components|30 mins|Thom|
|SageMaker Setup|Get configured for the day ahead|15 mins|Christian|
|[Crash Course (LAB)](labs/gluon_crash_course)|Walkthrough of core Gluon components, and use them to create and train a convolutional neural network|120 mins|Thom & Cyrus|
|Custom DataSets and DataLoaders|Create customized DataSets and DataLoaders for various data types|30 mins|Thom|
|[MultiGPU Training (LAB)](labs/multiple_gpus_gluon/multiple_gpus_gluon.ipynb)|An introduction to training using multiple GPUs in Gluon, and hands on lab |60 mins|Eden|
|Multi Host Training|A deeper dive into cluster setup, sizing, optimization algorithms, and other considerations for distributed training|60 mins|Eden & Cyrus|
|SageMaker Deep-Dive|A deep-dive into SageMaker SDK and security model|30 mins|Christian|
|SageMaker DevOps|A deep dive into how to integrate SageMaker into development pipeline for teams of data scientists|30 mins|Christian|
|[Multi Host Training with SageMaker(LAB)](labs/distributed_training_gluon/distributed_training_gluon.ipynb)|Participants learn the simple steps to training using a cluster with Gluon and Amazon Sagemaker|30 mins|Eden|

### Day2
|Title | Description | Duration | Presenter| 
|:---  |:---         |:---      |:---      |
|Deep Learning AMI Setup|An introduction to DLAMI and get configured for the day ahead|30 mins|Christian|
|Beyond the Defaults|Use simple regression model we take a look at alternatives for initialization, optimization and evaluation.|30 mins|Cyrus|
|MXBoard|Using MXNet with Tensorboard to monitor training|30 mins|Thom|
|GluonNLP|Walk through of key motivations and features of the GluonNLP toolkit.|30 mins|Cyrus|
|Stacked Bidirectional LSTM (LAB)|We work up from plain RNN to Stacked Bi-directional LSTMs using Gluon layers|60 mins|Thom|
|LSTM Model (LAB)|Participant in this section implement a simple LSTM network|60 mins|Cyrus|
|LSTNet Theory|This section describes theory of Long and Short Term Temporal Patterns with Deep Neural Networks|30 mins|Cyrus|
|LSTNet Implementation (LAB)|While referencing the paper, we construct the LSTNet model and train on multi-gpu|90 mins|Thom|
|LSTNet Multi GPU (LAB)|Scale training of LSTNet model with Multi GPU|30 mins|Cyrus|

### Day3
|Title | Description | Duration | Presenter| 
|:---  |:---         |:---      |:---      |
|Profiling MXNet|Analysis of runtime code to
identify performance bottlenecks|30 mins|Thom|
|[Moving LSTNet to SageMaker(LAB)](labs/porting_lstnet_to_sagemaker/porting_lstnet_to_sagemaker.ipynb)|The LSTNet code will be ported to train on Amazon SageMaker using the Python SDK|90 mins|Eden|
|[Distributed LSTNet on SageMaker(LAB)](labs/lstnet_multi_gpu_distributed/lstnet_multi_gpu_distributed.ipynb)|Participants will speed up the training of the LSTNet model with data distribution across multiple GPUs and hosts|30 mins|Eden|
|SageMaker Automatic Model Tuning|A look at SageMaker's hyperparameter optimization features|60 mins|Thom|
|[Deploying with SageMaker (LAB)](labs/deploying_endpoint/deploying_endpoint.ipynb)|We discover three scenarios, Deploying a model developed in SageMaker, Deploying model artefacts from S3, and deploying model artefacts from docker image|60 mins|Eden|
|MXNet on Edge with Raspberry PI|We demonstrate a simple model that runs on Raspberry PI 3|30 min|Christian|
|MXNet on Edge with Amazon Greengrass|IWe demonstrate deployment of a simple CV model on Amazon GreenGrass using Lambda|30 mins|Christian|
|Keras-MXNet 2 (LAB)|We use Keras to implement a bi-directional LSTM using Keras and the MXNet backend|30 mins|Cyrus|
|GluonCV toolkit|Walk through of key features of the GluonCV toolkit, with demo.|30 mins|Thom|

### Day4
|Title | Description | Duration | Presenter| 
|:---  |:---         |:---      |:---      |
|Gluon project structure|Suggestions on how you can organize your MXNet Gluon projects.|30 mins|Cyrus|
|Hackathon|A project based on bi-directional LSTM will be implemented to incorporate all the material from the first three days. If participants have prepared datasets of significant size we attempt to optimize the distributed training for large batch sizes|6.5 hours|All|

```{.python .input  n=4}
## Convert to markdown
# !notedown README.ipynb --to markdown > README.MD
```

```{.python .input  n=2}
## Convert to slides
# jupyter nbconvert <your notebook> --to slides --reveal-prefix reveal.js
## then copy the html file to `slides/note2slides` containing reveal.js
```
