# MXNet/Gluon Bootcamp
### Day 1
| Title | Description | Duration | Presenter|
|:---    |:---   |:---         |:---      |
|Sage Maker Architecture|A deep dive into SageMaker SDK and security model|30 min|Christian|
|DevOps with SageMaker|A deep dive into how to integrate SageMaker into development pipeline for teams of data scientists|30 min|Christian|
|MXNet overview|An overview of MXNet architecture and components|30 min|Thom|
|First glance at Gluon|Walking through a simple regression model to get an understanding of gluon components|30 min|Cyrus|
|LAB: Interactive GLuon crash course|A step-by-step implementation of a conv net using multiple components of Gluon|120 min|Thom, Cyrus|
|Multi GPU Training in Gluon|An introduction to training using multiple GPUs in Gluon |15 min|Eden|
|LAB: Multi GPU Training in Gluon|A hands-on lab to get started training using multiple GPUs with Gluon|30 min|Eden|
|Distributed Training with MXNet|A deeper dive into cluster setup, sizing, optimization algorithms, and other considerations for distributed training|60 min|Thom, Eden, Cyrus|
|LAB: Distributed training with Amazon SageMaker and Gluon|Participants learn the simple steps to training using a cluster with Gluon and Amazon Sagemaker|30 min|Eden|

### Day2
| Title | Description | Duration | Presenter| 
|:--- |:---   |:---         |:---      |
|Getting Started with Deep Learning AMIs|The particpants will be introduced to the Amazon Deep Learning AMIs and versions of frameworks it supports|30 min|Christian|
|LAB:Customized DataSets and Data Loaders in Gluon|Participants will learn how to derive customized datasets from DataSet class to fit their requirements|30 min|Thom|
|LAB: Block and Hybrid BLock|Participants learn how to created customized blocks in order to create flexible and non-sequential models|30 min|Thom|
|MXNBoard and Profiling|This module includes how to profile MXNet using pycharm professional and detect performance bottlenecks|30 min|Thom|
|LAB: Implement a simple LSTM|Participant in this section implement a simple LSTM network|60 min|Cyrus|
|Optimizing LSTM|Optimizing training performance of LSTM network in multi-device setting using MXBoard|30 min|Cyrus, Thom|
|Theory of LSTNet|This section describes theory of Long and Short Term Temporal Patterns with Deep Neural Networks|30 min|Cyrus|
|LAB: Implementing LSTNet|A code walk through will be presented to the audience as how LSTNet is implementd in Gluon|90 min|Thom|
|LAB: Optimizing your LSTNet for multi-device training|converting your code for multi-device training and then using optimization ideas to increased training speed|60 min|Thom and Cyrus|
|Hybrydize|At this point we change training mode from imperative to symbolic and traing the LSTNet model on multi-device setting in symbolic mode|30 min|Thom|

### Day3
| Title | Description | Duration | Presenter| 
|:---    |:---   |:---         |:---      |
|LAB: Back to Amazon SageMaker|The LSTNet code will be ported to train on Amazon SageMaker using the Python SDK|90 min|Eden|
|LAB: Distributed and multi-GPU LSTNet on SageMaker|Participants will speed up the training of the LSTNet model with data distribution across multiple GPUs and hosts|30 min|Eden|
|HyperParameterOptimization, the theory|SageMaker can optionally use Bayesian HPO for training optimization. This module describes the science of Bayesian HPO. Additionally we shall demonstrate how to use HPO for training of a simple gluon model in SageMaker|60 min|Thom|
|LAB:Deploying an end point using Amazon SageMaker|We discover three scenarios, deploying a model developed in SageMaker, deploying model artefacts from S3, and deploying model artefacts from docker image|60 min|Eden|
|MXNet on Edge, Pi3 version|In this module we demonstrate a simple model that runs on Raspberry Pi 3|30 min|Christian|
|MXNet on Edge, AWS Greengrass|In this module we demonstrate deployment of a simple computer vision model on AWS Greengrass using Lambda|30 min|Christian|
|LAB: Bidirectional LSTM using Gluon|In this module we extend our original simple LSTM to become bi-directional|30 min|Cyrus and Thom|
|LAB: Bidirectional LSTM using Keras 2 and MXNet|In this module we use Keras to implement a bidirection LSTM using Keras and MXNet backend|30 min|Cyrus and Thom|
|Advaned Gluon Libraries|Libraries such as gluonCV and gluon NLP will explained to familiarize the participants with the lastest releases.|30 min|Thom|

### Day4
| Title | Description | Duration | Presenter| 
|:---    |:---   |:---         |:---      |
|Hackathon|A project based on bi-directional LSTM will be implemented to incorporate all the material from the first three days. If participants have prepared datasets of significant size we attempt to optimize the distributed training for large batch sizes|7 hours|all|

```{.python .input  n=6}
# !notedown README.ipynb --to markdown > README.MD
```
