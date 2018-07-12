# MXNet/Gluon Bootcamp
## Fundamentals

### Day 1
| Title | Description |
Duration | Presenter| 
|:---    |:---   |:---         |:---      |
|Sage Maker
Architecture|A deep dive into SageMaker SDK and security model|30 min|Cristian|
|DevOps with SageMaker|A deep dive into how to integrage SageMaker into
development pipeline for teams of data scientists|30 min|Christian|
|MXNet
overview|an overview of MXNet architecture and components|30 min|Thom|
|First
glance at Gluon|Walking through a simple regression model to get an
understanding of gluon components|30 min|Cyrus|
|LAB: Interactive GLuon crash
course|A step-by-step implementation of a conv net using multiple components of
Gluon|120 min|Thom, Cyrus|
|Multi GPU and cluster training using Gluon|an
introduction to developing models to run across multiple devices and multiple
machiens|60 min|Eden|
|Optimizing for distributed training|A deeper dive into
cluster sizing, optimization algorithms, and other considerations for
distributed training|60 minutes|Thom, Eden, Cyrus|
|LAB: Multi-device
MNist|Participants continue on develop their crash course lab to run across
multiple devices|60 min|Eden|

### Day2
| Title | Description | Duration |
Presenter| 
|:---    |:---   |:---         |:---      |
|Getting Started with
Deep Learning AMI|The particpants will be introduced to DLAMI and versions of
frameworks it supports|30 min|Christian|
|LAB:Customized DataSets and Data
Loaders in Gluon|Participants will learn hour to derive customized datasets from
DataSet class to fit their requirements|30 min|Thom|
|LAB: Block and Hybrid
BLock|Participants learn how to created customized blocks in order to create
flexible and non-sequential models|30 min|Thom|
|MXNBoard and Profiling|This
module includes how to profile MXNet using pycharm professional and detect
performance bottlenecks|30 min|Thom|
|LAB: Implement a simple LSTM|Participant
in this section implement a simple LSTM network|60 min|Cyrus|
|Optimizing
LSTM|Optimizing training performance of LSTM network in multi-device setting
using MXBoard|30 min|Cyrus, Thom|
|Theory of LSTNet|This section describes
theory of Long and Short Term Temporal Patterns with Deep Neural Networks|30
min|Cyrus|
|LAB: Implementing LSTNet|A code walk through will be presented to
the audiance as how LSTNet is implementd in Gluon|90 min|Thom|
|LAB: Optimizing
your LSTNet for multi-device training|converting your code for multi-device
training and then using optimization ideas to increased training speed|60
min|Thom and Cyrus|
|Hybredize|At this point we change training mode from
imperative to symbolic and traing the LSTNet model on multi-device setting in
symbolic mode|30 min|Thom|

### Day3
| Title | Description | Duration |
Presenter| 
|:---    |:---   |:---         |:---      |
|LAB: Back to Amazon
SageMaker|The participants will wrap the gluon code using Amazon SageMaker
python SDK|90 min|Eden, Christian|
|LAB: Distribited training of LSTNet on
SageMaker|Participants attempt to train their model on multi-device and multi-
machine settings|30 min|Eden|
|HyperParameterOptimization, the theory|SageMaker
can optionally use Bayesian HPO for training optimization. This module describes
the science of Bayesian HPO. Additionally we shall demonstrate how to use HPO
for training of a simple gluon model in SageMaker|60 min|Thom|
|LAB:Deploying an
end point using Amazon SageMaker|We discover three scenarios, Deploying a model
developed in SageMaker, Deploying model artefacts from S3, and deploying model
artefacts from docker image|60 min|Eden and Christian|
|Hosting a model on
EC2|In this modlude we use MXNet on EC2 to host a model based on pretrained
model artefacts|30 min|Eden|
|MXNet on Edge, Pi3 version|In this module we
demonstrate a simple model that runs on RPI3|30 min|Christian|
|MXNet on Edge,
Amazon GreenGrass|In this module we demonstrate deployment of a simple computer
vision model on Amazon GreenGrass using Lambda|30 min|Christian|
|LAB:
Bidirectional LSTM using Gluon|In this module We extende our original simple
LSTM to become bi-directional|30 min|Cyrus and Thom|
|LAB: Bidirectional LSTM
using Keras 2 and MXNet|In this module we use Keras to implement a bidirection
LSTM using Keras and MXNet backend|30 min|Cyrus and Thom|
|Advaned GLuon
Linraries|Lobraries such as gluonCV and gluon NLP will explained to familiarize
the participants with the lastest releases.|30 min|Thom|

### Day4
| Title |
Description | Duration | Presenter| 
|:---    |:---   |:---         |:---      |
|Hackathon|A project based on bi-directional LSTM will be implemented to
incorporate all the material from the first three days. If participants have
prepared datasets of significant size we attempt to optimize the distributed
training for large batch sizes|7 hours|all|

```{.python .input  n=3}
!notedown README.ipynb --to markdown > README.MD
```
