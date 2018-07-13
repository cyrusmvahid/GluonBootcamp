import os

os.environ.update({
  "DMLC_ROLE": "server", # Could be "scheduler", "worker" or "server"
  "DMLC_PS_ROOT_URI": "127.0.0.1", # IP address of a scheduler
  "DMLC_PS_ROOT_PORT": "9000", # Port of a scheduler
  "DMLC_NUM_SERVER": "1", # Number of servers in cluster
  "DMLC_NUM_WORKER": "2", # Number of workers in cluster
  "PS_VERBOSE": "2" # Debug mode
})

import mxnet as mx
