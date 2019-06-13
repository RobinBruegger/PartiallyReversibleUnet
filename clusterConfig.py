import os
import argparse
import datetime

#process arguments
parser = argparse.ArgumentParser()
parser.add_argument("--isCluster", "--iscluster", help="1 for cluster, 0 for local, default is 1", type=int, default=1)
parser.add_argument("--id", help="id of the experiment", type=int)
args = parser.parse_args()
if args.isCluster:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ['SGE_GPU']
    print("Running on GPU " + os.environ['SGE_GPU'])
    IS_CLUSTER = True
else:
    IS_CLUSTER = False

id = args.id

