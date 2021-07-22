import numpy as np
import sys

# load yoav's query file
from_yoav_path = "/home/george/Projects/Deep1B_Benchmark_Data/from_yoav/deep1B_queries.npy"
from_yoav = np.load( from_yoav_path )
print( "numpy array from yoav's query file is", from_yoav.shape, from_yoav.dtype )

# load the competition query file
from_competition_site_path = '/home/george/Projects/Deep1B_Benchmark_Data/from_competition_site/query.public.10K.fbin'
n, d = map(int, np.fromfile(from_competition_site_path, dtype="uint32", count=2))
from_competition_site = np.fromfile(from_competition_site_path, dtype=np.float32, offset=8)
from_competition_site = from_competition_site.reshape( (n, d) )
print( "numpy array from competition site query file is", from_competition_site.shape, from_competition_site.dtype )

# load the compeition ground truth
from_competition_groundtruth = '/home/george/Projects/Deep1B_Benchmark_Data/from_competition_site/deep_new_groundtruth.public.10K.bin'
n, d = map(int, np.fromfile(from_competition_groundtruth, dtype="uint32", count=2))
f = open(from_competition_groundtruth, "rb")
f.seek(4+4)
I = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
D = I.astype( np.float32 )
print( "ground truth from competition site is", I.shape, I.dtype )

# iterate over all vectors in both arrays
print("Comparing vectors between both queries files...")
for i in range(10000):
    #print("comparing vectors at position %d/10000..." % i,)
    for j in range(96):
        if from_yoav[i][j] != from_competition_site[i][j]:       
            raise Exception("Vector %d element %d is not identical" % (i, j) )
    #print("is identical")
print("All vectors are the same.")



