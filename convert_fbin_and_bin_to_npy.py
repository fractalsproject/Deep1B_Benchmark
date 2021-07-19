import numpy
import os

def xbin_mmap(fname, dtype, maxn=-1):
    """ mmap the competition file format for a given type of items """
    n, d = map(int, numpy.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * numpy.dtype(dtype).itemsize
    if maxn > 0:
        n = min(n, maxn)
    return numpy.memmap(fname, dtype=dtype, mode="r", offset=8, shape=(n, d))

def knn_result_read(fname):
    n, d = map(int, numpy.fromfile(fname, dtype="uint32", count=2))
    assert os.stat(fname).st_size == 8 + n * d * (4 + 4)
    f = open(fname, "rb")
    f.seek(4+4)
    I = numpy.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
    D = numpy.fromfile(f, dtype="float32", count=n * d).reshape(n, d)
    return I, D

# read queries .fbin and get a contiguous array
queries = xbin_mmap("./query.public.10K.fbin", numpy.float32 )
print("queries type and shape:" , queries.dtype, queries.shape )

# read groundtruth .bin and get contiguous arrays
gtruth_indices, gtruth_distances = knn_result_read("./deep_new_groundtruth.public.10K.bin")
print("ground truth type and shape:", gtruth_indices.dtype, gtruth_indices.shape)

# save various sizes
sizes = [1, 10, 100, 1000, 10000]
for sz in sizes:
    q = queries[0:sz,:]
    q.tofile("deep1B_queries_from_website_%d.npy" % sz)
    print("saved %d queries in npy format with shape" % sz, q.shape)
    gt = gtruth_indices[0:sz,:]
    gt.tofile("deep1B_groundtruth_from_website_%d.npy" % sz)
    print("saved %d groundtruth vector(s) in npy format with shape" % sz, gt.shape)
