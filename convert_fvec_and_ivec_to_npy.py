
import numpy 

def fvecs_mmap(fname):
    x = numpy.memmap(fname, dtype='float32', mode='r')
    d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 1)[:, 1:]

def ivecs_read(fname):
    a = numpy.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

# open queries fvecs and get contiguous array
queries = fvecs_mmap( "./deep1B_queries.fvecs" )
print("query shape and type:", queries.shape, queries.dtype)

# save in numpy format
numpy.save( "./deep1B_queries_from_website.npy", queries)
print("queries converted to npy file")

# open ground truth ivecs and get a contigous array
groundtruth = ivecs_read( "./deep1B_groundtruth.ivecs" )
print("ground truth shape and type:", groundtruth.shape, groundtruth.dtype )

# same in numpy format
numpy.save("./deep1B_groundtruth_from_website.npy", groundtruth)
print("ground truth converted to npy file")
