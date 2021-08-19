import swagger_client
from swagger_client.models import *
from swagger_client.rest import ApiException
import sys
import time
import os
import numpy
import math
import faiss
import numpy as np
from multiprocessing.pool import ThreadPool

# Path to top-level data directory (queries, groundtruth, faissindex,...)
path_to_data = "/home/george/Projects/Deep1B_Benchmark_Data"

# Path to individual full query set
queries_file =  "from_yoav/deep1B_queries.npy"

# Path to ground truth 
groundtruth_file = "from_competition_site/deep_new_groundtruth.public.10K.bin"

# Number of queries to use in one batch search
queries_size = 10000

# Number of times for each search parameter set.
count = 3

# The nearest neighbors to retrieve
knn = 10 

# FAISS search batch size
fsearchbs = 8192

# FAISS search parameter set.
fsearchparams =  [ 1, 2, 4, 16, 32, 64, 128, 140, 160, 180, 200, 220, 240, 256 ]
#fsearchparams =  [ 1, 2, 4, 16, 32, 64, 128, 256, 512 ]

# FAISS index file ( set to None if you don't want to benchmark faiss )
# 99.32 findexfile = "/home/george/Projects/Deep1B_Benchmark_Data/deep-1B.IVF1048576,SQ8.faissindex"
findexfile = "/home/george/Projects/BigANN/harsha/big-ann-benchmarks/data/deep-1B.IVF1048576,SQ8.faissindex"

# FAISS search threads
fsearchthreads = -1 # -1 to use max threads

# FAISS search parallel mode
fparallelmode = 3

# Will contain the results of the benchmarks.
results = []

# Results will be written to this csv.
output = "./deep1B_results_gpu_only_%s.csv" % str(time.time()) 

# Print more intermediary results for debug purposes.
DEBUG = False

# Function that compute the standard recall@N.
def compute_recall(a, b):
    nq, rank = a.shape
    intersect = [ numpy.intersect1d(a[i, :rank], b[i, :rank]).size for i in range(nq) ]
    ninter = sum( intersect )
    return ninter / a.size, intersect

def unwind_index_ivf(index):
    if isinstance(index, faiss.IndexPreTransform):
        assert index.chain.size() == 1
        vt = index.chain.at(0)
        index_ivf, vt2 = unwind_index_ivf(faiss.downcast_index(index.index))
        assert vt2 is None
        return index_ivf, vt
    if hasattr(faiss, "IndexRefine") and isinstance(index, faiss.IndexRefine):
        return unwind_index_ivf(faiss.downcast_index(index.base_index))
    if isinstance(index, faiss.IndexIVF):
        return index, None
    else:
        return None, None

def rate_limited_iter(l):
    'a thread pre-processes the next element'
    pool = ThreadPool(1)
    res = None

    def next_or_None():
        try:
            return next(l)
        except StopIteration:
            return None

    while True:
        res_next = pool.apply_async(next_or_None)
        if res is not None:
            res = res.get()
            if res is None:
                return
            yield res
        res = res_next


class IndexQuantizerOnGPU:
    """ run query quantization on GPU """

    def __init__(self, index, search_bs):
        self.search_bs = search_bs
        index_ivf, vec_transform = unwind_index_ivf(index)
        self.index_ivf = index_ivf
        if vec_transform:
#            print(type(vec_transform),dir(vec_transform))
            self.vec_transform = vec_transform.apply
        else:
            self.vec_transform = None
        self.quantizer_gpu = faiss.index_cpu_to_all_gpus(self.index_ivf.quantizer)


    def produce_batches(self, x, bs):
        n = len(x)
        nprobe = self.index_ivf.nprobe
        ivf_stats = faiss.cvar.indexIVF_stats
        for i0 in range(0, n, bs):
            xblock = x[i0:i0 + bs]
            t0 = time.time()
            D, I = self.quantizer_gpu.search(xblock, nprobe)
            ivf_stats.quantization_time += 1000 * (time.time() - t0)
            yield i0, xblock, D, I

    def search(self, x, k):
        bs = self.search_bs
        if self.vec_transform:
            x = self.vec_transform(x)
        nprobe = self.index_ivf.nprobe
        n, d = x.shape
        assert self.index_ivf.d == d
        D = np.empty((n, k), dtype=np.float32)
        I = np.empty((n, k), dtype=np.int64)

        sp = faiss.swig_ptr
        stage2 = rate_limited_iter(self.produce_batches(x, bs))
        t0 = time.time()
        for i0, xblock, Dc, Ic in stage2:
            ni = len(xblock)
            self.index_ivf.search_preassigned(
                ni, faiss.swig_ptr(xblock),
                k, sp(Ic), sp(Dc),
                sp(D[i0:]), sp(I[i0:]),
                False
            )

        return D, I

    def range_search(self, x, radius):
        bs = self.search_bs
        if self.vec_transform:
            x = self.vec_transform(x)
        nprobe = self.index_ivf.nprobe
        n, d = x.shape
        assert self.index_ivf.d == d

        sp = faiss.swig_ptr
        rsp = faiss.rev_swig_ptr
        stage2 = rate_limited_iter(self.produce_batches(x, bs))
        t0 = time.time()
        all_res = []
        nres = 0
        for i0, xblock, Dc, Ic in stage2:
            ni = len(xblock)
            res = faiss.RangeSearchResult(ni)

            self.index_ivf.range_search_preassigned(
                ni, faiss.swig_ptr(xblock),
                radius, sp(Ic), sp(Dc),
                res
            )
            all_res.append((ni, res))
            lims = rsp(res.lims, ni + 1)
            nres += lims[-1]
        nres = int(nres)
        lims = np.zeros(n + 1, int)
        I = np.empty(nres, int)
        D = np.empty(nres, 'float32')

        n0 = 0
        for ni, res in all_res:
            lims_i = rsp(res.lims, ni + 1)
            nd = int(lims_i[-1])
            Di = rsp(res.distances, nd)
            Ii = rsp(res.labels, nd)
            i0 = int(lims[n0])
            lims[n0: n0 + ni + 1] = lims_i + i0
            I[i0:i0 + nd] = Ii
            D[i0:i0 + nd] = Di
            n0 += ni

        return lims, D, I

def load_index(idxfile, no_precomputed_tables=True, searchthreads=-1, parallel_mode=3,
                search_bs=8092 ):

    if not os.path.exists(idxfile):
        return False

    print("Loading index",idxfile)

    index = faiss.read_index(idxfile)

    index_ivf, vec_transform = unwind_index_ivf(index)
    if vec_transform is None:
        vec_transform = lambda x: x
    if index_ivf is not None:
        print("\tImbalance_factor=", index_ivf.invlists.imbalance_factor())

    if no_precomputed_tables:
        if isinstance(index_ivf, faiss.IndexIVFPQ):
            print("\tDisabling precomputed table")
            index_ivf.use_precomputed_table = -1
            index_ivf.precomputed_table.clear()

    precomputed_table_size = 0
    if hasattr(index_ivf, 'precomputed_table'):
        precomputed_table_size = index_ivf.precomputed_table.size() * 4
    print("\tPrecomputed tables size:", precomputed_table_size)

    # prep for the searches

    if searchthreads == -1:
        print("\tSearch threads:", faiss.omp_get_max_threads())
    else:
        print("\tSetting nb of threads to", searchthreads)
        faiss.omp_set_num_threads(searchthreads)

    if parallel_mode != -1:
        print("\tSetting IVF parallel mode to", parallel_mode)
        index_ivf.parallel_mode
        index_ivf.parallel_mode = parallel_mode

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    index_wrap = IndexQuantizerOnGPU(index, search_bs)

    cpu_index = index
    index = index_wrap

    return [ps, cpu_index, index]

try:

    print_header = False

    # Benchmark faiss if we have an index available
    if findexfile:
        # Do the faiss benchmarks.
        print("Capturing benchmarks for faiss...")

        # Reading faiss index file and configure.
        print("Reading faiss index file...")

        # Load the index
        ret = load_index(findexfile)
        if ret==False:
            raise Exception("Cannot read faiss index.")
        ps, cpu_index, index = ret
        print("Done reading index file.")

        # Do the faiss benchmarks.
        for device in [ "gpu", "cpu" ]:

            print("Running on %s..." % device)

            if DEBUG: print("Queries size=", queries_size)

            queries_path = os.path.join( path_to_data, queries_file )
            if DEBUG: print("Using queries from %s" % queries_path )

            # Load the queries array from file.
            queries = np.load( queries_path )

            # truncate based on the current subset test
            queries = queries[:queries_size,:]

            # Get the ground truth data file.
            gt_file = os.path.join( path_to_data, groundtruth_file )
            if DEBUG: print("Using groundtruth file %s" % gt_file)

            if DEBUG: print("Using knn=", knn )
            
            # Load the ground truth array from file.
            n, d = map(int, np.fromfile(gt_file, dtype="uint32", count=2))
            f = open(gt_file, "rb")
            f.seek(4+4)
            ground_truth = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
            if DEBUG: print("Ground truth=", n,d, ground_truth.shape )

            # truncate based on the current subset test and k test
            ground_truth = ground_truth[:queries_size,:knn]
            if DEBUG: print("Grouth truth dimensions and shape=", ground_truth.shape, ground_truth.dtype )

            # Iterate across search params.
            for params in fsearchparams:
               
                # Apply param to index
                sparam = "nprobe=%d" % params
                ps.set_index_parameters(cpu_index, sparam)

                # Load index to GPU as needed
                if device == "gpu":
                    index_wrap = index 
                else:
                    index_wrap = cpu_index

                # Reset stats
                ivf_stats = faiss.cvar.indexIVF_stats
                ivf_stats.reset()
           
                # Perform benchmark 'count' times and use best latency.
                best_latency = math.inf
                for c in range(count):

                    if device == "gpu":
                        t0 = time.time()
                        D, I = index_wrap.search(queries, knn)
                        t1 = time.time()
                    else:
                        t0 = time.time()
                        D, I = cpu_index.search(queries, knn)
                        t1 = time.time()

                    diff = t1-t0
                    if DEBUG: print("Latency:", diff)
                    if diff<best_latency:  best_latency = diff

                # Compute the recall
                recall, intersection = compute_recall(ground_truth[:, :knn], I[:, :knn])

                # Compute throughput
                qps = queries_size/best_latency

                # store results
                result =  [ device, queries_size, knn, params, best_latency, recall ] 
                results.append( result )
                if DEBUG: print("Result=",result) 

                # print results
                if not print_header:
                    print("device\tqsize\tknn\ttries\tprobes\tlatency\tqps\t\trecall")
                    print_header = True
                print("%s\t%d\t%d\t%d\t%d\t%1.1f\t%1.1f\t\t%1.2f" % (device,queries_size,knn,count,params,best_latency,qps,recall))
            
    else:
        print("No faiss index so not benchmarking it.")

    # Write the results
    if results:
        print("Writing results to file=%f" % output )
        f = open(output,"w+")
        f.write("device,query_set_size,k,params,latency,recall\n")
        for result in results:
            f.write(",".join([str(el) for el in result])+"\n")
        f.flush()
        f.close()
        print("Wrote results into file %s" % output)
    print("Done.")

except ApiException as e:
    print(e.status)
    print(e.body)
    print(e.reason)
    raise e

