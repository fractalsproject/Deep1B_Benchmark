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

# Configuration
configuration = swagger_client.configuration.Configuration()
configuration.verify_ssl = False
configuration.host = "http://localhost:7761/v1.0"
api_config = swagger_client.ApiClient(configuration)

# APIs
gsi_boards_apis = swagger_client.BoardsApi(api_config)
gsi_datasets_apis = swagger_client.DatasetsApi(api_config)
gsi_search_apis = swagger_client.SearchApi(api_config)

# Num boards (if not using an existing allocation id.  
# Set to 0 to disable gemini benchmarking.
#num_of_boards = 4
num_of_boards = 0

# Num of centroids to make active for gemini.
num_of_centroids = 2097152 #1048576

# Path to top-level data directory (queries, groundtruth, faissindex,...)
path_to_data = "/home/george/Projects/Deep1B_Benchmark_Data"

# Path to individual full query set
queries_file =  "from_yoav/deep1B_queries.npy"

# Query subsets to test
# queries_subsets = [ 1, 10, 100, 1000, 10000 ]
queries_subsets = [ 10000 ]

# Path to ground truth 
groundtruth_file = "from_competition_site/deep_new_groundtruth.public.10K.bin"

# Indicate if you want to search from a file path or a loaded array.
use_file_path_for_search = False

# Indicate if you need to load the dataset.
use_load_unload_api = False

# Indicate if you want to deallocate at the end.
deallocate = False

# Set to an existing dataset_id to avoid import.  Set to None for import.
dataset_id = '3cd5ed17-db3d-4e9b-9609-c33f1898db55' 

# Set to an existing allocation id to avoid allocation. Set to None to allocate.
allocation_id = '9e570096-fad6-11eb-a67d-0242ac110002'

# Gemini search params.
gsearchparams = [-1]

# Number of times for each search parameter set.
#count = 3
count = 3

# The nearest neighbors to retrieve.  Max of 100.
ks = [ 10 ] #, 50, 100 ]

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
output = "./deep1B_results_%s.csv" % str(time.time()) 

# Print more intermediary results for debug purposes.
DEBUG = True

# Function that compute the standard recall@N.
def compute_recall(a, b):
    nq, rank = a.shape
    intersect = [ numpy.intersect1d(a[i, :rank], b[i, :rank]).size for i in range(nq) ]
    ninter = sum( intersect )
    return ninter / a.size, intersect

# Needed by faiss ivf algo.
def unwind_index_ivfprev(index):
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

# Needed by faiss ivf algo.
def rate_limited_iterprev(l):
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


# This class wraps the faiss gpu index support.
class IndexQuantizerOnGPUPrev:

    def __init__(self, index, search_bs):
        self.search_bs = search_bs
        index_ivf, vec_transform = unwind_index_ivf(index)
        self.index_ivf = index_ivf
        self.vec_transform = vec_transform
        self.quantizer_gpu = faiss.index_cpu_to_all_gpus(self.index_ivf.quantizer)

    def search(self, x, k):
        bs = self.search_bs
        if self.vec_transform:
            x = self.vec_transform(x)
        nprobe = self.index_ivf.nprobe
        n, d = x.shape
        assert self.index_ivf.d == d
        D = np.empty((n, k), dtype=np.float32)
        I = np.empty((n, k), dtype=np.int64)
        i0 = 0
        ivf_stats = faiss.cvar.indexIVF_stats

        def produce_batches():
            for i0 in range(0, n, bs):
                xblock = x[i0:i0 + bs]
                t0 = time.time()
                D, I = self.quantizer_gpu.search(xblock, nprobe)
                ivf_stats.quantization_time += 1000 * (time.time() - t0)
                yield i0, xblock, D, I

        sp = faiss.swig_ptr
        stage2 = rate_limited_iter(produce_batches())
        t0 = time.time()
        for i0, xblock, Dc, Ic in stage2:
            ni = len(xblock)
            self.index_ivf.search_preassigned(
                ni, faiss.swig_ptr(xblock),
                k, sp(Ic), sp(Dc),
                sp(D[i0:]), sp(I[i0:]),
                False
            )
        ivf_stats.quantization_time += 1000 * (time.time() - t0)

        return D, I

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
        print("imbalance_factor=", index_ivf.invlists.imbalance_factor())

    if no_precomputed_tables:
        if isinstance(index_ivf, faiss.IndexIVFPQ):
            print("disabling precomputed table")
            index_ivf.use_precomputed_table = -1
            index_ivf.precomputed_table.clear()

    precomputed_table_size = 0
    if hasattr(index_ivf, 'precomputed_table'):
        precomputed_table_size = index_ivf.precomputed_table.size() * 4
    print("precomputed tables size:", precomputed_table_size)

    # prep for the searches

    if searchthreads == -1:
        print("Search threads:", faiss.omp_get_max_threads())
    else:
        print("Setting nb of threads to", searchthreads)
        faiss.omp_set_num_threads(searchthreads)

    if parallel_mode != -1:
        print("setting IVF parallel mode to", parallel_mode)
        index_ivf.parallel_mode
        index_ivf.parallel_mode = parallel_mode

    ps = faiss.ParameterSpace()
    ps.initialize(index)

    index_wrap = IndexQuantizerOnGPU(index, search_bs)

    cpu_index = index
    index = index_wrap

    return [ps, cpu_index, index]

try:

    if num_of_boards==0:
        print("Gemini benchmarking has been disabled.")

    else:
        print("Capturing benchmarks for gemini...")

        # Import dataset if needed
        if not dataset_id:
            if DEBUG: print("Importing dataset", dataset_path)
            response = gsi_datasets_apis.apis_import_dataset(body=ImportDatasetRequest(
                ds_file_path=dataset_path,
                ds_file_type="npy",
                train_ind=True))
            dataset_id = response.dataset_id
        if DEBUG: print("Using dataset_id=", dataset_id)

        # Allocate boards if needed
        if not allocation_id:
            if DEBUG: print("Allocating..")
            response = gsi_boards_apis.apis_allocate(body=AllocateRequest(
                    num_of_boards=num_of_boards,
                    max_num_of_threads=num_of_boards*4+4))
            allocation_id = response.allocation_id
        if DEBUG: print("Using allocation_id=",allocation_id)

        # Load dataset if needed
        if use_load_unload_api:

            # TODO: Not sure how to use the API to select an active set of centroids
            # TODO: but that should be included in here.

            if DEBUG: print("Loading dataset...", dataset_id)
            gsi_datasets_apis.apis_load_dataset(body=LoadDatasetRequest(allocation_id, dataset_id))
            if DEBUG: print("Load done")
            
        # Iterate across the queries sets...
        for idx, queries_size in enumerate(queries_subsets):
            
            if DEBUG: print("Query set size=", queries_size)

            # Get the query set file.
            queries_path = os.path.join( path_to_data, queries_file )
            if DEBUG: print("Using queries from %s" % queries_path )

            # Load the queries array as needed.
            if (not use_file_path_for_search):         
                #X = numpy.fromfile(queries_path,dtype=numpy.float32).reshape(queries_size, 96 )
                X = np.load( queries_path )

                # truncate based on the current subset test
                X = X[:queries_size,:]

            if DEBUG: print("X shape", X.shape)

            # Get the ground truth data file.
            gt_file = os.path.join( path_to_data, groundtruth_file )
            if DEBUG: print("Using groundtruth file %s" % gt_file)

            # Iterate across a set of k neighbors to retrieve...
            for k in ks:

                if DEBUG: print("Using k=", k )

                # Load the ground truth array from file.
                n, d = map(int, np.fromfile(gt_file, dtype="uint32", count=2))
                f = open(gt_file, "rb")
                f.seek(4+4)
                ground_truth = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
                if DEBUG: print(n,d, ground_truth.shape )

                # truncate based on the current subset test and k test
                ground_truth = ground_truth[:queries_size,:k]
                if DEBUG: print("Grouth truth dimensions and shape=", ground_truth.shape, ground_truth.dtype )

                # Iterate across search paramters.
                for params in gsearchparams:

                    # Perform the benchmark 'count' times and use the best latency.
                    best_latency = math.inf
                    for c in range(count):

                        if DEBUG: print("Performing search")

                        if (use_file_path_for_search):
                            t0 = time.time()
                            search_api_response = gsi_search_apis.apis_search(body=SearchRequest(allocation_id, dataset_id, queries_file_path=queries_path, topk=k))
                            t1 = time.time()
                            diff = t1-t0
                            if DEBUG: print("Latency:", diff)
                            if diff<best_latency: best_latency = diff
                            indices = search_api_response.indices
                            distance = search_api_response.distance
                            metadata = search_api_response.metadata
                            if DEBUG: print("Search done", len(indices), type(indices))
                        else:
                            if DEBUG: print("Queries shape=", X.shape)
                            pyX = X.tolist()
                            t0 = time.time()
                            print("about to call queries_list...")
                            search_api_response = gsi_search_apis.apis_search_by_queries_list(body=SearchByQueriesListRequest(allocation_id, 
                                    dataset_id, queries=pyX, topk=k))
                            print("after the call queries_list...")
                            t1 = time.time()
                            diff = t1 - t0
                            if DEBUG: print("Latency:", diff)
                            if diff<best_latency: best_latency = diff
                            indices = search_api_response.indices
                            distance = search_api_response.distance
                            metadata = search_api_response.metadata
                            if DEBUG: print("Search done", len(indices), type(indices))

                    # convert search results to numpy array
                    npIndices = numpy.array(indices)

                    # compute recall
                    recall, intersection = compute_recall(ground_truth,npIndices)

                    # store results
                    result = [ "apu", queries_size, k, params, best_latency, recall ]
                    results.append( result )
                    print("result=", result)

        # Unload dataset if needed
        if use_load_unload_api:
            if DEBUG: print("Unloading dataset")
            gsi_datasets_apis.apis_unload_dataset(body=UnloadDatasetRequest(allocation_id, dataset_id))
            if DEBUG: print("Done unload")

    # Benchmark faiss if we have an index available
    if findexfile:
        # Do the faiss benchmarks.
        print("Capturing benchmarks for faiss...")

        # Reading faiss index file and configure.
        print("Reading faiss index...")

        #GW index = faiss.read_index(findexfile)
        ret = load_index(findexfile)
        if ret==False:
            raise Exception("Cannot read faiss index.")
        ps, cpu_index, index = ret
        print("Done reading index.")

        if False:
            index_ivf, vec_transform = unwind_index_ivf(index)
            if vec_transform is None:
                vec_transform = lambda x: x
            if index_ivf is not None:
                if DEBUG: print("imbalance_factor=", index_ivf.invlists.imbalance_factor())
            if DEBUG: print("Index size on disk: ", os.stat(findexfile).st_size)
            precomputed_table_size = 0
            if hasattr(index_ivf, 'precomputed_table'):
                precomputed_table_size = index_ivf.precomputed_table.size() * 4
            if DEBUG: print("Precomputed tables size:", precomputed_table_size)

            # Configure the search.
            if fsearchthreads == -1:
                if DEBUG: print("Search threads:", faiss.omp_get_max_threads())
            else:
                if DEBUG: print("Setting nb of threads to", fsearchthreads)
                faiss.omp_set_num_threads(fsearchthreads)
            if fparallelmode != -1:
                if DEBUG: print("Setting IVF parallel mode to", fparallelmode)
                index_ivf.parallel_mode
                index_ivf.parallel_mode = fparallelmode
            ps = faiss.ParameterSpace()
            ps.initialize(index)

        # Do the faiss benchmarks.
        for device in [ "gpu", "cpu" ]:

            print("on %s..." % device)

            # Iterate across the queries sets...
            for idx, queries_size in enumerate(queries_subsets):
                
                if DEBUG: print("Query set size=", queries_size)

                queries_path = os.path.join( path_to_data, queries_file )
                if DEBUG: print("Using queries from %s" % queries_path )

                # Load the queries array from file.
                queries = np.load( queries_path )

                # truncate based on the current subset test
                queries = queries[:queries_size,:]

                # Get the ground truth data file.
                gt_file = os.path.join( path_to_data, groundtruth_file )
                if DEBUG: print("Using groundtruth file %s" % gt_file)

                # Iterate across a set of k neighbors to retrieve...
                for k in ks:

                    if DEBUG: print("Using k=", k )
                    
                    # Load the ground truth array from file.
                    n, d = map(int, np.fromfile(gt_file, dtype="uint32", count=2))
                    f = open(gt_file, "rb")
                    f.seek(4+4)
                    ground_truth = np.fromfile(f, dtype="int32", count=n * d).reshape(n, d)
                    if DEBUG: print("Ground truth=", n,d, ground_truth.shape )

                    # truncate based on the current subset test and k test
                    ground_truth = ground_truth[:queries_size,:k]
                    if DEBUG: print("Grouth truth dimensions and shape=", ground_truth.shape, ground_truth.dtype )

                    # Iterate across search params.
                    for params in fsearchparams:
                       
                        # Apply param to index
                        sparam = "nprobe=%d" % params
                        #GW ps.set_index_parameters(index, params)
                        #GW print("fs",type(cpu_index), sparam )
                        ps.set_index_parameters(cpu_index, sparam)
     
                        # Load index to GPU as needed
                        if device == "gpu":
                            #GW    index_wrap = IndexQuantizerOnGPU(index, fsearchbs)
                            index_wrap = index 
                            #GW print("GPU", type(index_wrap))
                        else:
                            index_wrap = cpu_index
                            #GW print("CPU", type(index_wrap))

                        # Apply param to index
                        #GW ps.set_index_parameters(index, params) 

                        # Reset stats
                        ivf_stats = faiss.cvar.indexIVF_stats
                        ivf_stats.reset()
                   
                        # Perform benchmark 'count' times and use best latency.
                        best_latency = math.inf
                        for c in range(count):

                            if device == "gpu":
                                t0 = time.time()
                                D, I = index_wrap.search(queries, k)
                                t1 = time.time()
                            else:
                                t0 = time.time()
                                D, I = cpu_index.search(queries, k)
                                t1 = time.time()

                            diff = t1-t0
                            if DEBUG: print("Latency:", diff)
                            if diff<best_latency:  best_latency = diff

                        # Compute the recall.
                        recall, intersection = compute_recall(ground_truth[:, :k], I[:, :k])

                        # store results
                        result =  [ device, queries_size, k, params, best_latency, recall ] 
                        results.append( result )
                        print("result=",result)                    
    else:
        print("No faiss index so not benchmarking it.")

    # Write the results
    if results:
        print("Writing results to file=%f", output )
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

finally:

    # Deallocate if needed
    if allocation_id is not None and deallocate:
        if DEBUG: print("Final deallocation", allocation_id)
        gsi_boards_apis.apis_deallocate(body=DeallocateRequest(allocation_id))


