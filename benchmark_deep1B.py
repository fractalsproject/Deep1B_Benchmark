import swagger_client
from swagger_client.models import *
from swagger_client.rest import ApiException
import sys
import time
import os
import numpy
import math
import faiss

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
num_of_boards = 3

# Path to top-level data directory (queries, groundtruth, faissindex,...)
path_to_data = "/home/george/Projects/Deep1B_Benchmark_Data"

# Path to individual queries and groundtruth under the top-level data dir.
queries_and_groundtruth = [ [ 10, 100, 1000 ],
                            [ "queries/deep1B_queries_from_website_10.npy",
                              "queries/deep1B_queries_from_website_100.npy",
                              "queries/deep1B_queries_from_website_1000.npy"],
                            [ "groundtruth/deep1B_groundtruth_from_website_10.npy",
                              "groundtruth/deep1B_groundtruth_from_website_100.npy",
                              "groundtruth/deep1B_groundtruth_from_website_1000.npy" ]
                          ]

# Indicate if you want to search from a file path or a loaded array.
use_file_path_for_search = False

# Indicate if you need to load the dataset.
use_load_unload_api = False

# Indicate if you want to deallocate at the end.
deallocate = False

# Set to an existing dataset_id to avoid import.  Set to None for import.
dataset_id = '0e7bdfda-8da9-4693-9bb8-01b79cbd59de'

# Set to an existing allocation id to avoid allocation. Set to None to allocate.
allocation_id = '49ea4c2a-e366-11eb-9751-0242ac110003'

# Gemini search params.
gsearchparams = [-1]

# Number of times for each search parameter set.
count = 3

# The nearest neighbors to retrieve.  Max of 100.
ks = [ 10, 50, 100 ]

# FAISS search batch size
fsearchbs = 8192

# FAISS search parameter set.
fsearchparams =  [ 1, 2, 4, 16, 32, 64, 128, 256 ]

# FAISS index file
findexfile = "/home/george/Projects/Deep1B_Benchmark_Data/deep-1B.IVF1048576,SQ8.faissindex"

# FAISS search threads
fsearchthreads = -1 # -1 to use max threads

# FAISS search parallel mode
fparallelmode = 3

# Will contain the results of the benchmarks.
results = []

# Results will be written to this csv.
output = "./deep1B_results.csv" 

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

# This class wraps the faiss gpu index support.
class IndexQuantizerOnGPU:

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

try:

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
                num_of_boards=1,
                max_num_of_threads=5))
        allocation_id = response.allocation_id
    if DEBUG: print("Using allocation_id=",allocation_id)

    # Load dataset if needed
    if use_load_unload_api:
        if DEBUG: print("Loading dataset...", dataset_id)
        gsi_datasets_apis.apis_load_dataset(body=LoadDatasetRequest(allocation_id, dataset_id))
        if DEBUG: print("Load done")
        
    # Iterate across the queries sets...
    for idx, queries_file in enumerate(queries_and_groundtruth[1]):

        # Get the query set file.
        queries_path = os.path.join( path_to_data, queries_file )
        if DEBUG: print("Using queries from %s" % queries_path )

        # Get number of queries.
        queries_size = queries_and_groundtruth[0][idx]
        if DEBUG: print("Query set size=", queries_size)

        # Load the queries array as needed.
        if (not use_file_path_for_search):         
            X = numpy.fromfile(queries_path,dtype=numpy.float32).reshape(queries_size, 96 )

        # Get the ground truth data file.
        gt_file = os.path.join( path_to_data, queries_and_groundtruth[2][idx] )
        if DEBUG: print("Using groundtruth file %s" % gt_file)

        # Iterate across a set of k neighbors to retrieve...
        for k in ks:

            if DEBUG: print("Using k=", k )

            # Load the ground truth array from file.
            ground_truth = numpy.fromfile(gt_file, dtype=numpy.int32).reshape( queries_size, 100 )[:,:k]
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
                        search_api_response = gsi_search_apis.apis_search_by_queries_list(body=SearchByQueriesListRequest(allocation_id, 
                                dataset_id, queries=pyX, topk=k))
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

    # Do the faiss benchmarks.
    print("Capturing benchmarks for faiss...")

    # Reading faiss index file and configure.
    index = faiss.read_index(findexfile)
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
    for device in [ "cpu", "gpu" ]:

        print("on %s..." % device)

        # Iterate across the queries sets...
        for idx, queries_file in enumerate(queries_and_groundtruth[1]):

            queries_path = os.path.join( path_to_data, queries_file )
            if DEBUG: print("Using queries from %s" % queries_path )

            # Get number of queries.
            queries_size = queries_and_groundtruth[0][idx]
            if DEBUG: print("Query set size=", queries_size)

            # Load the queries array from file.
            queries = numpy.fromfile(queries_path,dtype=numpy.float32).reshape(queries_size, 96 )

            # Get the ground truth data file.
            gt_file = os.path.join( path_to_data, queries_and_groundtruth[2][idx] )
            if DEBUG: print("Using groundtruth file %s" % gt_file)

            # Iterate across a set of k neighbors to retrieve...
            for k in ks:

                if DEBUG: print("Using k=", k )
                
                # Load the ground truth array from file.                    
                ground_truth = numpy.fromfile(gt_file, dtype=numpy.int32).reshape( queries_size, 100 )[:,:k]
                if DEBUG: print("Grouth truth dimensions and shape=", ground_truth.shape, ground_truth.dtype )

                # Iterate across search params.
                for params in fsearchparams:

                    # Load index to GPU as needed
                    if device == "gpu":
                        index_wrap = IndexQuantizerOnGPU(index, fsearchbs)
                    else:
                        index_wrap = index

                    # Set the search params to the index
                    ps.set_index_parameters(index, params)
                    ivf_stats = faiss.cvar.indexIVF_stats
                    ivf_stats.reset()
               
                    # Perform benchmark 'count' times and use best latency.
                    best_latency = math.inf
                    for c in range(count):
                        t0 = time.time()
                        D, I = index.search(queries, k)
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

    # Write the results
    if results:
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


