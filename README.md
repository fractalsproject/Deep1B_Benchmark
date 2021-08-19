# Deep1B_Benchmark

This repository contains various useful benchmark tools and results for Sales and Marketing.

# Results

![Image of Yaktocat](deep1B_compare.png)

# How To Run This Code

## How To Run "benchmark_deep1B_gpu_cpu_only.py":

* You should already have installed CUDA version 11.0 or compatible (higher may be possible, we only tested on 11.0)
* We downloaded and installed the Anaconda distribution of python but other distributions should word
* Create a python environment using python=3.8.5 and activate it
* Install python dependencies using "faiss-gpu_requirements.txt" requirements file
* Edit the benchmark_deep1B_gpu_cpu_only.py with your local settings ( change paths to various index and ground truth files - let me know if you need access to any of these files. )
* Run "python benchmark_deep1B_gpu_cpu_only.py"


## How To Run "benchmark_deep1B.py":

* Install the Gemini APU software and install the Deep1B data clusters
* Mount the Sunnyvale NAS1 file server (ask Hess for access ) and copy the directory /mnt/nas1/George/Benchmarks/Deep1B_Benchmark_Data to your local machine
* Install Anaconda distribution of python
* Create and activate a python=3.8.5 environment.
* Use python 'pip' to install the packages using the requirements file [faiss-gpu_requirements.txt](faiss-gpu_requirements.txt)
* Edit the top of the file [benchmark_deep1B.py](benchmark_deep1B.py) and change the parameters for your setup
* Run the script [benchmark_deep1B.py](benchmark_deep1B.py).  Upon completion this will emit a CSV file with the benchmark data.
* You can use the [Analyze.ipynb](Analyze.ipynb) notebook to load the CSV file, to produce a plot similar to the one above.
