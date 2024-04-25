#! /bin/bash
#SBATCH --job-name=nn
#SBATCH --output=nn.out
#SBATCH --error=nn.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=EPYC_7763
#SBATCH --time=01:00:00

export OMP_PROC_BIND=true
module load gcc python

max_n_threads=8
base_res_exp=10 #6
max_res_exp=10
mesh_width=0.01
nruns=1

cd build
make clean
make
# cd ..

echo "max_n_threads = $max_n_threads"
echo "npasses = $nruns"

for((nthreads = 1; nthreads <= max_n_threads; nthreads*=2)); do
  export OMP_NUM_THREADS=$nthreads
  echo "--- $nthreads Threads ---"
  for((exp = $base_res_exp; exp <= max_res_exp; ++exp)); do
    resolution=$((2**($exp)))
    for((run = 0; run < $nruns; ++run)); do
      result=$(./SharedMemoryNearestNeighbor $resolution $mesh_width) 
      echo "$result"
    done
  done
done

cd ..
python visualize.py
