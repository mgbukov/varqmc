# Prerequisites

```
conda install -c weinbe58 quspin omp # only for benchmarking, not really required
conda install cython
conda install mpi4py # or pip install mpi4py --no-cache-dir 
conda install yaml
pip install pyyaml
```

jax: https://github.com/google/jax


# Build c++ sources (lattice symemtries, etc)
```
python cpp_code/setup.py build_ext -i
```

Run code:
```
simulation/main.py
```

Model parameters in `simulation/config_params.yaml`
