# Prerequisites

```
conda install -c weinbe58 quspin omp # only for benchmarking, not really required
conda install cython
pip install mpi4py --no-cache-dir
conda install yaml
pip install pyyaml
```

jax: https://github.com/google/jax


# Build c++ sources
```
python cpp_code/setup.py build_ext -i
```

Run code:
```
simulation/main.py
```

Model parameters in `somulation/config_params.yaml`
