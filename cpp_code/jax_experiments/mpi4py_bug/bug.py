from mpi4py import MPI
import jax



seed=7
comm=MPI.COMM_WORLD

print(jax.__version__, seed, comm.Get_rank())

jax.lib.xla_bridge.get_backend().platform
rng = jax.random.PRNGKey(seed)


