import kernel_tuner
import numpy as np
from collections import OrderedDict
import json

#rrtmgp_path = '/home/bart/meteo/models/rte-rrtmgp-cpp'  # Home desktop
rrtmgp_path = '/home/ubuntu/models/rte-rrtmgp-cpp'      # AWS
with open('{}/src_kernels_cuda/gas_optics_kernels.cu'.format(rrtmgp_path)) as f:
    kernel_string = f.read()

# Settings
type_int = np.int32
type_float = np.float64
type_bool = np.int32     # = default without `RTE_RRTMGP_USE_CBOOL`

str_float = 'float' if type_float is np.float32 else 'double'

ncol  = type_int(144)    # ??, this is RCEMIP..
nlay  = type_int(200)    # case specific
nband = type_int(16)     # from coefficients_lw.nc (bnd dim)
ngpt  = type_int(256)    # from coefficients_lw.nc (gpt dim)

nflav = type_int(10)     # ??
neta  = type_int(9)      # 
npres = type_int(59)     # from coefficients_lw.nc (pressure dim)
ntemp = type_int(14)     # from coefficients_lw.nc (temperature dim)

# Output (Array_gpu.dump()) from RCEMIP
gpoint_flavor = np.fromfile('bin/gpoint_flavor.bin', dtype=type_int)
band_lims_gpt = np.fromfile('bin/band_lims_gpt.bin', dtype=type_int)
jeta = np.fromfile('bin/jeta.bin', dtype=type_int)
jtemp = np.fromfile('bin/jtemp.bin', dtype=type_int)
jpress = np.fromfile('bin/jpress.bin', dtype=type_int)

kmajor = np.fromfile('bin/kmajor.bin', dtype=type_float)
col_mix = np.fromfile('bin/col_mix.bin', dtype=type_float)
fmajor = np.fromfile('bin/fmajor.bin', dtype=type_float)

tropo = np.fromfile('bin/tropo.bin', dtype=type_bool)

# Dummy data -> does not work; kernel computations depend on input data..
"""
gpoint_flavor = np.ones(512, dtype=type_int)    # = 2 x ngpt?
band_lims_gpt = np.ones(32, dtype=type_int)     # = 2 x nband? -> yes, start and end position
jeta = np.ones(576000, dtype=type_int)          # = ??
jtemp = np.ones(28800, dtype=type_int)          # = ncol x nlay?
jpress = np.ones(28800, dtype=type_int)         # = ncol x nlay?

kmajor = np.zeros(1935360, dtype=type_float)    # = ?
col_mix = np.zeros(576000, dtype=type_float)    # = ?
fmajor = np.zeros(2304000, dtype=type_float)    # = ?
tau = np.zeros(7372800, dtype=type_float)       # = ?
tau_major = np.zeros(7372800, dtype=type_float) # = ?

tropo = np.zeros(28800, dtype=type_bool)        # = ncol x nlay?
"""

# Output from kernel:
tau = np.zeros(7372800, dtype=type_float)
tau_major = np.zeros(7372800, dtype=type_float)

# Reference data (from Array_gpu.dump()):
tau_ref = np.fromfile('bin/tau.bin', dtype=type_float)

# Kernel tuner
args = [ncol, nlay, nband, ngpt,
        nflav, neta, npres, ntemp,
        gpoint_flavor,
        band_lims_gpt,
        kmajor,
        col_mix, fmajor,
        jeta, tropo,
        jtemp, jpress,
        tau, tau_major]

problem_size = (nband, nlay, ncol)
kernel_name = 'compute_tau_major_absorption_kernel<{}>'.format(str_float)
cp = ['-I{}/include'.format(rrtmgp_path)]

# Check results
params = { 'block_size_x': 14,
           'block_size_y': 1,
           'block_size_z': 32}

results = kernel_tuner.run_kernel(
        kernel_name, kernel_string, problem_size,
        args, params, compiler_options=cp)

results_okay = np.allclose(results[-1], tau_ref, atol=1e-15)
max_diff = np.abs(results[-1] - tau_ref).max()

print('Results identical={}, max diff={}'.format(results_okay, max_diff))

# Tune!
tune_params = dict()
tune_params["block_size_x"] = [4,8,14,16,32,64]
tune_params["block_size_y"] = [1]
tune_params["block_size_z"] = [4,8,16,32,64]

answer = len(args)*[None]
answer[-1] = tau_ref

result, env = kernel_tuner.tune_kernel(
        kernel_name, kernel_string, problem_size,
        args, tune_params, compiler_options=cp,
        answer=answer, atol=1e-14)
