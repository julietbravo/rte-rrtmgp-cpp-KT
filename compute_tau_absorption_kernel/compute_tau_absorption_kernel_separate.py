import kernel_tuner
import numpy as np

# To supress the `int* != BOOL_TYPE` warnings:
import warnings
warnings.filterwarnings("ignore")

#rrtmgp_path = '/home/bart/meteo/models/rte-rrtmgp-cpp'  # Home desktop
rrtmgp_path = '/home/ubuntu/models/rte-rrtmgp-cpp'      # AWS
with open('{}/src_kernels_cuda/gas_optics_kernels.cu'.format(rrtmgp_path)) as f:
    kernel_string = f.read()

# Settings
type_int = np.int32
type_float = np.float64
type_bool = np.int32     # = default without `RTE_RRTMGP_USE_CBOOL`

str_float = 'float' if type_float is np.float32 else 'double'
cp = ['-I{}/include'.format(rrtmgp_path)]

ncol  = type_int(144)    # ??, this is RCEMIP..
nlay  = type_int(200)    # case specific
nband = type_int(16)     # from coefficients_lw.nc (bnd dim)
ngpt  = type_int(256)    # from coefficients_lw.nc (gpt dim)
nflav = type_int(10)     # ??
neta  = type_int(9)      # 
npres = type_int(59)     # from coefficients_lw.nc (pressure dim)
ntemp = type_int(14)     # from coefficients_lw.nc (temperature dim)
ngas  = type_int(7)

nscale_lower = type_int(44)
nscale_upper = type_int(19)
nminor_lower = type_int(44)
nminor_upper = type_int(19)
nminork_lower = type_int(704)
nminork_upper = type_int(304)
idx_h2o = type_int(1)

# Output (Array_gpu.dump()) from RCEMIP
gpoint_flavor = np.fromfile('bin/gpoint_flavor.bin', dtype=type_int)
band_lims_gpt = np.fromfile('bin/band_lims_gpt.bin', dtype=type_int)
jeta = np.fromfile('bin/jeta.bin', dtype=type_int)
jtemp = np.fromfile('bin/jtemp.bin', dtype=type_int)
jpress = np.fromfile('bin/jpress.bin', dtype=type_int)
minor_limits_gpt_lower = np.fromfile('bin/minor_limits_gpt_lower.bin', dtype=type_int)
minor_limits_gpt_upper = np.fromfile('bin/minor_limits_gpt_upper.bin', dtype=type_int)
idx_minor_lower = np.fromfile('bin/idx_minor_lower.bin', dtype=type_int)
idx_minor_upper = np.fromfile('bin/idx_minor_upper.bin', dtype=type_int)
idx_minor_scaling_lower = np.fromfile('bin/idx_minor_scaling_lower.bin', dtype=type_int)
idx_minor_scaling_upper = np.fromfile('bin/idx_minor_scaling_upper.bin', dtype=type_int)
kminor_start_lower = np.fromfile('bin/kminor_start_lower.bin', dtype=type_int)
kminor_start_upper = np.fromfile('bin/kminor_start_upper.bin', dtype=type_int)

kmajor = np.fromfile('bin/kmajor.bin', dtype=type_float)
col_mix = np.fromfile('bin/col_mix.bin', dtype=type_float)
fmajor = np.fromfile('bin/fmajor.bin', dtype=type_float)
fminor = np.fromfile('bin/fminor.bin', dtype=type_float)
kminor_lower = np.fromfile('bin/kminor_lower.bin', dtype=type_float)
kminor_upper = np.fromfile('bin/kminor_upper.bin', dtype=type_float)
play = np.fromfile('bin/play.bin', dtype=type_float)
tlay = np.fromfile('bin/tlay.bin', dtype=type_float)
col_gas = np.fromfile('bin/col_gas.bin', dtype=type_float)

tropo = np.fromfile('bin/tropo.bin', dtype=type_bool)
minor_scales_with_density_lower = np.fromfile('bin/minor_scales_with_density_lower.bin', dtype=type_bool)
minor_scales_with_density_upper = np.fromfile('bin/minor_scales_with_density_upper.bin', dtype=type_bool)
scale_by_complement_lower = np.fromfile('bin/scale_by_complement_lower.bin', dtype=type_bool)
scale_by_complement_upper = np.fromfile('bin/scale_by_complement_upper.bin', dtype=type_bool)

# Output from kernel:
tau = np.zeros(7372800, dtype=type_float)
tau_major = np.zeros(7372800, dtype=type_float)
tau_minor = np.zeros(7372800, dtype=type_float)

# Reference data (from Array_gpu.dump()):
tau_after_minor = np.fromfile('bin/tau_after_minor.bin', dtype=type_float)
tau_after_major = np.fromfile('bin/tau_after_major.bin', dtype=type_float)

# Kernel tuner
args_major = [
        ncol, nlay, nband, ngpt,
        nflav, neta, npres, ntemp,
        gpoint_flavor,
        band_lims_gpt,
        kmajor,
        col_mix, fmajor,
        jeta, tropo,
        jtemp, jpress,
        tau, tau_major]

itropo_lower = type_bool(1)
args_minor_lower = [
        ncol, nlay, ngpt,
        ngas, nflav, ntemp, neta,
        nscale_lower,
        nminor_lower,
        nminork_lower,
        idx_h2o,
        gpoint_flavor,
        kminor_lower,
        minor_limits_gpt_lower,
        minor_scales_with_density_lower,
        scale_by_complement_lower,
        idx_minor_lower,
        idx_minor_scaling_lower,
        kminor_start_lower,
        play,
        tlay,
        col_gas,
        fminor,
        jeta,
        jtemp,
        tropo,
        itropo_lower,
        tau,
        tau_minor]

itropo_upper = type_bool(0)
args_minor_upper = [
        ncol, nlay, ngpt,
        ngas, nflav, ntemp, neta,
        nscale_upper,
        nminor_upper,
        nminork_upper,
        idx_h2o,
        gpoint_flavor,
        kminor_upper,
        minor_limits_gpt_upper,
        minor_scales_with_density_upper,
        scale_by_complement_upper,
        idx_minor_upper,
        idx_minor_scaling_upper,
        kminor_start_upper,
        play,
        tlay,
        col_gas,
        fminor,
        jeta,
        jtemp,
        tropo,
        itropo_upper,
        tau,
        tau_minor]


problem_size_major = (nband, nlay, ncol)
kernel_name_major = 'compute_tau_major_absorption_kernel<{}>'.format(str_float)

problem_size_minor = (nlay, ncol)
kernel_name_minor = 'compute_tau_minor_absorption_kernel_separate<{}>'.format(str_float)

# Check results
def compare_fields(arr1, arr2, name):
    okay = np.allclose(arr1, arr2, atol=1e-15)
    max_diff = np.abs(arr1-arr2).max()
    if okay:
        print('results for {}: OKAY!'.format(name))
    else:
        print('results for {}: NOT OKAY, max diff={}'.format(name, max_diff))

# Major
params = { 'block_size_x': 14,
           'block_size_y': 1,
           'block_size_z': 32}

result = kernel_tuner.run_kernel(
        kernel_name_major, kernel_string, problem_size_major,
        args_major, params, compiler_options=cp)

compare_fields(result[-2], tau_after_major, 'major')

# Minor
params = { 'block_size_x': 32,
           'block_size_y': 32}

# Use output from major as input for minor
tau[:] = result[-2][:]

result = kernel_tuner.run_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_lower, params, compiler_options=cp)

# Store intermediate results for correctness check in tuning
tau_after_minor_sub = result[-2][:]
tau[:] = result[-2][:]

result = kernel_tuner.run_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_upper, params, compiler_options=cp)

compare_fields(result[-2], tau_after_minor, 'minor')

#
# Tune!
#
tune_params_major = dict()
tune_params_major["block_size_x"] = [4,8] # [4,8,16,32]
tune_params_major["block_size_y"] = [4,8] # [1,2,4]
tune_params_major["block_size_z"] = [4,8] # [4,8,16,32]

tune_params_minor = dict()
tune_params_minor["block_size_x"] = [4,6,8,10,12,16]
tune_params_minor["block_size_y"] = [4,6,8,10,12,16]

answer_major = len(args_major)*[None]
answer_major[-2] = tau_after_major

answer_minor_lower = len(args_minor_lower)*[None]
answer_minor_lower[-2] = tau_after_minor_sub

answer_minor_upper = len(args_minor_upper)*[None]
answer_minor_upper[-2] = tau_after_minor

# Reset input tau
tau[:] = 0.

result, env = kernel_tuner.tune_kernel(
        kernel_name_major, kernel_string, problem_size_major,
        args_major, tune_params_major, compiler_options=cp,
        answer=answer_major, atol=1e-14)

tau[:] = tau_after_major

result, env = kernel_tuner.tune_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_lower, tune_params_minor, compiler_options=cp,
        answer=answer_minor_lower, atol=1e-14)

tau[:] = tau_after_minor_sub

result, env = kernel_tuner.tune_kernel(
        kernel_name_minor, kernel_string, problem_size_minor,
        args_minor_upper, tune_params_minor, compiler_options=cp,
        answer=answer_minor_upper, atol=1e-14)
