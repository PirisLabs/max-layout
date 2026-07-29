"""Shared hardware acceleration settings used by every solver window."""
import os
import platform

CPU_COUNT=max(1,os.cpu_count() or 1)
DEFAULT_THREADS=max(1,min(CPU_COUNT,int(os.environ.get("PHOTONIC_CPU_THREADS",8))))


def gpu_backend():
 """Return the available array GPU backend without making it mandatory."""
 if os.environ.get("PHOTONIC_USE_GPU","1").lower() in {"0","false","no","off"}:return None,"Disabled"
 try:
  import cupy as cp
  if cp.cuda.runtime.getDeviceCount()>0:return cp,"CUDA GPU"
 except Exception:
  pass
 try:
  import torch
  if torch.backends.mps.is_built() and torch.backends.mps.is_available():return torch,"Apple Metal GPU (PyTorch MPS)"
 except Exception:
  pass
 if platform.system()=="Darwin" and platform.machine()=="arm64":
  return None,"Apple Metal GPU detected; PyTorch MPS runtime is not installed"
 return None,"No compatible CUDA GPU backend detected"


def configure(threads=None):
 global DEFAULT_THREADS
 threads=max(1,int(threads or DEFAULT_THREADS));value=str(threads)
 DEFAULT_THREADS=threads
 for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"):
  os.environ[name]=value
 os.environ["PHOTONIC_CPU_THREADS"]=value
 try:
  from threadpoolctl import threadpool_limits
  threadpool_limits(limits=threads)
 except Exception:
  pass
 return threads

configure()
