# Copyright (c) 2023, NVIDIA CORPORATION.
from pynvjitlink.api import NvJitLinker, NvJitLinkError

import os
import pathlib

_numba_version_ok = False
_numba_error = None

required_numba_ver = (0, 58)

mvc_docs_url = (
    "https://numba.readthedocs.io/en/stable/cuda/" "minor_version_compatibility.html"
)

try:
    import numba

    ver = numba.version_info.short
    if ver < required_numba_ver:
        _numba_error = (
            f"version {numba.__version__} is insufficient for "
            "patching - %s.%s is needed." % required_numba_ver
        )
    else:
        _numba_version_ok = True
except ImportError as ie:
    _numba_error = f"failed to import Numba: {ie}."

if _numba_version_ok:
    from numba.core import config
    from numba.cuda.cudadrv import nvrtc
    from numba.cuda.cudadrv.driver import (
        driver,
        FILE_EXTENSION_MAP,
        Linker,
        LinkerError,
    )

    if ver >= (0, 59):
        from numba.cuda.codegen import CUDACodeLibrary, JITCUDACodegen
        from numba.cuda.cudadrv import devices, nvvm
    else:
        CUDACodeLibrary = object
        JITCUDACodegen = object
else:
    # Prevent the definitions of patched classes failing if we have no Numba
    # equivalents - they won't be used anyway.
    Linker = object
    CUDACodeLibrary = object
    JITCUDACodegen = object


class PatchedLinker(Linker):
    def __init__(
        self,
        max_registers=None,
        lineinfo=False,
        cc=None,
        lto=False,
        additional_flags=None,
    ):
        if cc is None:
            raise RuntimeError("PatchedLinker requires CC to be specified")
        if not any(isinstance(cc, t) for t in [list, tuple]):
            raise TypeError("`cc` must be a list or tuple of length 2")

        sm_ver = f"{cc[0] * 10 + cc[1]}"
        arch = f"-arch=sm_{sm_ver}"
        options = [arch]
        if max_registers:
            options.append(f"-maxrregcount={max_registers}")
        if lineinfo:
            options.append("-lineinfo")
        if lto:
            options.append("-lto")
        if additional_flags is not None:
            options.extend(additional_flags)

        self._linker = NvJitLinker(*options)
        self.options = options

    @property
    def info_log(self):
        return self._linker.info_log

    @property
    def error_log(self):
        return self._linker.error_log

    def add_ptx(self, ptx, name="<cudapy-ptx>"):
        self._linker.add_ptx(ptx, name)

    def add_fatbin(self, fatbin, name="<external-fatbin>"):
        self._linker.add_fatbin(fatbin, name)

    def add_ltoir(self, ltoir, name="<external-ltoir>"):
        self._linker.add_ltoir(ltoir, name)

    def add_object(self, obj, name="<external-object>"):
        self._linker.add_object(obj, name)

    def add_file(self, path, kind):
        try:
            with open(path, "rb") as f:
                data = f.read()
        except FileNotFoundError:
            raise LinkerError(f"{path} not found")

        name = pathlib.Path(path).name
        if kind == FILE_EXTENSION_MAP["cubin"]:
            fn = self._linker.add_cubin
        elif kind == FILE_EXTENSION_MAP["fatbin"]:
            fn = self._linker.add_fatbin
        elif kind == FILE_EXTENSION_MAP["a"]:
            fn = self._linker.add_library
        elif kind == FILE_EXTENSION_MAP["ptx"]:
            return self.add_ptx(data, name)
        elif kind == FILE_EXTENSION_MAP["o"]:
            fn = self._linker.add_object
        else:
            raise LinkerError(f"Don't know how to link {kind}")

        try:
            fn(data, name)
        except NvJitLinkError as e:
            raise LinkerError from e

    def add_cu(self, cu, name):
        with driver.get_active_context() as ac:
            dev = driver.get_device(ac.devnum)
            cc = dev.compute_capability

        ptx, log = nvrtc.compile(cu, name, cc)

        if config.DUMP_ASSEMBLY:
            print(("ASSEMBLY %s" % name).center(80, "-"))
            print(ptx)
            print("=" * 80)

        # Link the program's PTX using the normal linker mechanism
        ptx_name = os.path.splitext(name)[0] + ".ptx"
        self.add_ptx(ptx.encode(), ptx_name)

    def complete(self):
        try:
            cubin = self._linker.get_linked_cubin()
            self._linker._complete = True
            return cubin
        except NvJitLinkError as e:
            raise LinkerError from e


def new_patched_linker(
    max_registers=0, lineinfo=False, cc=None, lto=False, additional_flags=None
):
    return PatchedLinker(
        max_registers=max_registers,
        lineinfo=lineinfo,
        cc=cc,
        lto=lto,
        additional_flags=additional_flags,
    )


class LTOCUDACodeLibrary(CUDACodeLibrary):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ltoir_cache = {}

    def get_ltoir(self, cc=None):
        if cc is None:
            device = devices.get_context().device
            cc = device.compute_capability

        ltoir = self._ltoir_cache.get(cc, None)
        if ltoir:
            return ltoir

        arch = nvvm.get_arch_option(*cc)
        options = self._nvvm_options.copy()
        options['arch'] = arch
        options['gen-lto'] = None

        irs = self.llvm_strs
        ltoir = nvvm.llvm_to_ptx(irs, **options)
        self._ltoir_cache[cc] = ltoir

        return ltoir


class JITLTOCUDACodegen(JITCUDACodegen):
    _library_class = LTOCUDACodeLibrary


def patch_numba_linker():
    if not _numba_version_ok:
        msg = f"Cannot patch Numba: {_numba_error}"
        raise RuntimeError(msg)

    Linker.new = new_patched_linker


def patch_numba_for_lto():
    if not _numba_version_ok:
        msg = f"Cannot patch Numba: {_numba_error}"
        raise RuntimeError(msg)

    if ver < (0, 59):
        msg = f"Numba version 0.59 needed for LTO - you have {ver[0]}.{ver[1]}"
        raise RuntimeError(msg)

    from numba.cuda.descriptor import cuda_target
    lto_codegen = JITLTOCUDACodegen("numba.cuda.jit")
    cuda_target.target_context._internal_codegen = lto_codegen

    # Next step - try this out, see if patching appears to be ok.
    # Then, update it to make LTO possible / optional
