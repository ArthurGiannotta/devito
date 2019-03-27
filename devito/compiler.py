from functools import partial
from hashlib import sha1
from os import environ, path
from time import time
from distutils import version
from subprocess import DEVNULL, CalledProcessError, check_output, check_call
import platform
import warnings
import sys

import numpy.ctypeslib as npct
from codepy.jit import compile_from_string
from codepy.toolchain import GCCToolchain

from devito.exceptions import CompilationError
from devito.logger import debug, warning, error
from devito.parameters import configuration
from devito.tools import (as_tuple, change_directory, filter_ordered,
                          memoized_func, make_tempdir)

__all__ = ['jit_compile', 'load', 'make', 'GNUCompiler']


def sniff_compiler_version(cc):
    """
    Detect the compiler version.

    Adapted from: ::

        https://github.com/OP2/PyOP2/
    """
    try:
        ver = check_output([cc, "--version"]).decode("utf-8")
    except (CalledProcessError, UnicodeDecodeError):
        return version.LooseVersion("unknown")
    except FileNotFoundError:
        error("The `%s` compiler isn't available on this system" % cc)
        sys.exit(1)

    if ver.startswith("gcc"):
        compiler = "gcc"
    elif ver.startswith("clang"):
        compiler = "clang"
    elif ver.startswith("Apple LLVM"):
        compiler = "clang"
    elif ver.startswith("icc"):
        compiler = "icc"
    else:
        compiler = "unknown"

    ver = version.LooseVersion("unknown")
    if compiler in ["gcc", "icc"]:
        try:
            # gcc-7 series only spits out patch level on dumpfullversion.
            ver = check_output([cc, "-dumpfullversion"], stderr=DEVNULL).decode("utf-8")
            ver = '.'.join(ver.strip().split('.')[:3])
            ver = version.StrictVersion(ver)
        except CalledProcessError:
            try:
                ver = check_output([cc, "-dumpversion"], stderr=DEVNULL).decode("utf-8")
                ver = '.'.join(ver.strip().split('.')[:3])
                ver = version.StrictVersion(ver)
            except (CalledProcessError, UnicodeDecodeError):
                pass
        except UnicodeDecodeError:
            pass

    # Pure integer versions (e.g., ggc5, rather than gcc5.0) need special handling
    try:
        ver = version.StrictVersion(float(ver))
    except TypeError:
        pass

    return ver


def sniff_mpi_distro(mpiexec):
    """
    Detect the MPI version.
    """
    try:
        ver = check_output([mpiexec, "--version"]).decode("utf-8")
        if "open-mpi" in ver:
            return 'OpenMPI'
        elif "HYDRA" in ver:
            return 'MPICH'
    except (CalledProcessError, UnicodeDecodeError):
        pass
    return 'unknown'


class Compiler(GCCToolchain):
    """
    Base class for all compiler classes.

    The base class defaults all compiler specific settings to empty lists.
    Preset configurations can be built by inheriting from `Compiler` and setting
    specific flags to the desired values, e.g.:

    .. code-block:: python
      def class MyCompiler(Compiler):
          def __init__(self):
              self.cc = 'mycompiler'
              self.cflags += ['list', 'of', 'all', 'compiler', 'flags']

    The flags that can be set are:
        * :data:`self.cc`
        * :data:`self.cflag`
        * :data:`self.ldflags`
        * :data:`self.include_dirs`
        * :data:`self.libraries`
        * :data:`self.library_dirs`
        * :data:`self.defines`
        * :data:`self.src_ext`
        * :data:`self.so_ext`
        * :data:`self.undefines`

    Parameters
    ----------
    suffix : str, optional
        The JIT compiler version to be used. For example, assuming ``CC=gcc`` and
        ``suffix='4.9'``, the ``gcc-4.9`` will be used as JIT compiler.
    cpp : bool, optional
        If True, JIT compile using a C++ compiler. Defaults to False.
    mpi : bool, optional
        If True, JIT compile using an MPI compiler. Defaults to False.
    """

    fields = {'cc', 'ld'}

    def __init__(self, **kwargs):
        super(Compiler, self).__init__(**kwargs)

        self.__lookup_cmds__()

        self.suffix = kwargs.get('suffix')
        if not kwargs.get('mpi'):
            self.cc = self.CC if kwargs.get('cpp', False) is False else self.CXX
            self.cc = self.cc if self.suffix is None else ('%s-%s' %
                                                           (self.cc, self.suffix))
        else:
            self.cc = self.MPICC if kwargs.get('cpp', False) is False else self.MPICXX
        self.ld = self.cc  # Wanted by the superclass

        self.cflags = ['-O3', '-g', '-fPIC', '-Wall', '-std=c99']
        self.ldflags = ['-shared']

        self.include_dirs = []
        self.libraries = []
        self.library_dirs = []
        self.defines = []
        self.undefines = []

        self.src_ext = 'c' if kwargs.get('cpp', False) is False else 'cpp'

        if platform.system() == "Linux":
            self.so_ext = '.so'
        elif platform.system() == "Darwin":
            self.so_ext = '.dylib'
        elif platform.system() == "Windows":
            self.so_ext = '.dll'
        else:
            raise NotImplementedError("Unsupported platform %s" % platform)

        if self.suffix is not None:
            try:
                self.version = version.StrictVersion(str(float(self.suffix)))
            except (TypeError, ValueError):
                self.version = version.LooseVersion(self.suffix)
        else:
            # Knowing the version may still be useful to pick supported flags
            self.version = sniff_compiler_version(self.CC)

    def __lookup_cmds__(self):
        self.CC = 'unknown'
        self.CXX = 'unknown'
        self.MPICC = 'unknown'
        self.MPICXX = 'unknown'

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "DevitoJITCompiler[%s]" % self.__class__.__name__

    def __getstate__(self):
        # The superclass would otherwise only return a subset of attributes
        return self.__dict__

    def add_include_dirs(self, dirs):
        self.include_dirs = filter_ordered(self.include_dirs + list(as_tuple(dirs)))

    def add_library_dirs(self, dirs):
        self.library_dirs = filter_ordered(self.library_dirs + list(as_tuple(dirs)))

    def add_libraries(self, libs):
        self.libraries = filter_ordered(self.libraries + list(as_tuple(libs)))

    def add_ldflags(self, flags):
        self.ldflags = filter_ordered(self.ldflags + list(as_tuple(flags)))


class GNUCompiler(Compiler):

    def __init__(self, *args, **kwargs):
        super(GNUCompiler, self).__init__(*args, **kwargs)

        self.cflags += ['-march=native', '-Wno-unused-result', '-Wno-unused-variable',
                        '-Wno-unused-but-set-variable']

        try:
            if self.version >= version.StrictVersion("4.9.0"):
                # Append the openmp flag regardless of configuration['openmp'],
                # since GCC4.9 and later versions implement OpenMP 4.0, hence
                # they support `#pragma omp simd`
                self.ldflags += ['-fopenmp']
        except (TypeError, ValueError):
            if configuration['openmp']:
                self.ldflags += ['-fopenmp']

    def __lookup_cmds__(self):
        self.CC = 'gcc'
        self.CXX = 'g++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


class ClangCompiler(Compiler):

    def __init__(self, *args, **kwargs):
        super(ClangCompiler, self).__init__(*args, **kwargs)
        self.cflags += ['-Wno-unused-result', '-Wno-unused-variable']
        if configuration['platform'].name in ['power8', 'power9']:
            # -march isn't supported on power architectures
            self.cflags += ['-mcpu=native']
        else:
            self.cflags += ['-march=native']

    def __lookup_cmds__(self):
        self.CC = 'clang'
        self.CXX = 'clang++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


class IntelCompiler(Compiler):

    def __init__(self, *args, **kwargs):
        super(IntelCompiler, self).__init__(*args, **kwargs)
        self.cflags += ["-xhost"]
        if configuration['platform'].name == 'skx':
            # Systematically use 512-bit vectors on skylake
            self.cflags += ["-qopt-zmm-usage=high"]
        try:
            if self.version >= version.StrictVersion("15.0.0"):
                # Append the OpenMP flag regardless of configuration['openmp'],
                # since icc15 and later versions implement OpenMP 4.0, hence
                # they support `#pragma omp simd`
                self.ldflags += ['-qopenmp']
        except (TypeError, ValueError):
            if configuration['openmp']:
                # Note: fopenmp, not qopenmp, is what is needed by icc versions < 15.0
                self.ldflags += ['-fopenmp']

        # Make sure the MPI compiler uses `icc` underneath -- whatever the MPI distro is
        if kwargs.get('mpi'):
            ver = check_output([self.MPICC, "--version"]).decode("utf-8")
            if not ver.startswith("icc"):
                warning("The MPI compiler `%s` doesn't use the Intel "
                        "C/C++ compiler underneath" % self.MPICC)

    def __lookup_cmds__(self):
        self.CC = 'icc'
        self.CXX = 'icpc'

        # On some systems, the Intel distribution of MPI may be available, in
        # which case the MPI compiler may be shipped either as `mpiicc` or `mpicc`.
        # On other systems, there may be no Intel distribution of MPI available,
        # thus the MPI compiler is expected to be the classic `mpicc`. Here,
        # we try to use `mpiicc` first, while `mpicc` is our fallback, which may
        # or may not be an Intel distribution
        try:
            check_output(["mpiicc", "--version"]).decode("utf-8")
            self.MPICC = 'mpiicc'
            self.MPICXX = 'mpiicpc'
        except FileNotFoundError:
            self.MPICC = 'mpicc'
            self.MPICXX = 'mpicxx'


class IntelKNLCompiler(IntelCompiler):

    def __init__(self, *args, **kwargs):
        super(IntelKNLCompiler, self).__init__(*args, **kwargs)
        self.cflags += ["-xMIC-AVX512"]
        if not configuration['openmp']:
            warning("Running on Intel KNL without OpenMP is highly discouraged")


class CustomCompiler(Compiler):
    """
    Custom compiler based on standard environment flags.

    Notes
    -----
    Currently honours CC, CFLAGS and LDFLAGS, with defaults similar to the
    default GNU/gcc settings. If DEVITO_ARCH is enabled, the OpenMP linker
    flags are read from OMP_LDFLAGS or otherwise default to ``-fopenmp``.
    """

    CC = environ.get('CC', 'gcc')
    CXX = environ.get('CXX', 'g++')
    MPICC = environ.get('MPICC', 'mpicc')
    MPICXX = environ.get('MPICXX', 'mpicxx')

    def __init__(self, *args, **kwargs):
        super(CustomCompiler, self).__init__(*args, **kwargs)
        default = '-O3 -g -march=native -fPIC -Wall -std=c99'
        self.cflags = environ.get('CFLAGS', default).split(' ')
        self.ldflags = environ.get('LDFLAGS', '-shared').split(' ')
        if configuration['openmp']:
            self.ldflags += environ.get('OMP_LDFLAGS', '-fopenmp').split(' ')

    def __lookup_cmds__(self):
        self.CC = 'gcc'
        self.CXX = 'g++'
        self.MPICC = 'mpicc'
        self.MPICXX = 'mpicxx'


@memoized_func
def get_jit_dir():
    """A deterministic temporary directory for jit-compiled objects."""
    return make_tempdir('jitcache')


@memoized_func
def get_codepy_dir():
    """A deterministic temporary directory for the codepy cache."""
    return make_tempdir('codepy')


def load(soname):
    """
    Load a compiled shared object.

    Parameters
    ----------
    soname : str
        Name of the .so file (w/o the suffix).

    Returns
    -------
    obj
        The loaded shared object.
    """
    return npct.load_library(str(get_jit_dir().joinpath(soname)), '.')


def save(soname, binary, compiler):
    """
    Store a binary into a file within a temporary directory.

    Parameters
    ----------
    soname : str
        Name of the .so file (w/o the suffix).
    binary : obj
        The binary data.
    compiler : Compiler
        The toolchain used for JIT compilation.
    """
    sofile = get_jit_dir().joinpath(soname).with_suffix(compiler.so_ext)
    if sofile.is_file():
        debug("%s: `%s` was not saved in `%s` as it already exists"
              % (compiler, sofile.name, get_jit_dir()))
    else:
        with open(str(sofile), 'wb') as f:
            f.write(binary)
        debug("%s: `%s` successfully saved in `%s`"
              % (compiler, sofile.name, get_jit_dir()))


def jit_compile(soname, code, compiler):
    """
    JIT compile some source code given as a string.

    This function relies upon codepy's ``compile_from_string``, which performs
    caching of compilation units and avoids potential race conditions due to
    multiple processing trying to compile the same object.

    Parameters
    ----------
    soname : str
        Name of the .so file (w/o the suffix).
    code : str
        The source code to be JIT compiled.
    compiler : Compiler
        The toolchain used for JIT compilation.
    """
    target = str(get_jit_dir().joinpath(soname))
    src_file = "%s.%s" % (target, compiler.src_ext)

    cache_dir = get_codepy_dir().joinpath(soname[:7])
    if configuration['jit-backdoor'] is False:
        # Typically we end up here
        # Make a suite of cache directories based on the soname
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        # Warning: dropping `code` on the floor in favor to whatever is written
        # within `src_file`
        try:
            with open(src_file, 'r') as f:
                code = f.read()
            # Bypass the devito JIT cache
            # Note: can't simply use Python's `mkdtemp()` as, with MPI, different
            # ranks would end up creating different cache dirs
            cache_dir = cache_dir.joinpath('jit-backdoor')
            cache_dir.mkdir(parents=True, exist_ok=True)
        except FileNotFoundError:
            raise ValueError("Trying to use the JIT backdoor for `%s`, but "
                             "the file isn't present" % src_file)

    # `catch_warnings` suppresses codepy complaining that it's taking
    # too long to acquire the cache lock. This warning can only appear
    # in a multiprocess session, typically (but not necessarily) when
    # many processes are frequently attempting jit-compilation (e.g.,
    # when running the test suite in parallel)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        tic = time()
        # Spinlock in case of MPI
        sleep_delay = 0 if configuration['mpi'] else 1
        _, _, _, recompiled = compile_from_string(compiler, target, code, src_file,
                                                  cache_dir=cache_dir,
                                                  debug=configuration['debug-compiler'],
                                                  sleep_delay=sleep_delay)
        toc = time()

    if recompiled:
        debug("%s: compiled `%s` [%.2f s]" % (compiler, src_file, toc-tic))
    else:
        debug("%s: cache hit `%s` [%.2f s]" % (compiler, src_file, toc-tic))


def make(loc, args):
    """Invoke the ``make`` command from within ``loc`` with arguments ``args``."""
    hash_key = sha1((loc + str(args)).encode()).hexdigest()
    logfile = path.join(get_jit_dir(), "%s.log" % hash_key)
    errfile = path.join(get_jit_dir(), "%s.err" % hash_key)

    tic = time()
    with change_directory(loc):
        with open(logfile, "w") as lf:
            with open(errfile, "w") as ef:

                command = ['make'] + args
                lf.write("Compilation command:\n")
                lf.write(" ".join(command))
                lf.write("\n\n")
                try:
                    check_call(command, stderr=ef, stdout=lf)
                except CalledProcessError as e:
                    raise CompilationError('Command "%s" return error status %d. '
                                           'Unable to compile code.\n'
                                           'Compile log in %s\n'
                                           'Compile errors in %s\n' %
                                           (e.cmd, e.returncode, logfile, errfile))
    toc = time()
    debug("Make <%s>: run in [%.2f s]" % (" ".join(args), toc-tic))


compiler_registry = {
    'custom': CustomCompiler,
    'gnu': GNUCompiler,
    'gcc': GNUCompiler,
    'clang': ClangCompiler,
    'osx': ClangCompiler,
    'intel': IntelCompiler,
    'icpc': IntelCompiler,
    'icc': IntelCompiler,
    'intel-knl': IntelKNLCompiler,
    'knl': IntelKNLCompiler,
}
"""
Registry dict for deriving Compiler classes according to the environment variable
DEVITO_ARCH. Developers should add new compiler classes here.
"""
compiler_registry.update({'gcc-%s' % i: partial(GNUCompiler, suffix=i)
                          for i in ['4.9', '5', '6', '7', '8']})
