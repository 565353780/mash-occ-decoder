import os
import numpy
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension

lib_root_folder_path = os.getcwd() + "/../mash-occ-decoder/mash_occ_decoder/Lib/"

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    "mash_occ_decoder.Lib.libmcubes.mcubes",
    sources=[
        lib_root_folder_path + "libmcubes/mcubes.pyx",
        lib_root_folder_path + "libmcubes/pywrapper.cpp",
        lib_root_folder_path + "libmcubes/marchingcubes.cpp",
    ],
    language="c++",
    extra_compile_args=["-std=c++11"],
    include_dirs=[numpy_include_dir],
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    "mash_occ_decoder.Lib.libmesh.triangle_hash",
    sources=[lib_root_folder_path + "libmesh/triangle_hash.pyx"],
    libraries=["m"],  # Unix-like specific
    include_dirs=[numpy_include_dir],
)

# mise (efficient mesh extraction)
mise_module = Extension(
    "mash_occ_decoder.Lib.libmise.mise",
    sources=[lib_root_folder_path + "libmise/mise.pyx"],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    "mash_occ_decoder.Lib.libsimplify.simplify_mesh",
    sources=[lib_root_folder_path + "libsimplify/simplify_mesh.pyx"],
    include_dirs=[numpy_include_dir],
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    "mash_occ_decoder.Lib.libvoxelize.voxelize",
    sources=[lib_root_folder_path + "libvoxelize/voxelize.pyx"],
    libraries=["m"],  # Unix-like specific
)

# Gather all extension modules
ext_modules = [
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(ext_modules=cythonize(ext_modules), cmdclass={"build_ext": BuildExtension})
