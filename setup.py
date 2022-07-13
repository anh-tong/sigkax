import os
import re
import pathlib
import setuptools
import subprocess

from setuptools.command.build_ext import build_ext

here = pathlib.Path(__file__).resolve().parent

name = "sigkax"


with open( here / name / "__init__.py") as f:
    match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if match:
        version = match.group(1)
    else:
        RuntimeError("Can not determine version")
        
class CMakeBuildExt(build_ext):
    
    def build_extensions(self) -> None:
        
        import platform
        import sys
        from distutils import sysconfig
        
        import pybind11
        
        if platform.system() == "windows":
            raise RuntimeError("Not support Windows due to JAX")
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME")
            )
        
        cmake_python_include_dir = sysconfig.get_python_inc()
        
        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)
        
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_LIBRARIES={cmake_python_library}",
            f"-DPython_INCLUDE_DIRS={cmake_python_include_dir}",
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
            f"-DCMAKE_PREFIX_PATH={pybind11.get_cmake_dir()}"
        ]
        
        if os.environ.get("SIGKAX_CUDA", "no").lower() == "yes":
            cmake_args.append("-DSIGKAX_CUDA=yes")
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", here] + cmake_args, cwd=self.build_temp
        )
        
        super().build_extensions()
        
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp
        )
        
    def build_extension(self, ext) -> None:
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp
        )
        
extensions = [
    setuptools.Extension("sigkax.cpu_ops", ["sigkax/backends/cpu_ops.cc"])
]

if os.environ.get("SIGKAX_CUDA", "no").lower() == "yes":
    extensions.append(
        setuptools.Extension(
            "sigkax.gpu_ops",
            ["sigkax/backends/gpu_ops.cc",
             "sigkax/backends/cuda_kernels.cc.cu"]
        )
    )
    

python_requires = "~=3.7"
install_requires = ["jax>=0.3.10", "pybind11>=2.6", "cmake"]

setuptools.setup(
    name=name,
    author="Anh Tong",
    url="https://github.com/anh-tong/sigkax",
    version=version,
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt}
    
)
