from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="titan-py",
    version="0.1.0",
    rust_extensions=[RustExtension("titan_core.titan_core", binding=Binding.PyO3, path="titan_core/Cargo.toml")],
    packages=["titan_py"],
    zip_safe=False,
)
