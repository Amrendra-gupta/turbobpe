from setuptools import setup, Extension

extensions = [
    Extension(
        "turbobpe.utils",
        sources=["src/turbobpe/utils.c"],
		extra_compile_args=["-O3"],
    )
]

setup(
    ext_modules=extensions,
    package_dir={"": "src"},
)