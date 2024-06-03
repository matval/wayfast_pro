from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]

requirements = parse_requirements_file("requirements.txt")

setup(
    name='energy_traversability',
    version='0.1.0',
    author='Mateus V Gasparino',
    author_email='mvalve2@illinois.edu',
    description='A package for energy-based traversability prediction in robotics',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/matval/energy_traversability',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    ext_modules=[
        CUDAExtension(
            name='voxel_op', 
            sources=['traversability/models/pointpillars/voxelization/voxelization.cpp',
                     'traversability/models/pointpillars/voxelization/voxelization_cpu.cpp',
                     'traversability/models/pointpillars/voxelization/voxelization_cuda.cu',
                    ],
            define_macros=[('WITH_CUDA', None)]    
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
