# from setuptools import find_packages, setup

# setup(
#    name='anisonet',
#    packages=find_packages(),
# )


from setuptools import setup

setup(
    name='anisonet',
    version='0.1',    
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Arash Golmohammadi',
    author_email='arashgmn@gmail.com',
    license='MIT',
    packages=['anisonet'],
    install_requires=[
        'numpy',
		'matplotlib',
		'brian2==2.6',
        'scipy',
        'pandas',
        'scikit-learn',
		'noise',             
    ],

    # classifiers=[
    #     'Development Status :: 2 - developing',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: BSD License',  
    #     'Operating System :: POSIX :: Linux',        
    #     'Programming Language :: Python :: 3.10',
    # ],
)
