# Brian's anisotropic network

This is a `Brian2` implementation of anisotropic network introduced in [From space to time: Spatial inhomogeneities lead to the emergence of spatiotemporal sequences in spiking neuronal networks](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007432). This packages depends on the following libraries:

- brian2
- numpy
- scipy
- noise (for Perlin noise)
- matplotlib
- pandas, scikit-learn, seaborn, ... (may be removed or expanded during development) 

# Documentation
For further details check the documentation here: [https://arashgmn.github.io/anisonet-brian](https://arashgmn.github.io/anisonet-brian)

# Installation
## Download/clone repository
This repository is under development. Thus, I recommend you to clone this repository so that you can get the updates easier later on. But if you prefer, you can download the repo instead.

## Setting up an environment 
I recommand setting up an isolated Conda environment and install and run this code in that environment. In short, Conda is a package manager by which you can install many libaries independent from one another. There are several benefits to this approach. but for now, just take my words. It is jsut better to use Conda. Here is what you need to do:

1. Go to [Anaconda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) and download the install either Anaconda or Miniconda (**I highly recommand the latter**) for your OS.
2. Open a terminal (or *Anaconda prompt*) and create a new environment: `conda create -n <YOUR_ENV_NAME> python=3.10`. Note that the name of this environment (`<YOUR_ENV_NAME>`) can be anything. Also, we have anchored the python version to `3.10` as our dependencies must be first tested on the newest python version `3.11`.
3. Activate your environment: ``conda activate <YOUR_ENV_NAME>``. 

## Install
1. **Make sure your environment is activated**. In your terminal or Anaconda prompt enter ``conda activate <YOUR_ENV_NAME>``.
2. Update your `setuptools` by entring the following command: ``pip install --upgrade setuptools``
3. Navigate to the root directory of the cloned or downloaded repository within terminal and type ``pip install -e .`` (with the period at the end). 

**Note for Windows machines**: If the last step failed and you got an error indicating that you have to install Visual studio, it is likely that [you don't have the proper c++ compiler installed on your computer](https://wiki.python.org/moin/WindowsCompilers). In short, since we used `python=3.10` you must install version **14.x** of Visual Studio which is a hefy download, unfortunately. As indicated in the referenced link:

> -  Install [Microsoft Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) (or later).
> - Install the Python development workload and the optional Python native development tools option.
> - Install the latest Windows SDK (under Native development in the installer).
> - Optional: Set $env:PlatformToolset to your toolset version before building, if it doesn't detect it.


# How to run
After installation, simply navigate to `anisonet/` and run the following command: ``python demo.py``. This makes a new directory with which contains the results of the *Inhibitory Network* of the aforementioned paper. This code, constructs and warms up the netowrk, and stores the states. It also visualizes landscape, in- and out-degrees, and the realized landscape, and saves the summary of neuronal activity over a 2.5-second simulation. For changing the network configuration and extending to other possible cases, please read the documentation of ``configs`` module.

# Code Structure
We have a main class called `Simulate` which sets up the network and runs the simulation. Other files are dedicated to specified tasks:

- `analyze`: a set of functions to analyze the neuronal activity
- `anisofy`: samples post-synapses in an anisotropic manner
- `configs`: stores some default configuration settings for convenience
- `equations`: prepares Brian-friendly equations for neurons and synapses according to the configuration dictionaries
- `landscape`: prepares the angular and shift ($\phi, r$) landscape
- `utils`: some useful utility functions
- `viz`: a set of functions for visualizing results


# Important Note
This implementation is work in progress and possibly contains bugs! Any feedback is welcome.


# FAQ 
## I don't have git command on my Linux machine
Some of the recent linux distribution don't come with git commands pre-installed. On Debian/Ubuntu, you can install it with ``sudo apt-get install git-all``. Change command for Arch or RedHat appropriately.

## I don't have terminal on my Windows machine to clone the repository
The most prominent workaround is installing [Git-bash](https://git-scm.com/download/win) on Windows systems. It emulates a terminal with some unix-based commands (``ls, cd, cat``, etc.) but more importantly, ``git``. You can use either the GUI or the command line to clone the repository. 
