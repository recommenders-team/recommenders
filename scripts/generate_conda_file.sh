#!/bin/bash
# This script generates a conda file for python, pyspark, gpu or
# all libraries.
# For generating a conda file for running only python code:
# $ sh generate_conda_file.sh
# For generating a conda file for running python gpu:
# $ sh generate_conda_file.sh --gpu
# For generating a conda file for running pyspark:
# $ sh generate_conda_file.sh --pyspark
# For generating a conda file for running python gpu and pyspark:
# $ sh generate_conda_file.sh --gpu --pyspark
#

# first check if conda is installed
CONDA_BINARY=$(which conda)
if [ -x "$CONDA_BINARY" ] ; then
	echo "Installation script will use this conda binary ${CONDA_BINARY} for installation"
else
	echo "No conda found!! Please see the README.md file for installation prerequisites."
	exit 1
fi


# File which containt conda configuration
# virtual environment name
CONDA_FILE="conda_bare.yaml"

# default CPU-only no-pySpark versions of conda packages.
pytorch="pytorch-cpu"
# TODO: torchvision-cpu does not seem to exist in pytorch channel
torchvision="torchvision"
tensorflow="tensorflow"
pyspark="#"

# flags to detect if both CPU and GPU are specified
gpu_flag=false
pyspark_flag=false

while [ ! $# -eq 0 ]
do
	case "$1" in
		--help)
			echo "Please specify --gpu to install with GPU-support and"
			echo "--pyspark to install with pySpark support"
			exit
			;;
		--gpu)
			pytorch="pytorch"
			torchvision="torchvision"
			tensorflow="tensorflow-gpu"
			CONDA_FILE="conda_gpu.yaml"
			gpu_flag=true
			;;
		--pyspark)
			pyspark=""
			CONDA_FILE="conda_pyspark.yaml"
			pyspark_flag=true
			;;			
		*)
			echo $"Usage: $0 invalid argument $1 please run with --help for more information."
			exit 1
	esac
	shift
done

if [ "$pyspark_flag" = true ] && [ "$gpu_flag" = true ]; then
	CONDA_FILE="conda_full.yaml"
fi

/bin/cat <<EOM >${CONDA_FILE}
# To create the conda environment:
# $ conda env create -n my_env_name -f conda.yml
#
# To update the conda environment:
# $ conda env update -n my_env_name -f conda.yaml
#
# To register the conda environment in Jupyter:
# $ python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"
#

channels:
- pytorch
- conda-forge
- defaults
dependencies:
- jupyter==1.0.0
- python==3.6
- numpy>=1.13.3
- dask>=0.17.1
${pyspark}- pyspark==2.2.0
- pymongo>=3.6.1
- ipykernel>=4.6.1
- ${tensorflow}==1.5.0
- ${pytorch}==0.4.0
- scikit-surprise>=1.0.6
- scikit-learn==0.19.1
- jupyter>=1.0.0
- fastparquet>=0.1.6
- pip:
  - pandas>=0.22.0
  - scipy>=1.0.0
  - azure-storage>=0.36.0
  - tffm==1.0.1
  - pytest==3.6.4
  - pytest-cov
  - pytest-datafiles>=1.0
  - ${torchvision}
  - pylint>=2.0.1
  - pytest-pylint==0.11.0
EOM



