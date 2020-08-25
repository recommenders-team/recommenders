# Environment setup
The following setup instructions assume users work in a Linux system. The testing was performed on a Ubuntu Linux system.
We use Conda to install packages and manage the virtual environment. Type ``` conda list ``` to check if you have conda in your machine. If not, please follow the instructions on https://conda.io/projects/conda/en/latest/user-guide/install/linux.html to install either Miniconda or Anaconda (preferred) before we proceed. 

1. Clone the repository
    ```bash
    git clone https://github.com/microsoft/recommenders 
    ```

1. Check out the tutorial branch
    ```bash
    cd recommenders
    git checkout kdd2020_tutorial
    ```
    The materials for the tutorial are located under the directory of `recommenders/scenarios/academic/KDD2020-tutorial`.
    ```bash
    cd scenarios/academic/KDD2020-tutorial
    ```
1. Download the dataset
    1. Download the dataset for hands on experiments and unzip to data_folder:
    ```bash
    wget https://recodatasets.blob.core.windows.net/kdd2020/data_folder.zip
    unzip data_folder.zip -d data_folder
    ```
    After you unzip the file, there are two folders under data_folder, i.e. 'raw' and 'my_cached'.   'raw' folder contains original txt files from the COVID MAG dataset. 'my_cached' folder contains processed data files, if you miss some steps during the hands-on tutorial, you can make it up by copying corresponding files into experiment folders.
1. Install the dependencies
    1. The model pre-training will use a tool for converting the original data into embeddings. Use of the tool will require `g++`. The following installs `g++` on a Linux system.
        ```bash
        sudo apt-get install g++
        ```
    1. The Python script will be run in a conda environment where the dependencies are installed. This can be done by using the `reco_gpu_kdd.yaml` file provided in the branch subfolder with the following commands.
        ```bash
        conda env create -n kdd_tutorial_2020 -f reco_gpu_kdd.yaml
        conda activate kdd_tutorial_2020
        ```
1. The tutorial will be conducated by using the Jupyter notebooks. The newly created conda kernel can be registered with the Jupyter notebook server
    ```bash
    python -m ipykernel install --user --name kdd_tutorial_2020 --display-name "Python (kdd tutorial)"
    ```

# Tutorial notebooks/scripts
After the setup, the users should be able to launch the notebooks locally with the command 
```bash
jupyter notebook --port=8080
```
Then the notebook can be spinned off in a browser at the address of `localhost:8080`.
Alternatively, if the jupyter notebook server is on a remote server, the users can launch the jupyter notebook by using the following command.
```bash
jupyter notebook --no-browser --ip=10.214.70.89 --port=8080
```
From the local browser, the notebook can be spinned off at the address of `10.214.70.89:8080`.
