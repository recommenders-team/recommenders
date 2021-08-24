Docker Support
==============
The Dockerfile in this directory will build Docker images with all the dependencies and code needed to run example notebooks or unit tests included in this repository.

Multiple environments are supported by using [multistage builds](https://docs.docker.com/develop/develop-images/multistage-build/). In order to efficiently build the Docker images in this way, [Docker BuildKit](https://docs.docker.com/develop/develop-images/build_enhancements/) is necessary.
The following examples show how to build and run the Docker image for CPU, PySpark, and GPU environments. 

<i>Note:</i> On some platforms, one needs to manually specify the environment variable for `DOCKER_BUILDKIT`to make sure the build runs well. For example, on a Windows machine, this can be done by the powershell command as below, before building the image
```
$env:DOCKER_BUILDKIT=1
```

<i>Warning:</i> On some platforms using Docker Buildkit interferes with Anaconda environment installation. If you find that the docker build is hanging during Anaconda environment setup stage try building the container without Buildkit enabled.

Once the container is running you can access Jupyter notebooks at http://localhost:8888.

Building and Running with Docker
--------------------------------

<details>
<summary><strong><em>CPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:cpu --build-arg ENV="cpu" .
docker run -p 8888:8888 -d recommenders:cpu
```

</details>

<details>
<summary><strong><em>PySpark environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:pyspark --build-arg ENV="pyspark" .
docker run -p 8888:8888 -d recommenders:pyspark
```

</details>

<details>
<summary><strong><em>GPU environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:gpu --build-arg ENV="gpu" .
docker run --runtime=nvidia -p 8888:8888 -d recommenders:gpu
```

</details>

<details>
<summary><strong><em>GPU + PySpark environment</em></strong></summary>

```
DOCKER_BUILDKIT=1 docker build -t recommenders:full --build-arg ENV="full" .
docker run --runtime=nvidia -p 8888:8888 -d recommenders:full
```

</details>

Build Arguments
---------------

There are several build arguments which can change how the image is built. Similar to the `ENV` build argument these are specified during the docker build command.

Build Arg|Description|
---------|-----------|
ENV|Environment to use, options: cpu, psypark, gpu, full|
ANACONDA|Anaconda installation script (defaults to miniconda3 4.6.14)|

Example:

```
DOCKER_BUILDKIT=1 docker build -t recommenders:cpu --build-arg ENV="cpu" .
```

In order to see detailed progress with BuildKit you can provide a flag during the build command: ```--progress=plain```

Running tests with docker
-------------------------

To run the tests using e.g. the CPU image, do the following: 
```
docker run -it recommenders:cpu bash -c 'pip install pytest; \
pip install pytest-cov; \
apt-get install -y git; \
git clone https://github.com/microsoft/recommenders.git; \
cd recommenders; \
pytest tests/unit -m "not spark and not gpu and not notebooks"'
```