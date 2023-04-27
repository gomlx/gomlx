## JupyterLab Notebook Docker

### Running it
Simplest way to run it, go to a directory you want to be made visible for the Jupyter Notebook -- where you
will store your Go notebook files -- and run:

```bash
$ docker run -it --rm -p 8888:8888 -v "${PWD}":/home/gomlx/work janpfeifer/gomlx_jupyterlab:latest
```

It will output the `localhost:8888` link you can use -- with the secret token to connect to Jupyter.

It includes [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)(a Jupyter Notebook web UI),
[GoNB](https://github.com/janpfeifer/gonb)(a Go kernel for Jupyter notebooks), **GoMLX** and the CUDA
drivers to support GPU -- but it also works if your machine doesn't have it -- at the cost of being a
larger docker (TODO: split on 2 dockers in the future)

### Building the Docker

Unfortunately due to a couple of issues with NVidia drivers and Docker runtime selection during build, we need some hacking around:

1. I have no idea why, but using CUDA/CuDNN with XLA the very first time is very slow. It takes a couple of minutes to run. So we run a trivial model once upfront, so the docker is ready to run fast.
1. The problem is that `docker build` more recently is not run with access to GPU (it doesn't use Nvidia runtime by default). See issue here:
  * https://forums.developer.nvidia.com/t/nvidia-driver-is-not-available-on-latest-docker/246265/2
  * https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime/61737404#61737404

So what we do is, after the image is built, immediately run a test train and commit the changed container to the image. We have to then also add a change to set update `CMD`, otherwise it will rewrite the `CMD` with our test script.

```bash
$ docker build -t janpfeifer/gomlx_jupyterlab:latest -f docker/jupyterlab/Dockerfile . \
  && docker run --gpus all -it gomlx_jupyterlab:latest bash -c 'cd gomlx/examples/linear ; go run . --platform=CUDA' \
  && docker commit --message "Warm up CUDA driver." \
      --change='CMD ["jupyter-lab", "--no-browser", "--ip=0.0.0.0"]' \
      $(docker container ls --latest --quiet) janpfeifer/gomlx_jupyterlab:latest
```

### TODOs

- Create a version without CUDA, to save space for those not using it.
- Create an organization named `gomlx` in Docker Hub (it costs $9 per month as of 4/2023), and use that to store the docker.

