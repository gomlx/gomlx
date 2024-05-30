## JupyterLab Notebook Docker

### Running it
Simplest way to run it, go to **a directory you want to be made visible for the Jupyter Notebook** -- where you
will store your Go notebook files -- and run:

```bash
docker pull janpfeifer/gomlx_jupyterlab:latest
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jupyter/work janpfeifer/gomlx_jupyterlab:latest
```

If you have GPU(s) and want to make them accessible, use instead:

```bash
docker run -it --gpus all --rm -p 8888:8888 -v "${PWD}":/home/jupyter/work janpfeifer/gomlx_jupyterlab:latest
```

The command will output the `localhost:8888` link you can use -- it will include the secret token to connect to Jupyter.

Once you have the JupyterLab opened, try opening the tutorial under `Projects/gomlx/examples/tutorial/tutorial.ipynb`.

It includes [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)(a Jupyter Notebook web UI),
[GoNB](https://github.com/janpfeifer/gonb)(a Go kernel for Jupyter notebooks), **GoMLX** and the CUDA
drivers to support GPU if they are available (at the cost of being a larger docker -- in the future we 
should split on 2 dockers)

### Building the Docker

You need a copy of the binary pre-build GoMLX C library.
You can download it from the latest release (in 
`https://github.com/gomlx/gomlx/releases/download/${GOMLX_VERSION}/gomlx_xla-linux-amd64.tar.gz`)
or if you build it yourself (see `docs/building.md`), the built file is usually in ``./c/bazel-bin/gomlx_xla.tar.gz`.
Copy or download the file and rename to `gomlx_xla.tar.gz`, in the repository root directory -- or 
set the path to the file by adding `--build-arg GOMLX_XLA_TAR_GZ=<path_to_file>` to the Docker build
command below.

Due to a couple of issues with NVidia drivers and Docker runtime selection during build, we need a bit of trickery:

1. I have no idea why, but using CUDA/CuDNN with XLA the very first time is very slow.  
   It takes a couple of minutes to run.
   So we run a trivial model once upfront, so the docker is ready to run faster.
1. Unfortunately `docker build` more recently cannot run with access to GPUs (it doesn't use Nvidia runtime by default).
   See issue here:
  * https://forums.developer.nvidia.com/t/nvidia-driver-is-not-available-on-latest-docker/246265/2
  * https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime/61737404#61737404

So what we do is, after the image is built, immediately run a test train and commit the changed container to the 
image. We have to then also add a change to set update `CMD`, otherwise it will rewrite the `CMD` with our test script.


```bash
docker build -t janpfeifer/gomlx_jupyterlab:latest -f docker/jupyterlab/Dockerfile . \
  && docker run --gpus all -it janpfeifer/gomlx_jupyterlab:latest bash -c 'cd Projects/gomlx/examples/linear ; go run . --platform=CUDA' \
  && docker commit --message "Warm up CUDA driver." \
      --change='CMD ["jupyter-lab", "--no-browser", "--ip=0.0.0.0"]' \
      $(docker container ls --latest --quiet) janpfeifer/gomlx_jupyterlab:latest
```

Finally, to push the image, set the variable `version` to the desired version and do:

```bash
docker tag janpfeifer/gomlx_jupyterlab:latest janpfeifer/gomlx_jupyterlab:${version} \
  && docker push janpfeifer/gomlx_jupyterlab:latest \
  && docker push janpfeifer/gomlx_jupyterlab:${version}
```

### TODOs

- Create a version without CUDA, to save space for those not using it.
- Create an organization named `gomlx` in [Docker Hub](https://hub.docker.com/) (it costs $9 per month as of 4/2023), 
  and use that to store the docker -- it shouldn't be located on a personal namespace.
- Figure a way to have JupyterLab to open the tutorial automatically when it is first opened.

