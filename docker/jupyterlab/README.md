## JupyterLab Notebook Docker

### Running it
Simplest way to run it, go to **a directory you want to be made visible for the Jupyter Notebook** -- where you
will store your Go notebook files -- and run:

```bash
docker pull janpfeifer/gomlx_jupyterlab:latest
docker run -it --rm -p 8888:8888 -v "${PWD}":/home/jupyter/host janpfeifer/gomlx_jupyterlab:latest
```

If you have GPU(s) and want to make them accessible, use instead:

```bash
docker run -it --gpus all --rm -p 8888:8888 -v "${PWD}":/home/jupyter/host janpfeifer/gomlx_jupyterlab:latest
```

The command will output the `localhost:8888` link you can use -- it will include the secret token to connect to Jupyter.

Once you have the JupyterLab opened, try opening the tutorial under `Projects/gomlx/examples/tutorial/tutorial.ipynb`.

It includes [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/)(a Jupyter Notebook web UI),
[GoNB](https://github.com/janpfeifer/gonb)(a Go kernel for Jupyter notebooks), **GoMLX** and the CUDA
drivers to support GPU if they are available (at the cost of being a larger docker -- in the future we 
should split on 2 dockers)

### Building the Docker

The `Dockerfile` has 2 hardcoded versions that needs updating at each release: `GO_VERSION` and `GOPJRT_VERSION`.

Note: the Dockerfile is configured to pull GoMLX (and gopjrt) from GitHub, so it won't use the contents on the current directory. 

```bash
docker build -t janpfeifer/gomlx_jupyterlab:latest -f docker/jupyterlab/Dockerfile . 
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

