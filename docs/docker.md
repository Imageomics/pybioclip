# Docker Container

A docker container for pybioclip is hosted at [ghcr.io](https://github.com/Imageomics/pybioclip/pkgs/container/pybioclip).
This container has CPU support for Mac, Windows, and Linux. GPU(CUDA) support is only for Windows and Linux.

### Volumes
In order to access files the docker container requires you to [mount a volume](https://docs.docker.com/engine/storage/volumes).
The working and home directories in this container are both set to `/home/bcuser`.
Minimally you need to mount a volume into this directory so pybioclip can read your images.
When running pybioclip the software will download various BioCLIP files into a `/home/bcuser/.cache` subdirectory. If you want to store the `.cache` folder in your home directory you will need to mount that directory (`~/.cache`) into the container at `/home/bcuser/.cache`.

### Setup
The examples below require an image file named [Ursus-arctos.jpeg](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/ef075807a55687b320427196ac1662b9383f988f/examples/Ursus-arctos.jpeg) in the current directory.

## Mac/Linux CPU Usage
The following command will create predictions for the `Ursus-arctos.jpeg` image in the current directory.
The command mounts the current directory into the container at `/home/bcuser`.
The command mounts the `~/.cache` directory into the container to cache BioCLIP files in your home directory.
```console
docker run --platform linux/amd64 \
           -v $(pwd):/home/bcuser \
           -v ~/.cache:/home/bcuser/.cache \
           --rm ghcr.io/imageomics/pybioclip:1.1.0 \
           bioclip predict Ursus-arctos.jpeg
```

## Linux GPU Usage
The following command will create predictions using a GPU for the `Ursus-arctos.jpeg` image in the current directory.
```console
docker run --gpus all \
           --platform linux/amd64 \
           -v $(pwd):/home/bcuser \
           -v ~/.cache:/home/bcuser/.cache \
           --rm ghcr.io/imageomics/pybioclip:1.1.0 \
           bioclip predict --device cuda Ursus-arctos.jpeg
```

## Windows CPU Usage
The following command will create predictions for the `Ursus-arctos.jpeg` image in the current directory.
Since this command does not mount `/home/bcuser/.cache` in the container the `.cache` directory will be created within the current directory.
```console
docker run --rm -v %cd%:/home/bcuser ghcr.io/imageomics/pybioclip:1.0.0 bioclip predict Ursus-arctos.jpeg
```

## Windows GPU Usage
The following command will create predictions using a GPU for the `Ursus-arctos.jpeg` image in the current directory.
```console
docker run --rm --gpus all -v %cd%:/home/bcuser ghcr.io/imageomics/pybioclip bioclip:1.0.0 predict --device cuda Ursus-arctos.jpeg
```
