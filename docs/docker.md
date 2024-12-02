# Docker Container

A docker container for pybioclip is hosted at [ghcr.io](https://github.com/Imageomics/pybioclip/pkgs/container/pybioclip).
This container has CPU support for Mac, Windows, and Linux. GPU(CUDA) support is only for Windows and Linux.
In order to access your images the docker container requires you to [mount a volume](https://docs.docker.com/engine/storage/volumes). To avoid redownloading various bioclip files mounting a volume to hold a `.cache` directory is recommended. By default the working and home directories for this container are both set to `/home/bcuser`.

All examples below require an image file named [Ursus-arctos.jpeg](https://huggingface.co/spaces/imageomics/bioclip-demo/blob/main/examples/Ursus-arctos.jpeg) in the current directory.

## Mac/Linux CPU Usage
```console
docker run --platform linux/amd64 -v $(pwd):/home/bcuser -it ghcr.io/imageomics/pybioclip:1.0.0 bioclip predict Ursus-arctos.jpeg
```

## Linux GPU Usage
```console
docker run --gpus all --platform linux/amd64 -v $(pwd):/home/bcuser -it ghcr.io/imageomics/pybioclip:1.0.0 bioclip predict --device cuda Ursus-arctos.jpeg
```

## Windows CPU Usage

```console
docker run -it -v %cd%:/home/bcuser ghcr.io/imageomics/pybioclip:1.0.0 bioclip predict Ursus-arctos.jpeg
```

## Windows GPU Usage

```console
docker run -it --gpus all -v %cd%:/home/bcuser ghcr.io/imageomics/pybioclip bioclip:1.0.0 predict --device cuda Ursus-arctos.jpeg
```
