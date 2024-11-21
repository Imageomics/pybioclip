# Apptainer Container
[Apptainer/Singularity](https://apptainer.org/docs/user/main/index.html) images for pybioclip are provided at [ghcr.io/Imageomics/pybioclip-sif registry](https://github.com/Imageomics/pybioclip/pkgs/container/pybioclip-sif).

_NOTE: It is also possible to download the [pybioclip docker container](docker.md) and convert that into a singularity container, but that process can take quite a while._

## Tutorial

### Download example images
Download two images from the [bioclip-demo](https://huggingface.co/spaces/imageomics/bioclip-demo).

```console
wget https://huggingface.co/spaces/imageomics/bioclip-demo/resolve/main/examples/Ursus-arctos.jpeg
wget https://huggingface.co/spaces/imageomics/bioclip-demo/resolve/main/examples/Felis-catus.jpeg
```

### Download a pybioclip container

```console
apptainer pull oras://ghcr.io/imageomics/pybioclip-sif:1.0.0
```
The above command will create a `pybioclip_1.0.0.sif` container image file.

### Create predictions using a CPU
```console
./pybioclip_sif_1.0.0.sif bioclip predict Ursus-arctos.jpeg Felis-catus.jpeg
```

### Create predictions using a GPU
This step requires a cuda GPU.

```console
apptainer exec -nv ./pybioclip_sif_1.0.0.sif bioclip predict --device cuda Ursus-arctos.jpeg Felis-catus.jpeg
```

### Create predictions using a GPU via a Slurm Job
This step requires being on a [Slurm cluster](https://slurm.schedmd.com/documentation.html).

Create a Slurm sbatch script named `bioclip.sh` with the following content:
```
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=1
apptainer exec --nv ./pybioclip_sif_1.0.0.sif bioclip predict --device cuda $*
```
Run the slurm job filling in your Slurm account:
```console
sbatch --account <SLURMACCT> bioclip.sh Ursus-arctos.jpeg Felis-catus.jpeg
```
