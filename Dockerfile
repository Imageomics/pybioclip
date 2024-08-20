FROM condaforge/miniforge3:24.3.0-0

ARG PYBIOCLIP_VERSION

# Install pybioclip from the release archive because pypi can take a while to start working.
RUN pip install "https://github.com/Imageomics/pybioclip/archive/refs/tags/${PYBIOCLIP_VERSION}.tar.gz" && \
    pip cache purge
