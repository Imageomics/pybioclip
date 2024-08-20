FROM condaforge/miniforge3:24.3.0-0

# Pass -s to tini since apptainer doesn't run as PID 1
ENTRYPOINT ["tini", "-s", "--"]

ARG PYBIOCLIP_VERSION

# Install pybioclip from the release archive because pypi can take a while to start working.
RUN pip install "https://github.com/Imageomics/pybioclip/archive/refs/tags/${PYBIOCLIP_VERSION}.tar.gz" && \
    pip cache purge

# Create a non-root user with a home directory to help cache pybioclip files
RUN useradd -ms /bin/bash bcuser
USER bcuser
ENV PATH="$PATH:/home/bcuser/.local/bin"
WORKDIR /home/bcuser
