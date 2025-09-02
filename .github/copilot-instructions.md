# pybioclip Development Instructions

pybioclip is a Python package and CLI tool that simplifies using BioCLIP for taxonomic image classification, custom label prediction, and embedding generation. This tool requires network connectivity to HuggingFace Hub for downloading pre-trained models.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup and Installation
- Ensure Python 3.10+ is available (compatible with PyTorch)
- `python -m pip install --upgrade pip` -- upgrade pip first
- `pip install .` -- install package in development mode (5-15 minutes with network). NEVER CANCEL. Set timeout to 20+ minutes.
- Alternative installation: `pip install pybioclip` -- for stable release

### Testing
- `python -m unittest` -- run all tests (3-5 seconds for offline tests, but many require network connectivity)
- `python -m unittest tests.test_main -v` -- run CLI argument parsing tests (always works offline, 3 seconds)
- `python -m unittest tests.test_recorder -v` -- run logging/recording tests (works offline, 2 seconds)
- NEVER CANCEL network-dependent tests: model download tests require HuggingFace access and will timeout in restricted environments
- Tests that download models from HuggingFace Hub will fail without internet connectivity

### Command Line Interface Validation
- `bioclip --help` -- verify CLI is working
- `bioclip predict --help` -- verify predict command help
- `bioclip embed --help` -- verify embed command help
- `bioclip list-models` -- list available models (works offline, cached model list)
- CLI parsing works offline but actual predictions require model downloads

### Documentation
- `pip install mkdocs mkdocs-material mkdocs-material-extensions mkdocstrings-python` -- install docs dependencies (may timeout without network)
- `mkdocs serve` -- serve documentation locally (requires docs dependencies)
- `mkdocs build` -- build static documentation
- Documentation source files are in `/docs/` directory

## Validation

### Network Connectivity Requirements
- **CRITICAL**: First-time model usage downloads from HuggingFace Hub (2-5 GB models). NEVER CANCEL. Set timeout to 60+ minutes.
- Models are cached in `~/.cache/huggingface/` after first download
- Without network access, only CLI parsing tests and documentation generation work
- Container alternatives (Docker/Apptainer) can be used if network restrictions prevent local setup

### Manual Testing Scenarios  
When making changes to pybioclip, ALWAYS validate:
1. **CLI Interface**: Run `bioclip --help` and ensure all commands are listed correctly
2. **Argument Parsing**: Test command parsing with `bioclip predict --help` and verify options
3. **Version Check**: Run `bioclip --version` to confirm installation
4. **Model List**: Run `bioclip list-models` to verify model access (works offline with cached model list)
5. **Unit Tests**: Run `python -m unittest tests.test_main tests.test_recorder -v` (22 tests, ~3 seconds, works offline)
6. **Installation**: Ensure `pip install .` completes without errors (with network access)
7. **Full Prediction**: Test with sample image requires network for model download on first use

### Container Testing (Network Alternative)
If unable to install locally due to network restrictions:
```bash
# Download example image first
wget https://huggingface.co/spaces/imageomics/bioclip-demo/resolve/main/examples/Ursus-arctos.jpeg

# Test with Docker (requires Docker installed)
docker run --rm \
  -v $(pwd):/home/bcuser \
  ghcr.io/imageomics/pybioclip:latest \
  bioclip predict Ursus-arctos.jpeg
```

### CI/CD Validation
- All tests pass in GitHub Actions CI (with network access)
- The CI runs across Python 3.10, 3.11, 3.12, 3.13
- No explicit linting tools configured - code style follows existing patterns
- NEVER add linting tools unless specifically required

## Common Tasks

### Repository Structure
```
.
├── .github/workflows/    # CI/CD workflows
├── docs/                 # MkDocs documentation
├── examples/             # Jupyter notebook examples  
├── src/bioclip/          # Main package source code
│   ├── __main__.py       # CLI entry point
│   ├── predict.py        # Prediction classes
│   └── recorder.py       # Logging functionality
├── tests/                # Unit tests
│   ├── test_main.py      # CLI tests (work offline)
│   ├── test_predict.py   # Prediction tests (require network)
│   └── test_recorder.py  # Logging tests (work offline)
├── pyproject.toml        # Build configuration with hatch
├── mkdocs.yml           # Documentation configuration
└── README.md            # Main documentation
```

### Key Files to Modify
- `src/bioclip/__main__.py` -- CLI argument parsing and command routing
- `src/bioclip/predict.py` -- Prediction logic and model handling
- `src/bioclip/recorder.py` -- Prediction logging and recording
- `tests/test_*.py` -- Add tests for new functionality
- `docs/*.md` -- Update documentation for user-facing changes

### Build System
- Uses `hatch` build backend (specified in pyproject.toml)
- No Makefile or shell scripts for building
- Package metadata in `pyproject.toml`
- Version defined in `src/bioclip/__about__.py`

### Dependencies
Core dependencies (automatically installed):
- `torch` and `torchvision` -- PyTorch for ML operations
- `open_clip_torch` -- OpenCLIP for model loading
- `prettytable` -- CLI table formatting
- `pandas` -- Data handling

### Expected Timing
- CLI tests: 3 seconds
- Package installation: 5-15 minutes (with network for dependencies)
- Model download on first use: 10-60 minutes (2-5 GB download)
- Full test suite: 1-5 minutes (with cached models)

### Common Scenarios
1. **Adding new CLI options**: Modify `src/bioclip/__main__.py` and add tests in `tests/test_main.py`
2. **Changing prediction logic**: Modify `src/bioclip/predict.py` and add tests in `tests/test_predict.py`
3. **Documentation updates**: Edit files in `docs/` directory and test with `mkdocs serve`
4. **Container testing**: Use provided Docker/Apptainer containers for validation

## Troubleshooting

### Network Issues
- Tests requiring HuggingFace access will fail: this is expected in restricted environments
- Use offline tests (`tests.test_main`) to validate CLI logic changes
- Use containers for end-to-end testing when network restricted

### Installation Problems
- Always upgrade pip first: `python -m pip install --upgrade pip`
- Large dependency downloads may timeout: increase pip timeout or use containers
- PyTorch installation varies by platform: see PyTorch website for platform-specific instructions

### Container Usage
- Docker: `ghcr.io/imageomics/pybioclip:latest`
- Apptainer: `oras://ghcr.io/imageomics/pybioclip-sif:latest`
- Both containers include all dependencies and cached models