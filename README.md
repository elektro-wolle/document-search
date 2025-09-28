# PDF Vector Search System

A lightweight command‑line tool that ingests PDF documents, generates embeddings with **Ollama**, stores them in **ChromaDB**, and lets you search the corpus with include/exclude filters.

> **Author**: Wolfgang Jung  
> **Python**: 3.11+ (tested with 3.11)  

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
    - [Install Python Dependencies](#install-python-dependencies)
    - [Install Ollama](#install-ollama)
    - [Set Up the Project](#set-up-the-project)
- [Usage](#usage)
    - [Ingest PDFs](#ingest-pdfs)
    - [Search](#search)
    - [Interactive Mode](#interactive-mode)
    - [Database Info](#database-info)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Prerequisites

| Item                   | Version | Why                                                           |
|------------------------|---------|---------------------------------------------------------------|
| **Python**             | 3.11+   | Script is written in Python 3.                                |
| **pip**                | Latest  | Package manager for dependencies.                             |
| **Ollama**             | Latest  | Provides the embedding model.                                 |

> **Tip**: If you already have Anaconda or Miniconda, you can create a fresh environment:

```shell script
conda create -n pdfsearch python=3.11
conda activate pdfsearch
```

Otherwise, the installation of [uv](https://docs.astral.sh/uv/getting-started/installation/) is recommended. 

---

## Installation

### Install Python Dependencies

#### Using `conda`:

```shell script
pip install PyPDF2 chromadb pymupdf numpy requests
```

#### Using uv


```bash
# Create a uv-managed virtual environment (creates `.uv` directory)
uv venv .uv

# Activate the environment (Unix/macOS)
source .uv/bin/activate

# (Windows) .uv\Scripts\activate

# Install all dependencies from pyproject.toml
uv sync
```

### Install Ollama

**Ollama** is the local LLM serving engine used to generate embeddings.

- **Linux / macOS**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
- 
- **Windows**

    1. Download the latest installer from the official page: https://ollama.com/download
    2. Run the installer and follow the wizard.

- **Verify installation**

```bash
ollama --version
```
You should see something like:
```

  Ollama version 0.1.4
```
- **Start Ollama**

```bash
ollama serve
```

- **Pull the embedding model**

```bash
ollama pull nomic-embed-text
```

> **Note**: The script defaults to `http://localhost:11434`. If you use a custom port or host, pass `--ollama-url`.

### Set Up the Project

Clone or download the repository, then:

```bash
git clone https://github.com/your-username/pdf-vector-search.git
cd pdf-vector-search
```

Ensure the `text-search.py` script (the main executable) is present.

---

## Usage

The tool is invoked via the `text-search.py` script. It accepts several sub‑commands:

```bash
python text-search.py <command> [options]
```

### Ingest PDFs

```bash
python text-search.py ingest --directory /path/to/pdfs
```


- Scans the directory for all `*.pdf` files.
- Extracts text, splits into chunks, generates embeddings, and stores them in ChromaDB.
- Progress is printed to the console.

### Search

```bash
python text-search.py search \
  --include "machine learning" "algorithms" \
  --exclude "java" "neural networks" \
  --threshold 0.3 \
  --top-k 10
```

- `--include`: terms that **must** match (AND logic).
- `--exclude`: terms that **must not** match (OR logic).
- `--threshold`: minimum similarity to consider (default `0.3`).
- `--top-k`: number of results to return (default `10`).

The output lists rank, file, page, similarity scores, and a preview of the matched text.

### Interactive Mode

```bash
python text-search.py interactive
```

A REPL is started:

```
PDF Vector Search System - Interactive Mode
Commands:
```


- `--include`: terms that **must** match (AND logic).
- `--exclude`: terms that **must not** match (OR logic).
- `--threshold`: minimum similarity to consider (default `0.3`).
- `--top-k`: number of results to return (default `10`).

The output lists rank, file, page, similarity scores, and a preview of the matched text.

### Interactive Mode

```shell script
python text-search.py interactive
```


A REPL is started:

```
PDF Vector Search System - Interactive Mode
Commands:
  ingest <directory>
  search --include term1,term2 --exclude term3,term4
  search term1,term2
  info
  quit
>
```


### Database Info

```shell script
python text-search.py info
```


Displays the collection name, path, and document count.

---

## Examples

| Scenario                                        | Command                                                                   |
|-------------------------------------------------|---------------------------------------------------------------------------|
| **Ingest all PDFs in `./docs`**                 | `python text-search.py ingest --directory ./docs`                         |
| **Search for "deep learning" excluding "java"** | `python text-search.py search --include "deep learning" --exclude "java"` |
| **Run in interactive mode**                     | `python text-search.py interactive`                                       |
| **Check database status**                       | `python text-search.py info`                                              |

---

## Troubleshooting

| Problem                  | Symptom                               | Fix                                                |
|--------------------------|---------------------------------------|----------------------------------------------------|
| Ollama not responding    | Timeout or connection refused         | Ensure `ollama serve` is running.                  |
| Embedding request fails  | `requests.exceptions.ConnectionError` | Verify `--ollama-url` matches the Ollama endpoint. |
| PDF chunks missing       | No output after ingestion             | PDFs may be scanned or encrypted. Try another PDF. |
| Too many similar results | Overwhelming output                   | Increase `--top-k` or adjust `--threshold`.        |

---

## Contributing

Feel free to open issues or pull requests. Follow the standard GitHub workflow:

1. Fork the repo
2. Create a feature branch
3. Commit changes with clear messages
4. Submit a pull request

---

## License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

Happy searching!
