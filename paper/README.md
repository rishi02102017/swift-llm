# SWIFT-LLM Research Paper

A NeurIPS-style conference paper documenting the SWIFT-LLM framework.

## Files

- `main.tex` - Main paper (6-8 pages)
- `neurips_2024.sty` - NeurIPS 2024 style file
- `references.bib` - Bibliography with 17 citations
- `Makefile` - Build automation

## Compilation Options

### Option 1: Overleaf (Recommended - No Install Required)

1. Go to [Overleaf.com](https://www.overleaf.com)
2. Create new project → Upload Project
3. Upload all files from this folder (`main.tex`, `neurips_2024.sty`, `references.bib`)
4. Click "Recompile" - PDF will be generated automatically

### Option 2: Local LaTeX Installation

**macOS:**
```bash
# Install MacTeX (large, ~4GB)
brew install --cask mactex

# Or install BasicTeX (smaller)
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install collection-fontsrecommended algorithms algorithmicx titlesec natbib
```

**Compile:**
```bash
cd paper
make
# Output: main.pdf
```

### Option 3: Docker

```bash
docker run --rm -v $(pwd):/workdir texlive/texlive pdflatex main.tex
docker run --rm -v $(pwd):/workdir texlive/texlive bibtex main
docker run --rm -v $(pwd):/workdir texlive/texlive pdflatex main.tex
docker run --rm -v $(pwd):/workdir texlive/texlive pdflatex main.tex
```

## Paper Structure

1. **Abstract** - 150 words summarizing key contributions
2. **Introduction** - Problem statement, contributions
3. **Related Work** - LLM optimization, semantic caching, query routing
4. **Methodology** - System architecture, algorithms, math
5. **Experiments** - Setup, metrics, datasets
6. **Results** - Tables and analysis
7. **Discussion** - Findings, limitations, future work
8. **Conclusion** - Summary

## Key Results Highlighted

| Metric | Value |
|--------|-------|
| Cache Hit Latency | 0.5ms |
| Speedup | 3000x |
| Cache Hit Rate | 74.3% |
| Routing Accuracy | 86% |
| Cost Reduction | 99.7% |

## Citations

The paper cites 17 relevant works including:
- GPT-3, GPT-4, LLaMA (foundation models)
- FlashAttention (optimization)
- FAISS (similarity search)
- FrugalGPT (cost optimization)
- GPTCache (semantic caching)

