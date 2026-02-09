# Environment Setup

> Get your development environment ready in 10 minutes.

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd ai-engineer-fundamentals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

## Required API Keys

### Claude API (Required)
1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account or sign in
3. Navigate to API Keys
4. Create a new key and copy it

```bash
export ANTHROPIC_API_KEY="sk-ant-api..."
```

### OpenAI API (Optional)
Only needed if you want to compare models or use OpenAI embeddings.

```bash
export OPENAI_API_KEY="sk-..."
```

## Environment File

Create a `.env` file in the project root:

```env
# Required
ANTHROPIC_API_KEY=sk-ant-api...

# Optional
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

## Verifying Installation

Run this script to verify everything is set up correctly:

```python
# verify_setup.py
import sys

def check_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name} - NOT INSTALLED")
        return False

def check_api_key(key_name):
    import os
    if os.environ.get(key_name):
        print(f"✓ {key_name} is set")
        return True
    else:
        print(f"✗ {key_name} - NOT SET")
        return False

print("Checking Python version...")
print(f"Python {sys.version}")
print()

print("Checking required packages...")
packages = [
    "anthropic",
    "chromadb",
    "sentence_transformers",
    "torch",
    "transformers",
    "fastapi",
]
all_installed = all(check_import(p) for p in packages)
print()

print("Checking API keys...")
check_api_key("ANTHROPIC_API_KEY")
print()

if all_installed:
    print("✓ Setup complete! You're ready to start.")
else:
    print("✗ Some packages are missing. Run: pip install -r requirements.txt")
```

Run with:
```bash
python verify_setup.py
```

## Hardware Requirements

### Minimum (API-only usage)
- Any modern computer
- Internet connection
- 4GB RAM

### Recommended (Local model work)
- 16GB RAM
- NVIDIA GPU with 8GB+ VRAM (for LoRA fine-tuning)
- 50GB disk space

### For Module 06 (Efficiency) full exercises
- NVIDIA GPU with 16GB+ VRAM
- Or use Google Colab Pro

## Using Google Colab

All notebooks are designed to work in Google Colab:

1. Open notebook in Colab (click badge at top of notebook)
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Add your API key as a Colab secret:
   - Click the key icon in the left sidebar
   - Add `ANTHROPIC_API_KEY` with your key value

## Platform-Specific Notes

### macOS
```bash
# If using Apple Silicon (M1/M2/M3)
# PyTorch will use MPS (Metal Performance Shaders)
pip install torch torchvision torchaudio
```

### Windows
```bash
# Use Windows Subsystem for Linux (WSL2) for best experience
# Or use native Windows with PowerShell
.\venv\Scripts\activate
```

### Linux
```bash
# CUDA setup for NVIDIA GPUs
# Install CUDA toolkit from NVIDIA
# Then install PyTorch with CUDA support:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Docker Setup (Optional)

For a fully reproducible environment:

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
```

```bash
docker build -t ai-engineer-course .
docker run -p 8888:8888 -v $(pwd):/app ai-engineer-course
```

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size
- Use quantization (INT8 or INT4)
- Use smaller model variants
- Clear GPU cache: `torch.cuda.empty_cache()`

### "Module not found"
- Ensure virtual environment is activated
- Reinstall: `pip install -r requirements.txt`

### "API rate limit"
- Add delays between requests
- Use caching for development
- Check your API tier limits

### ChromaDB issues on macOS
```bash
# If you get SQLite errors
pip install pysqlite3-binary
```

## IDE Recommendations

### VS Code
Recommended extensions:
- Python
- Jupyter
- Pylance
- GitLens

### PyCharm
- Use Professional edition for Jupyter support
- Configure virtual environment in Project Settings

## Next Steps

Once setup is complete:
1. Open `quick-starts/00_your_first_llm_call.ipynb`
2. Run the notebook to verify API access
3. Continue with Module 00

---

Need help? Check the course discussions or open an issue.
