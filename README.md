# LangExtract

## Introduction

LangExtract is an extension of Google's LangExtract package that provides a powerful and unified interface for extracting structured information from unstructured text using Large Language Models (LLMs). Built with enterprise-grade scalability and reliability in mind, it seamlessly integrates with all major LLM providers including OpenAI, Anthropic, Google, or local models via Ollama. Open weight models like OpenAI's gpt-oss-120b and gpt-oss-20b via HuggingFace's OpenAI-compatible API is also supported.

The library enables developers to transform raw text into structured data through natural language instructions and example-driven guidance, making it ideal for information extraction, entity recognition, relationship mapping, and content analysis tasks across various domains.

## Key Takeaways

- **Multi-Provider Support**: Works with OpenAI GPT models, Anthropic Claude, Google Gemini, HuggingFace OpenAI-compatible API, and local Ollama models. Compatible with latest models like GPT-5, Claude-4 and more.
- **Example-Driven Few-Shot Learning**: LangExtract minimizes the need for extensive data labeling and model fine-tuning, making it accessible to users with varying technical expertise. Uses high-quality examples to guide extraction quality and consistency
- **Long-Context Processing**: The tool efficiently handles large datasets while maintaining contextual accuracy, making it ideal for complex NLP tasks.
- **Parallel Processing**: Concurrent API calls with configurable worker pools for high-throughput processing.
- **Multi-Pass Extraction**: Sequential extraction passes to improve recall and find additional entities
- **Flexible Input**: Process strings, documents, or URLs with automatic content downloading
- **Rich Visualization**: Interactive HTML visualizations of extraction results
- **Production Ready**: Environment variable management, error handling, and comprehensive testing

## Installation

```bash
git clone <this repo url>
cd langextract
pip install -e .
```

Or using uv (recommended for development):

```bash
uv init && uv sync
```

### Dependencies

LangExtract requires Python 3.8+ and installs the following key dependencies:

- `google-genai` - Google Gemini API client
- `anthropic` - Anthropic Claude API client  
- `openai` - OpenAI GPT API client
- `huggingface-hub` - HuggingFace API client
- `requests` - HTTP client for Ollama and URL downloads
- `pydantic` - Data validation and serialization

## Language Model Classes

LangExtract supports four different language model backends:

### ClaudeLanguageModel (Anthropic)
```python
from langextract.inference import ClaudeLanguageModel

model = ClaudeLanguageModel(
    model_id='claude-3-5-haiku-latest',  # or claude-3-5-sonnet-latest
    api_key='your-api-key',  # or set ANTHROPIC_API_KEY
    temperature=0.0,
    max_workers=10,
    structured_schema=None,  # Optional schema for structured output
    format_type=data.FormatType.JSON
)
```


### OpenAILanguageModel
```python
from langextract.inference import OpenAILanguageModel

model = OpenAILanguageModel(
    model_id='gpt-5-nano',  # or gpt-4o
    api_key='your-api-key',  # or set OPENAI_API_KEY
    organization='your-org-id',  # Optional
    temperature=0.0,
    max_workers=10,
    structured_schema=None,  # Optional schema for structured output
    format_type=data.FormatType.JSON
)
```


### GeminiLanguageModel (Google)
```python
from langextract.inference import GeminiLanguageModel

model = GeminiLanguageModel(
    model_id='gemini-2.5-flash',  # or gemini-1.5-pro
    api_key='your-api-key',  # or set GOOGLE_API_KEY
    temperature=0.0,
    max_workers=10,
    structured_schema=None,  # Optional schema for structured output
    format_type=data.FormatType.JSON
)
```


### HFLanguageModel (HuggingFace)
```python
from langextract.inference import HFLanguageModel

model = HFLanguageModel(
    model_id='openai/gpt-oss-120b:cerebras',  # or other HF models
    api_key='your-hf-token',  # or set HF_TOKEN environment variable
    base_url='https://router.huggingface.co/v1',  # Default HF router
    temperature=0.0,
    max_workers=10,
    structured_schema=None,  # Optional schema for structured output
    format_type=data.FormatType.JSON
)
```

**Popular HuggingFace Models:**
- `openai/gpt-oss-120b:cerebras` - High-performance open model
- `meta-llama/llama-3.2-3b-instruct` - Meta's Llama 3.2
- `microsoft/DialoGPT-medium` - Microsoft's conversational model
- `mistralai/mixtral-8x7b-instruct` - Mistral's mixture of experts

### OllamaLanguageModel (Local/Self-Hosted)
```python
from langextract.inference import OllamaLanguageModel

model = OllamaLanguageModel(
    model='gemma2:latest',  # or llama3.1, mistral, etc.
    model_url='http://localhost:11434',  # Default Ollama endpoint
    structured_output_format='json',
    temperature=0.8,
    constraint=None  # Optional schema constraint
)
```


## Quick Start Example

Here's a complete example based on the included `example.py`:

```bash
uv run example.py
```

### Use cases from `example.py`

The included script demonstrates end‑to‑end extraction driven by a natural‑language prompt and a high‑quality example, across multiple model providers. It:

- Defines a prompt and a guiding example with `lx.data.ExampleData` and `lx.data.Extraction`.
- Selects a provider and model (`openai`, `google`, `anthropic`, or `hf`) and maps it to the matching `inference.*LanguageModel`.
- Runs `lx.extract(...)` with tuned parameters (`extraction_passes=1`, `max_workers=10`, `max_char_buffer=1000`).
- Saves results to `output/{provider}_extraction_results.jsonl` and renders an HTML visualization at `output/{provider}_extraction_results_visualization.html`.

Run the example for different providers (ensure the corresponding API key env vars are set as shown below):

```bash
# OpenAI (default in __main__)
uv run example.py

# Google Gemini
uv run python -c "import example; example.main('google')"

# Anthropic Claude
uv run python -c "import example; example.main('anthropic')"

# HuggingFace OpenAI-compatible router (open-weight models)
uv run python -c "import example; example.main('hf')"
```

Expected outputs:

- JSONL: `output/{provider}_extraction_results.jsonl`
- HTML visualization: `output/{provider}_extraction_results_visualization.html`

Tip: The prompt and the in‑code example in `example.py` show how to nudge models toward high‑quality, consistent entity and relationship extraction using exact text spans.

## Environment Variables and Configuration

Set up API keys using environment variables:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-claude-api-key"

# For OpenAI GPT models  
export OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# For HuggingFace models
export HF_TOKEN="your-huggingface-token"

# Ollama doesn't require an API key (local models)
```

Or create a `.env` file in your project root:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Advanced Usage

### Multi-Pass Extraction
Improve recall by running multiple extraction passes:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    extraction_passes=3,  # Run 3 independent passes
    model_id='claude-3-5-haiku-latest',
    language_model_type=inference.ClaudeLanguageModel
)
```

### Parallel Processing
Configure parallel workers for high-throughput processing:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    max_workers=20,        # Parallel API calls
    batch_length=50,       # Chunks per batch
    model_id='gpt-4o-mini',
    language_model_type=inference.OpenAILanguageModel
)
```

### Using HuggingFace Models
Access cutting-edge open-source models via HuggingFace's router:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id='openai/gpt-oss-120b:cerebras',
    language_model_type=inference.HFLanguageModel,
    language_model_params={
        'api_key': 'your-hf-token',  # or set HF_TOKEN env var
        'temperature': 0.0
    }
)
```

### Custom Schema Constraints
Use structured schemas for consistent output format:

```python
from langextract import schema

# Define custom schema
custom_schema = schema.StructuredSchema.from_examples(examples)

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    use_schema_constraints=True,  # Enable structured output
    language_model_params={'structured_schema': custom_schema}
)
```

### Processing Documents and URLs
Process various input types:

```python
# Process URL
result = lx.extract(
    text_or_documents="https://example.com/article",
    prompt_description=prompt,
    examples=examples
)

# Process multiple documents
documents = [
    lx.data.Document(text="Document 1 content", metadata={"source": "doc1"}),
    lx.data.Document(text="Document 2 content", metadata={"source": "doc2"})
]

results = lx.extract(
    text_or_documents=documents,
    prompt_description=prompt,
    examples=examples
)
```

## API Reference

### Core Functions
- `lx.extract()` - Main extraction function
- `lx.visualize()` - Generate HTML visualizations
- `lx.io.save_annotated_documents()` - Save results to JSONL

### Key Classes
- `lx.data.ExampleData` - Training examples for extraction
- `lx.data.Extraction` - Individual extraction results
- `lx.data.AnnotatedDocument` - Document with extractions
- `lx.data.Document` - Input document structure

### Language Models
- `inference.ClaudeLanguageModel` - Anthropic Claude
- `inference.OpenAILanguageModel` - OpenAI GPT
- `inference.GeminiLanguageModel` - Google Gemini
- `inference.HFLanguageModel` - HuggingFace models via OpenAI-compatible API
- `inference.OllamaLanguageModel` - Local Ollama models

## Cost Considerations

API costs vary by provider and model. Key factors affecting cost:

- **Token Volume**: Larger `max_char_buffer` values reduce API calls but process more tokens per call
- **Extraction Passes**: Each additional pass reprocesses tokens (3 passes = 3x token cost)
- **Parallel Workers**: `max_workers` improves speed without additional token costs
- **Model Selection**: Larger models (GPT-4o, Claude-3 Opus) cost more than smaller ones (GPT-4o-mini, Claude-3 Haiku)

**Cost Optimization Tips:**
- Start with smaller models for testing (gpt-4o-mini, claude-3-5-haiku-latest)  
- Use `extraction_passes=1` initially, increase only if recall is insufficient
- Monitor usage with small test runs before processing large datasets
- Consider models hosted on HuggingFace for access to open-source alternatives
- Consider local Ollama models for cost-sensitive applications


## Roadmap
- [ ] Add support fo Azure OpenAI
- [ ] Improve example-driven extraction with more complex schemas
- [ ] Self improving prompt, e.g. how to incooporate GPT-5 Prompt Optimizer  https://cookbook.openai.com/examples/gpt-5/prompt-optimization-cookbook
- [ ] fix the extraction instability in some LLMs. 

## Contributing

## License

## Support
