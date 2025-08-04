# LangExtract

## Introduction

LangExtract is an extension of Google's LangExtract package that provides a powerful and unified interface for extracting structured information from unstructured text using Large Language Models (LLMs). Built with enterprise-grade scalability and reliability in mind, it seamlessly integrates with all major LLM providers including OpenAI, Anthropic, Google, and local models via Ollama.

The library enables developers to transform raw text into structured data through natural language instructions and example-driven guidance, making it ideal for information extraction, entity recognition, relationship mapping, and content analysis tasks across various domains.

## Features and Capabilities

- **Multi-Provider Support**: Works with OpenAI GPT models, Anthropic Claude, Google Gemini, and local Ollama models
- **Structured Output**: Built-in schema constraints and JSON/YAML formatting
- **Parallel Processing**: Concurrent API calls with configurable worker pools for high-throughput processing
- **Multi-Pass Extraction**: Sequential extraction passes to improve recall and find additional entities
- **Flexible Input**: Process strings, documents, or URLs with automatic content downloading
- **Rich Visualization**: Interactive HTML visualizations of extraction results
- **Production Ready**: Environment variable management, error handling, and comprehensive testing
- **Example-Driven**: Uses high-quality examples to guide extraction quality and consistency

## Installation


```bash
git clone git@gitlab.com:accreteai-main/ai-research/relation-extraction-research/langextract.git
cd langextract
pip install -e .
```

Or using uv (recommended for faster dependency resolution):

```bash
uv init && uv sync
```

### Dependencies

LangExtract requires Python 3.8+ and installs the following key dependencies:

- `google-genai` - Google Gemini API client
- `anthropic` - Anthropic Claude API client  
- `openai` - OpenAI GPT API client
- `requests` - HTTP client for Ollama and URL downloads
- `python-dotenv` - Environment variable management
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

**Available Models:**
- `claude-3-5-haiku-latest` - Fast, cost-effective
- `claude-3-5-sonnet-latest` - Balanced performance
- `claude-3-opus-latest` - Highest capability

### OpenAILanguageModel
```python
from langextract.inference import OpenAILanguageModel

model = OpenAILanguageModel(
    model_id='gpt-4o-mini',  # or gpt-4o, gpt-4-turbo
    api_key='your-api-key',  # or set OPENAI_API_KEY
    organization='your-org-id',  # Optional
    temperature=0.0,
    max_workers=10,
    structured_schema=None,  # Optional schema for structured output
    format_type=data.FormatType.JSON
)
```

**Available Models:**
- `gpt-4o-mini` - Cost-effective, fast
- `gpt-4o` - Latest flagship model
- `gpt-4-turbo` - High performance
- `gpt-3.5-turbo` - Legacy, budget option

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

**Available Models:**
- `gemini-2.5-flash` - Latest, fastest
- `gemini-1.5-pro` - High capability
- `gemini-1.5-flash` - Balanced performance

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

**Popular Ollama Models:**
- `gemma2:latest` - Google's Gemma 2
- `llama3.1:latest` - Meta's Llama 3.1
- `mistral:latest` - Mistral AI
- `codellama:latest` - Code-specialized Llama

## Quick Start Example

Here's a complete example based on the included `example.py`:

```bash
uv run example.py
```

## Environment Variables and Configuration

Set up API keys using environment variables:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-claude-api-key"

# For OpenAI GPT models  
export OPENAI_API_KEY="your-openai-api-key"

# For Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

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
- Consider local Ollama models for cost-sensitive applications

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Support

- **Documentation**: [Link to docs when available]
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Examples**: Check the `examples/` directory for more use cases