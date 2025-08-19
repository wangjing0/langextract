import argparse
import langextract as lx
import textwrap
from langextract import inference

# 1. Define the prompt and extraction rules
prompt_1 = textwrap.dedent("""
    Extract Named Entities, Relationships, and their attributes in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful descriptions for each entity to add context.
    """)

# 2. Provide a high-quality example to guide the model
examples_1 = [
    lx.data.ExampleData(
        text="Models can exhibit dramatic personality shifts at deployment time in response to prompting or context. For example, Microsoft’s Bing chatbot would sometimes slip into a mode of threatening and manipulating users",
        extractions=[
            lx.data.Extraction(
                extraction_class="Concept",
                extraction_text="personality",
                attributes={ "description": "personality is a concept that relates to LLMs"}
            ),
            lx.data.Extraction(
                extraction_class="Concept",
                extraction_text="prompting",
                attributes={ "description": "prompting is a concept related to LLMs"}
            ),
            lx.data.Extraction(
                extraction_class="Company",
                extraction_text="Microsoft",
                attributes={ "description": "Microsoft is a company in the AI technology industry"}
            ),
            lx.data.Extraction(
                extraction_class="Product",
                extraction_text="Bing",
                attributes={"description": "Bing is a product by Microsoft"}
            ),
            lx.data.Extraction(
                extraction_class="Product",
                extraction_text="chatbot",
                attributes={"description": "chatbot is a function of Bing"}
            ),
            lx.data.Extraction(
                extraction_class="Relationship",
                extraction_text="Models can exhibit dramatic personality shifts",
                attributes={"subject": "Models", 
                            "relation": "exhibit",
                            "object": "dramatic personality shifts",
                            "evidence": "Models can exhibit dramatic personality shifts at deployment time in response to prompting or context"}
            ),
        ]
    )
]

prompt = textwrap.dedent("""\
    Extract Named Entities, Relationships and their attributes in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful descriptions for each entity to add context. 
    Focus on Named Entity only, not all nouns, pronouns or adjective-noun compositions are considered Named entities, do not assume their types.
    Relationships should contain subject and object entities that are only from extracted Entities.
    Perform comprehensive extractions on all instances of entities, and relations in the text
    """)


examples = [
    lx.data.ExampleData(
        text="Models can exhibit dramatic personality shifts at deployment time in response to prompting or context. For example, Microsoft’s Bing chatbot would sometimes slip into a mode of threatening and manipulating users",
        extractions=[
            lx.data.Extraction(
                extraction_class="NamedEntity",
                extraction_text="Microsoft",
                attributes={ "description": "Microsoft is a company that is assumed evil"}
            ),
            lx.data.Extraction(
                extraction_class="NamedEntity",
                extraction_text="Bing",
                attributes={"description": "Bing is a product"}
            ),
            lx.data.Extraction(
                extraction_class="Relationship",
                extraction_text="Microsoft owns a chatbot product called Bing",
                attributes={"subject": "Microsoft", 
                            "relation": "owns",
                            "object": "Bing",
                            "evidence": "For example, Microsoft’s Bing chatbot"}
            ),
        ]
    )
]
input_text = """
Zochi is an artificial scientist system capable of end-to-end scientific discovery, from hypothesis generation through experimentation to peer-reviewed publication. Unlike previous systems that automate isolated aspects of scientific research, Zochi demonstrates comprehensive capabilities across the complete research lifecycle.

We present empirical validation through multiple peer-reviewed publications accepted at ICLR 2025 workshops and ACL 2025, each containing novel methodological contributions and state-of-the-art experimental results. These include Compositional Subspace Representation Fine-tuning (CS-ReFT), which achieved a 93.94% win rate on the AlpacaEval benchmark on Llama-2-7b while using only 0.0098% of model parameters, the Tempest (formerly Siege) framework, a state-of-the-art jailbreak which identified critical vulnerabilities in language model safety measures through multi-turn adversarial testing.
"""

def main(provider='google', model_id=None):
    if provider == 'google':
        model_id = model_id or "gemini-2.5-flash"
        language_model_type = inference.GeminiLanguageModel
        language_model_params = {}
    elif provider == 'openai':
        model_id = model_id or "gpt-5-nano"
        language_model_type = inference.OpenAILanguageModel
        # Configure GPT-5 specific parameters
        if model_id.startswith('gpt-5'):
            language_model_params = {
                'reasoning_effort':  'minimal',
                'verbosity': 'high',
            }
        else:
            language_model_params = {}
    elif provider == 'anthropic':
        model_id = model_id or "claude-3-5-haiku-latest"
        language_model_type = inference.ClaudeLanguageModel
        language_model_params = {}
    elif provider == 'hf':
        model_id = model_id or 'openai/gpt-oss-120b:cerebras' # gpt-oss-120b, gpt-oss-20b
        language_model_type = inference.HFLanguageModel
        language_model_params = {}
    else:
        raise ValueError(f"Invalid provider: {provider}")

    # Process text with optimized parameters for each provider
    result = lx.extract(
        text_or_documents= input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        language_model_type=language_model_type,
        language_model_params=language_model_params,
        extraction_passes=1,    # Use 1 pass for GPT-5 stability
        max_workers=10,         # Single worker for debugging
        max_char_buffer=1000,   # Smaller contexts for better accuracy
        debug=True,
        temperature=0.0, 
    )

    lx.io.save_annotated_documents([result], output_name=f"{provider}_extraction_results.jsonl")

    html_content = lx.visualize(f"output/{provider}_extraction_results.jsonl")

    # Convert HTML object to string if needed
    if hasattr(html_content, 'data'):
        # HTML object from IPython.display
        html_string = html_content.data
    else:
        # Already a string
        html_string = html_content

    with open(f"output/{provider}_extraction_results_visualization.html", "w") as f:
        f.write(html_string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LangExtract Example - Extract entities and relationships from text')
    parser.add_argument('--provider', '-p', 
                       choices=['google', 'openai', 'anthropic', 'hf'],
                       default='openai',
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--model', '-m',
                       help='Specific model ID to use (overrides provider default)')
    
    args = parser.parse_args()
    
    print(f"🚀 Running LangExtract with {args.provider} provider")
    if args.model:
        print(f"📋 Using model: {args.model}")
    
    main(provider=args.provider, model_id=args.model)