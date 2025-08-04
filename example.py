import langextract as lx
import textwrap
from langextract import inference

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""
    Extract Named Entities, Relationships, and their attributes in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful descriptions for each entity to add context.
    """)

# 2. Provide a high-quality example to guide the model
examples = [
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

input_text = """
Langfun is a PyGlove powered library that aims to make language models (LM) fun to work with. Its central principle is to enable seamless integration between natural language and programming by treating language as functions. Through the introduction of Object-Oriented Prompting, Langfun empowers users to prompt LLMs using objects and types, offering enhanced control and simplifying agent development.

To unlock the magic of Langfun, you can start with Langfun 101. Notably, Langfun is compatible with popular LLMs such as Gemini, GPT, Claude, all without the need for additional fine-tuning.
"""

# Process Romeo & Juliet directly from Project Gutenberg
result = lx.extract(
    text_or_documents= input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="claude-3-5-haiku-latest",
    language_model_type=inference.ClaudeLanguageModel,
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=10,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)

lx.io.save_annotated_documents([result], output_name="claude_extraction_results.jsonl")

html_content = lx.visualize("output/claude_extraction_results.jsonl")

# Convert HTML object to string if needed
if hasattr(html_content, 'data'):
    # HTML object from IPython.display
    html_string = html_content.data
else:
    # Already a string
    html_string = html_content

with open("output/claude_extraction_results_visualization.html", "w") as f:
    f.write(html_string)



result = lx.extract(
    text_or_documents= input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o-mini",
    language_model_type=inference.OpenAILanguageModel,
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=10,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)
lx.io.save_annotated_documents([result], output_name="gpt_extraction_results.jsonl")

html_content = lx.visualize("output/gpt_extraction_results.jsonl")

# Convert HTML object to string if needed
if hasattr(html_content, 'data'):
    html_string = html_content.data
else:
    html_string = html_content

with open("output/gpt_extraction_results_visualization.html", "w") as f:
    f.write(html_string)



result = lx.extract(
    text_or_documents= input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    language_model_type=inference.GeminiLanguageModel,
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=10,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)
lx.io.save_annotated_documents([result], output_name="gemini_extraction_results.jsonl")

html_content = lx.visualize("output/gemini_extraction_results.jsonl")

# Convert HTML object to string if needed
if hasattr(html_content, 'data'):
    html_string = html_content.data
else:
    html_string = html_content

with open("output/gemini_extraction_results_visualization.html", "w") as f:
    f.write(html_string)