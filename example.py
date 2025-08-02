import langextract as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]


input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Process Romeo & Juliet directly from Project Gutenberg
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="claude-3-5-haiku-latest",
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=10,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)

lx.io.save_annotated_documents([result], output_name="extraction_results_romeo_juliet.jsonl")

html_content = lx.visualize("output/extraction_results_romeo_juliet.jsonl")
with open("output/visualization_romeo_juliet.html", "w") as f:
    f.write(html_content)