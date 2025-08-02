import langextract as lx
import textwrap

# Simple test to verify Claude integration
prompt = "Extract characters and emotions from the text."

examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks?",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
        ]
    )
]

input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Test extraction without file saving
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="claude-3-5-haiku-latest",
    extraction_passes=1,
    max_workers=1,
    max_char_buffer=1000
)

print("\n✅ Claude integration successful!")
print(f"Extracted {len(result.extractions)} entities:")
for extraction in result.extractions:
    print(f"  - {extraction.extraction_class}: '{extraction.extraction_text}'")