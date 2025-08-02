import langextract as lx
import textwrap

# Quick test with smaller text
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

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

# Smaller input text for faster processing
input_text = """
JULIET
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love,
And I'll no longer be a Capulet.

ROMEO
Shall I hear more, or shall I speak at this?

JULIET  
'Tis but thy name that is my enemy;
Thou art thyself, though not a Montague.
"""

print("🚀 Testing Claude integration with Romeo & Juliet excerpt...")

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="claude-3-5-haiku-latest",
    extraction_passes=1,    # Single pass for speed
    max_workers=2,          # Reduced workers
    max_char_buffer=1000    
)

print(f"\n✅ Success! Extracted {len(result.extractions)} entities:")
for i, extraction in enumerate(result.extractions, 1):
    attrs = ", ".join(f"{k}: {v}" for k, v in extraction.attributes.items()) if extraction.attributes else "no attributes"
    print(f"  {i}. {extraction.extraction_class}: '{extraction.extraction_text}' ({attrs})")

# Save results to file
try:
    lx.io.save_annotated_documents([result], output_name="quick_test_results.jsonl")
    print(f"\n📁 Results saved to: quick_test_results.jsonl")
    html_content = lx.visualize("output/quick_test_results.jsonl")
    with open("output/quick_test_results.html", "w") as f:
        f.write(html_content)
except Exception as e:
    print(f"\n⚠️  File save failed: {e}")
    print("But extraction was successful!")