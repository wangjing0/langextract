import langextract as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
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
                extraction_class="Company",
                extraction_text="Microsoft",
                attributes={ "description": "Microsoft is a company that is evil"}
            ),
            lx.data.Extraction(
                extraction_class="Product",
                extraction_text="Bing",
                attributes={"description": "Bing is a product that is evil"}
            ),
            lx.data.Extraction(
                extraction_class="Relationship",
                extraction_text="Models can exhibit dramatic personality shifts",
                attributes={"description": "Models can exhibit dramatic personality shifts at deployment time in response to prompting or context"}
            ),
        ]
    )
]

input_text = """"
Large language models interact with users through a simulated “Assistant” persona. While the Assistant is typically trained to be helpful, harmless, and honest,
it sometimes deviates from these ideals. In this paper, we identify directions in the model’s activation space—persona vectors—underlying several traits, such as
evil, sycophancy, and propensity to hallucinate. We confirm that these vectors can be used to monitor fluctuations in the Assistant’s personality at deployment time.
We then apply persona vectors to predict and control personality shifts that occur during training. We find that both intended and unintended personality changes
after finetuning are strongly correlated with shifts along the relevant persona vectors. These shifts can be mitigated through post-hoc intervention, or avoided in
the first place with a new preventative steering method. Moreover, persona vectors can be used to flag training data that will produce undesirable personality
changes, both at the dataset level and the individual sample level. Our method for extracting persona vectors is automated and can be applied to any personality trait
of interest, given only a natural-language description
"""

# Process Romeo & Juliet directly from Project Gutenberg
result = lx.extract(
    text_or_documents= input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="claude-3-5-haiku-latest",
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=10,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)

lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl")

html_content = lx.visualize("output/extraction_results.jsonl")

# Convert HTML object to string if needed
if hasattr(html_content, 'data'):
    # HTML object from IPython.display
    html_string = html_content.data
else:
    # Already a string
    html_string = html_content

with open("output/extraction_results_visualization.html", "w") as f:
    f.write(html_string)