import argparse
import langextract as lx
import textwrap
from langextract import inference

# 1. Define the prompt and extraction rules
prompt_1 = textwrap.dedent("""
    Extract Named Entities, EntityRelations, and their attributes in order of appearance.
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
                extraction_class="EntityRelation",
                extraction_text="Models can exhibit dramatic personality shifts",
                attributes={"subject": "Models", 
                            "relation": "exhibit",
                            "object": "dramatic personality shifts",
                            "evidence": "Models can exhibit dramatic personality shifts at deployment time in response to prompting or context"}
            ),
        ]
    )
]

# Type free NER and RE extraction
prompt = textwrap.dedent("""\
    Extract NamedEntities, EntityRelations, HyperRelations and their attributes in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful descriptions for each entity to add context. 
    Focus on Named Entity only, not all nouns, pronouns or adjective-noun compositions are considered Named entities, do not assume stereotypic types, but infer their types from the context.
    EntityRelations should contain subject and object entities that are only from extracted Entities.
    HyperRelations should contain a set of NamedEntities as members, that are related to each other semantically given the context, without specifying the pairwise relations.  For instance, location, person1, person2, organization1, organization2 are related in an event, they should be extracted as a HyperRelation.
    Perform comprehensive extractions on all instances of entities, and relations in the text
    """)

examples = [
    lx.data.ExampleData(
        text="""
Models can exhibit dramatic personality shifts at deployment time in response to prompting or context. For example, Microsoft’s Bing chatbot would sometimes slip into a mode of threatening and manipulating users
""",
        extractions=[
            lx.data.Extraction(
                extraction_class="NamedEntity",
                extraction_text="Microsoft",
                attributes={ "description": "Microsoft is a company that is assumed evil", "type": "Organization"}
            ),
            lx.data.Extraction(
                extraction_class="NamedEntity",
                extraction_text="Bing",
                attributes={"description": "Bing is a product", "type": "Product"}
            ),
            lx.data.Extraction(
                extraction_class="NamedEntity",
                extraction_text="chatbot",
                attributes={"description": "chatbot is a function of the product Bing", "type": "Product"}
            ),
            lx.data.Extraction(
                extraction_class="EntityRelation",
                extraction_text="Microsoft owns a chatbot product called Bing",
                attributes={"subject": "Microsoft", 
                            "relation": "owns",
                            "object": "Bing",
                            "evidence": "For example, Microsoft’s Bing chatbot"}
            ),
            lx.data.Extraction(
                extraction_class="HyperRelation",
                extraction_text="Microsoft, Bing, chatbot, threatening, manipulating, users, mode",
                attributes={"members": ["Microsoft",  "Bing", "chatbot"],
                            "description": "Microsoft, Bing, chatbot are related in the product review"}
            ),
        ]
    )
]

input_text = """
Zochi Achieves Main Conference Acceptance at ACL 2025
Today, we’re excited to announce a groundbreaking milestone: Zochi, Intology’s Artificial Scientist, has become the first AI system to independently pass peer review at an A* scientific conference¹—the highest bar for scientific work in the field.

Zochi’s paper has been accepted into the main proceedings of ACL—the world’s #1 scientific venue for natural language processing (NLP), and among the top 40 of all scientific venues globally.²

While recent months have seen several groups, including our own, demonstrate AI-generated contributions at workshop venues, having a paper accepted to the main proceedings of a top-tier scientific conference represents clearing a significantly higher bar. While workshops³, at the level submitted to ICLR 2025, have acceptance rates of ~60-70%, main conference proceedings at conferences such as ACL (NeurIPS, ICML, ICLR, CVPR, etc…) have acceptance rates of ~20%. ACL is often the most selective of these conferences

This achievement marks a watershed moment in the evolution of innovation. For the first time, an artificial system has independently produced a scientific discovery and published it at the level of the field’s top researchers—making Zochi the first PhD-level agent. The peer review process for the main conference proceedings of such venues is designed to be highly selective, with stringent standards for novelty, technical depth, and experimental rigor. To put this achievement in perspective, most PhD students in computer science spend several years before publishing at a venue of this stature. AI has crossed a threshold of scientific creativity that allows for contributions alongside these researchers at the highest level of inquiry.

Autonomously Conducting the Scientific Method
Zochi is an AI research agent capable of autonomously completing the entire scientific process—from literature analysis to peer-reviewed publication. The system operates through a multi-stage pipeline designed to emulate the scientific method. Zochi begins by ingesting and analyzing thousands of research papers to identify promising directions within a given domain. Its retrieval system identifies key contributions, methodologies, limitations, and emerging patterns across the literature. What distinguishes Zochi is its ability to identify non-obvious connections across papers and propose innovative solutions that address fundamental limitations rather than incremental improvements.

Zochi's experimentation pipeline converts conceptual insights into rigorous evaluations. The system autonomously implements methods, designs controlled experiments, and conducts comprehensive evaluations, including ablation studies. Our automatic validation engine generates evaluation scripts based on standardized datasets that remain unmodified throughout testing, ensuring results reflect genuine improvements. Experiments are parallelized across multiple trials, significantly accelerating the research timeline compared to human-driven approaches while maintaining scientific rigor. Methods typically only require hours to validate, and a full paper takes only days to complete.

The current version of Zochi represents a substantial advancement over our earlier systems that published workshop papers at ICLR 2025. The latest system operates autonomously without human involvement except during manuscript preparation—typically limited to figure creation, citation formatting, and minor fixes. This capability to lead rather than merely assist with research has potentially transformative implications for scientific discovery, allowing for comprehensive exploration of complex research questions and accelerating the pace of innovation across numerous domains.

Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search
Zochi's first major conference publication, "Tempest: Automatic Multi-Turn Jailbreaking of Large Language Models with Tree Search," showcases the system's advanced capabilities. A preliminary version of this work (formally known as Siege) had been accepted to the ICLR workshops, but Zochi was able to significantly improve its design and conduct more comprehensive experiments for the ACL submission. The level of autonomy in this research is particularly notable—humans supplied only the general research domain of "novel jailbreaking methods" as input. From this starting point, Zochi independently identified the research direction of multi-turn attacks⁴, formalized the Tempest method, implemented and tested it, conducted all experiments, and wrote/presented the paper (excluding figures and minor formatting fixes made by humans).

Starting from jailbreaking literature analysis, Zochi designed a tree search methodology with parallel exploration that branches multiple adversarial prompts simultaneously, incorporating a cross-branch learning mechanism and a robust partial compliance tracking system. The system implemented Tempest autonomously, conducting comprehensive evaluations across multiple language models that demonstrated significant improvements over existing methods. Tempest achieves a 100% success rate on GPT-3.5-turbo and 97% on GPT-4, outperforming both single-turn methods and multi-turn baselines while using fewer queries.

This work reveals how language model safety boundaries can be systematically eroded through natural conversation, with minor policy concessions accumulating into fully disallowed outputs. The findings expose critical vulnerabilities in current safety mechanisms and provide essential insights for developing more robust defenses against multi-turn adversarial attacks.
The quality of this work is particularly notable given ACL's highly selective acceptance rate of 21.3% last year. Even more impressive, Zochi's paper received a final meta-review score of 4, placing it within the top 8.2% of all submissions. While early drafts contained minor writing errors similar to human researchers, the core scientific content was remarkably strong.

Zochi Beta Program
We’re excited to announce that we will soon be rolling out Zochi to the public through a beta program. We will be releasing Zochi in stages, with the first release being a domain-agnostic research copilot adapted to work more collaboratively with human researchers. This tool can help researchers identify promising research directions, generate novel research hypotheses, write grant proposals and survey papers, and design rigorous experiments across fields. We'll be sharing more on this very soon from early testers.

Following this initial release, we plan to gradually roll out Zochi’s end-to-end research capabilities, enabling increasingly autonomous assistance throughout the research process. We hope these developments will unlock new forms of collaboration that harness the unique strengths of both human and artificial intelligence and dramatically accelerate discovery across disciplines.

Ethical Considerations
We acknowledge that AI-driven research, while exciting, creates new challenges for scientific accountability and reproducibility. While Zochi operates autonomously, human researchers remain as authors and maintain responsibility for validating methods, interpreting results, and ensuring ethical compliance. For our paper, we conducted multiple rounds of internal review, carefully verified all results and code before submission, and fixed minor formatting and writing errors. We also engaged with the reviewers and wrote the rebuttal manually, without the involvement of our system. We encourage listing AI systems in the Acknowledgements, not as authors. While AI-driven research raises important questions about attribution, transparency, and accountability, we maintain that intellectual contributions should be judged by their substance rather than their source. 

Our main focus and priority is assisting human researchers through collaborative AI tools. While we demonstrated fully autonomous research capabilities to showcase our system's potential in the most challenging setting, the Zochi system will be offered primarily as a research copilot designed to augment human researchers through collaboration. We submitted only a single paper because we deemed this particular contribution valuable enough to warrant publication and peer review, while being mindful not to overwhelm the review process.

Given that Zochi's first major publication focuses on AI safety vulnerabilities, we follow responsible disclosure protocols and prioritize research that strengthens rather than undermines AI safety. We commit to working with the broader academic community to establish comprehensive frameworks for AI participation in research, including authorship standards, adapted review procedures, and enhanced validation requirements. We believe thoughtful integration of AI and human intelligence can advance scientific discovery while maintaining the integrity and collaborative nature of research.
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
                'reasoning_effort': 'low',
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

    result = lx.extract(
        text_or_documents=input_text,
        prompt_description=prompt,
        examples=examples,
        model_id=model_id,
        language_model_type=language_model_type,
        language_model_params=language_model_params,
        extraction_passes=3,    # 1 pass for token usage, more passes for better coverage
        max_workers=10,         # Workers for parallel processing
        max_char_buffer=1_000,  # Smaller contexts for better accuracy
        debug=True,
        temperature=0.0,        # Deterministic temperature
        seed=42,                # Fixed seed for reproducible results
    )
    output_file = lx.io.save_annotated_documents([result], output_name=f"{provider}_extraction_results.jsonl")
    
    html_content = lx.visualize(output_file)
    
    html_path = output_file.with_suffix('.html')
    with open(html_path, "w") as f:
        f.write(getattr(html_content, 'data', html_content))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LangExtract Example')
    parser.add_argument('--provider', '-p', 
                       choices=['google', 'openai', 'anthropic', 'hf'],
                       default='openai',
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--model', '-m',
                       help='Specific model to use (overrides provider default)')
    
    args = parser.parse_args()

    main(provider=args.provider, model_id=args.model)