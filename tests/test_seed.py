#!/usr/bin/env python3
"""Simple test to verify seed parameter functionality."""

import os
import pytest
import langextract
from langextract import data

def test_seed_parameter_acceptance():
    """Test that seed parameter is accepted and stored properly."""
    
    # Set up test data
    text = "John Smith works at Google as a software engineer. He lives in San Francisco."
    prompt = "Extract the person's name, company, and job title."
    seed_value = 12345
    
    # Create required examples
    examples = [
        data.ExampleData(
            text="Alice Johnson is a Data Scientist at Microsoft. She works in Seattle.",
            extractions=[
                data.Extraction(
                    extraction_class="person",
                    extraction_text="Alice Johnson"
                ),
                data.Extraction(
                    extraction_class="company",
                    extraction_text="Microsoft"
                ),
                data.Extraction(
                    extraction_class="job_title",
                    extraction_text="Data Scientist"
                )
            ]
        )
    ]
    
    # Make sure we have API key
    api_key = os.environ.get("LANGEXTRACT_API_KEY")
    if not api_key:
        pytest.skip("LANGEXTRACT_API_KEY environment variable not set")
    
    print("Testing seed parameter acceptance...")
    print(f"Using seed: {seed_value}")
    print("Note: Claude API doesn't currently support seeds, but parameter should be accepted")
    
    # Test that extract accepts seed parameter without error
    try:
        result = langextract.extract(
            text,
            prompt_description=prompt,
            examples=examples,
            seed=seed_value,
            temperature=0.0,
            api_key=api_key
        )
        print("✅ SUCCESS: Seed parameter accepted without error")
        print(f"Extraction result: {len(result.extractions)} extractions found")
        
    except Exception as e:
        print(f"❌ FAILED: Error with seed parameter: {e}")
        raise

if __name__ == "__main__":
    test_seed_parameter_acceptance()