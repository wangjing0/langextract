#!/usr/bin/env python3

import sys
sys.path.insert(0, './langextract')

from langextract import schema, data

def test_claude_schema():
    """Test that ClaudeSchema works correctly."""
    examples_data = [
        data.ExampleData(
            text='Patient has diabetes.',
            extractions=[
                data.Extraction(
                    extraction_text='diabetes',
                    extraction_class='condition',
                    attributes={'chronicity': 'chronic'},
                )
            ],
        )
    ]

    expected_schema = {
        'type': 'object',
        'properties': {
            'extractions': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'condition': {'type': 'string'},
                        'condition_attributes': {
                            'type': 'object',
                            'properties': {
                                'chronicity': {'type': 'string'},
                            },
                            'nullable': True,
                        },
                    },
                },
            },
        },
        'required': ['extractions'],
    }

    claude_schema = schema.ClaudeSchema.from_examples(examples_data)
    actual_schema = claude_schema.schema_dict

    assert actual_schema == expected_schema, (
        f"ClaudeSchema test FAILED\n"
        f"Expected: {expected_schema}\n"
        f"Actual: {actual_schema}"
    )

def test_claude_language_model():
    """Test that ClaudeLanguageModel can be imported and initialized."""
    from langextract import inference
    
    # Test class exists
    assert hasattr(inference, 'ClaudeLanguageModel'), "ClaudeLanguageModel class missing"
    
    # Test old class doesn't exist
    assert not hasattr(inference, 'GeminiLanguageModel'), "GeminiLanguageModel still exists"
    
    # Test initialization
    model = inference.ClaudeLanguageModel(api_key='dummy_key')
    assert model.model_id == 'claude-3-5-haiku-latest', f"Wrong default model: {model.model_id}"

if __name__ == "__main__":
    try:
        test_claude_schema()
        test_claude_language_model()
        print("\n🎉 ALL TESTS PASSED - Gemini to Claude migration successful!")
    except AssertionError as e:
        print(f"\n💥 Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)