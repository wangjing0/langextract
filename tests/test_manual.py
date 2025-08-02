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

    if actual_schema == expected_schema:
        print("✅ ClaudeSchema test PASSED")
        return True
    else:
        print("❌ ClaudeSchema test FAILED")
        print("Expected:", expected_schema)
        print("Actual:", actual_schema)
        return False

def test_claude_language_model():
    """Test that ClaudeLanguageModel can be imported and initialized."""
    from langextract import inference
    
    # Test class exists
    assert hasattr(inference, 'ClaudeLanguageModel'), "ClaudeLanguageModel class missing"
    
    # Test old class doesn't exist
    assert not hasattr(inference, 'GeminiLanguageModel'), "GeminiLanguageModel still exists"
    
    # Test initialization
    try:
        model = inference.ClaudeLanguageModel(api_key='dummy_key')
        assert model.model_id == 'claude-3-5-haiku-latest', f"Wrong default model: {model.model_id}"
        print("✅ ClaudeLanguageModel test PASSED")
        return True
    except Exception as e:
        print(f"❌ ClaudeLanguageModel test FAILED: {e}")
        return False

if __name__ == "__main__":
    schema_passed = test_claude_schema()
    model_passed = test_claude_language_model()
    
    if schema_passed and model_passed:
        print("\n🎉 ALL TESTS PASSED - Gemini to Claude migration successful!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)