from langextract.accrete_data_models import (
    EntityResponse, RelationResponse, Entity, EntityType
)
from langextract.data import AnnotatedDocument, Extraction, CharInterval
from langextract import tokenizer
import langextract as lx


def find_entity_positions(entity_text: str, document_text: str, tokenized_text: tokenizer.TokenizedText = None) -> list[tuple[CharInterval, tokenizer.TokenInterval | None]]:
    """
    Find all occurrences of an entity in the document text and return their positions.
    Only matches complete words/phrases with proper word boundaries and exact case matching.
    
    Args:
        entity_text: The entity text to search for (case-sensitive)
        document_text: The full document text to search in
        tokenized_text: Optional pre-tokenized text for efficiency
        
    Returns:
        List of tuples (CharInterval, TokenInterval) for all found occurrences, or empty list if not found
    """
    if not document_text or not entity_text:
        return []
    
    import re
    
    # Escape special regex characters in entity_text
    escaped_entity = re.escape(entity_text)
    
    # Create pattern with flexible word boundaries
    # Use lookbehind and lookahead to ensure the entity is not part of another word
    # (?<!\w) = negative lookbehind for word character
    # (?!\w) = negative lookahead for word character
    pattern = r'(?<!\w)' + escaped_entity + r'(?!\w)'
    
    # Pre-tokenize text once if not provided for efficiency
    if tokenized_text is None:
        tokenized_text = tokenizer.tokenize(document_text)
    
    # Find all character positions (case-sensitive search with word boundaries)
    matches = list(re.finditer(pattern, document_text))
    
    if not matches:
        return []
    
    results = []
    
    for match in matches:
        char_start = match.start()
        char_end = match.end()
        char_interval = CharInterval(start_pos=char_start, end_pos=char_end)
        
        # Find tokens that overlap with the character interval
        start_token_idx = None
        end_token_idx = None
        
        for i, token in enumerate(tokenized_text.tokens):
            token_start = token.char_interval.start_pos
            token_end = token.char_interval.end_pos
            
            # Check if token overlaps with entity character interval
            if (token_start < char_end and token_end > char_start):
                if start_token_idx is None:
                    start_token_idx = i
                end_token_idx = i + 1  # end_index is exclusive
        
        # Create token interval if tokens were found
        token_interval = None
        if start_token_idx is not None and end_token_idx is not None:
            token_interval = tokenizer.TokenInterval(
                start_index=start_token_idx,
                end_index=end_token_idx
            )
        
        results.append((char_interval, token_interval))
    
    return results


def entity_response_to_annotated_document(entity_response: EntityResponse, 
                                        document_id: str = None,
                                        text: str = None) -> AnnotatedDocument:
    """
    Convert EntityResponse to AnnotatedDocument.
    
    Args:
        entity_response: The EntityResponse object to convert
        document_id: Optional document ID for the AnnotatedDocument
        text: Optional source text for the AnnotatedDocument
    
    Returns:
        AnnotatedDocument with extracted entities as Extraction objects
    """
    extractions = []
    tokenized_text = None
    
    # Pre-tokenize text once if provided for efficiency
    if text:
        tokenized_text = tokenizer.tokenize(text)
    
    for idx, entity in enumerate(entity_response.output):
        if entity.isNamedEntity:
            # Find all entity positions in document text
            positions = find_entity_positions(
                entity.NamedEntity, text, tokenized_text
            )
            
            # Create an extraction for each occurrence
            for occurrence_idx, (char_interval, token_interval) in enumerate(positions):
                extraction = Extraction(
                    extraction_class=entity.NamedEntityType.BroadType,
                    extraction_text=entity.NamedEntity,
                    extraction_index=idx * 1000 + occurrence_idx,  # Unique index for each occurrence
                    char_interval=char_interval,
                    token_interval=token_interval,
                    attributes={
                        'Description': entity.NamedEntityDescription,
                        'CanonicalName': entity.CanonicalNamedEntity,
                        'SpecificType': entity.NamedEntityType.SpecificType,
                        'FineType': entity.NamedEntityType.FineType,
                        'OccurrenceIndex': occurrence_idx
                    }
                )
                extractions.append(extraction)
    
    return AnnotatedDocument(
        document_id=document_id,
        extractions=extractions,
        text=text
    )

def test_adapter_with_Accrete_extraction(file_path: str) -> str:
    import pandas, json
    from typing import Iterable
    with open(file_path, 'r') as f:
        data = pandas.DataFrame(json.load(f))
    data['_len'] = data['text'].apply(len)
    data.sort_values(by='_len', ascending=False, inplace=True)
    print(data.head(20))
    # Create an iterable to store entities with document_index
    annotated_docs: Iterable[AnnotatedDocument] = []
    for _, row in data[1:10].iterrows():
        entities = []
        for ent in row.entities:
            if isinstance(ent, dict):
                entities.append(Entity(**ent))

        annotated_doc = entity_response_to_annotated_document(
            EntityResponse(output=entities),
            document_id=row.document_index,
            text=row.text
        )
        annotated_docs.append(annotated_doc)
    print(len(annotated_docs))
    output_file = lx.io.save_annotated_documents(annotated_docs, output_name=f"{file_path.split('/')[-1]}_adapter_example.jsonl")
    lx.visualize(output_file, save_html=True)

    return output_file

if __name__ == "__main__":
    # Example usage
    entity_response = EntityResponse(output=[
        Entity(
        NamedEntity="John Doe", 
        NamedEntityDescription="John Doe is a person", 
        CanonicalNamedEntity="John Doe", 
        isNamedEntity=True, 
        NamedEntityType=EntityType(BroadType="PERSON", 
                                   SpecificType="PERSON", 
                                   FineType="PERSON")),
        Entity(
        NamedEntity="Jane Doe", 
        NamedEntityDescription="Jane Doe is a person", 
        CanonicalNamedEntity="Jane Doe", 
        isNamedEntity=True, 
        NamedEntityType=EntityType(BroadType="PERSON", 
                                   SpecificType="PERSON", 
                                   FineType="PERSON"))
    ])
    
    print("Original EntityResponse JSON:")
    print(entity_response.to_json())
    
    print("\nConverted to AnnotatedDocument:")
    test_text = "John Doe is person, he is mentioned in this text. Jane Doe is a person too."
    annotated_doc = entity_response_to_annotated_document(
        entity_response, 
        document_id="example_doc", 
        text=test_text
    )

    # output to .jsonl file and convert to html
    output_file = lx.io.save_annotated_documents([annotated_doc], output_name="adapter_example.jsonl")
    lx.visualize(output_file, save_html=True)

    output_file = test_adapter_with_Accrete_extraction("docs/examples/washington_post_dc.final.json")
    print(f"Output visualization: {output_file}")
