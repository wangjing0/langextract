from langextract.accrete_data_models import (
    EntityResponse, RelationResponse, Entity, EntityType
)
from langextract.data import AnnotatedDocument, Extraction, CharInterval
from langextract import tokenizer
import langextract as lx


def find_entity_position(entity_text: str, document_text: str, tokenized_text: tokenizer.TokenizedText = None) -> tuple[CharInterval | None, tokenizer.TokenInterval | None]:
    """
    Find the first occurrence of an entity in the document text and return its position.
    
    Args:
        entity_text: The entity text to search for
        document_text: The full document text to search in
        tokenized_text: Optional pre-tokenized text for efficiency
        
    Returns:
        Tuple of (CharInterval, TokenInterval) or (None, None) if not found
    """
    if not document_text or not entity_text:
        return None, None
    
    # Find character position (case-insensitive search)
    entity_lower = entity_text.lower()
    doc_lower = document_text.lower()
    char_start = doc_lower.find(entity_lower)
    
    if char_start == -1:
        return None, None
    
    char_end = char_start + len(entity_text)
    char_interval = CharInterval(start_pos=char_start, end_pos=char_end)
    
    # Find token position
    if tokenized_text is None:
        tokenized_text = tokenizer.tokenize(document_text)
    
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
    
    if start_token_idx is not None and end_token_idx is not None:
        token_interval = tokenizer.TokenInterval(
            start_index=start_token_idx,
            end_index=end_token_idx
        )
    
    return char_interval, token_interval


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
            # Find entity position in document text
            char_interval, token_interval = find_entity_position(
                entity.NamedEntity, text, tokenized_text
            )
            
            extraction = Extraction(
                extraction_class=entity.NamedEntityType.BroadType,
                extraction_text=entity.NamedEntity,
                extraction_index=idx,
                char_interval=char_interval,
                token_interval=token_interval,
                attributes={
                    'Description': entity.NamedEntityDescription,
                    'CanonicalName': entity.CanonicalNamedEntity,
                    'SpecificType': entity.NamedEntityType.SpecificType,
                    'FineType': entity.NamedEntityType.FineType
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
    with open(file_path, 'r') as f:
        data = pandas.DataFrame(json.load(f))
    print(data.head(20))
    # Create a list to store entities with document_index
    annotated_docs = []
    for _, row in data[-1:].iterrows():
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
