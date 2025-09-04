from typing import List, Optional, TypedDict
from pydantic import BaseModel, Field, field_validator


class EntityTypeDefinition(TypedDict):
    type_name: str
    definition: str

ENTITY_TYPES: List[EntityTypeDefinition] = [
    {
        "type_name": "ACADEMIC",
        "definition": "Educational institutions, academic programs, degrees, research fields, and scholarly concepts.",
    },
    {
        "type_name": "ACTIVITY",
        "definition": "Actions, processes, or sustained endeavors performed by agents (people, organizations).",
    },
    {
        "type_name": "BIOLOGY",
        "definition": "Biological sciences, processes, concepts, fields of study, and biological molecules/parts (distinct from whole organisms).",
    },
    {
        "type_name": "CREATIVE_WORK",
        "definition": "Named artistic creations, literary works, performances, broadcasts, and other forms of creative expression. Also includes musical works (songs, albums, operas), genres, instruments, bands/artists, and musical concepts.",
    },
    {
        "type_name": "CULTURAL_PHENOMENA",
        "definition": "Cultural practices, traditions, movements, named beliefs (non-religious), and social customs.",
    },
    {
        "type_name": "DATE",
        "definition": "Specific calendar dates, named time periods (e.g., holidays, historical eras), and date ranges.",
    },
    {
        "type_name": "DEMOGRAPHIC",
        "definition": "Named population groups based on characteristics like ethnicity, nationality, age, gender, or social status.",
    },
    {
        "type_name": "EVENT",
        "definition": "Specific named occurrences, incidents, happenings, planned gatherings, elections, natural disasters, or historical events with a defined scope.",
    },
    {
        "type_name": "FACILITY",
        "definition": "Named physical structures, buildings, infrastructure, and places designed for specific purposes (e.g., airports, bridges, hospitals, stadiums).",
    },
    {
        "type_name": "FINANCIAL",
        "definition": "Monetary concepts, economic indicators, currencies, stock names/indices, financial instruments, taxes, and economic activities/sectors.",
    },
    {
        "type_name": "FOOD",
        "definition": "Named food items, dishes, beverages, ingredients, cuisines, and dietary elements.",
    },
    {
        "type_name": "INDUSTRY_SECTOR",
        "definition": "Named business sectors, industries, and economic domains.",
    },
    {
        "type_name": "LANGUAGE",
        "definition": "Specific named natural or artificial languages.",
    },
    {
        "type_name": "LEGAL",
        "definition": "Named laws, treaties, regulations, legal concepts, judicial bodies, case names, and legal roles/proceedings.",
    },
    {
        "type_name": "LOCATION",
        "definition": "Named geographic places (cities, countries, continents), regions, political/administrative divisions, natural landforms, bodies of water, and astronomical objects/locations.",
    },
    {
        "type_name": "MATERIAL",
        "definition": "Named physical substances, chemicals, elements, minerals, textiles, and building materials.",
    },
    {
        "type_name": "MATHEMATICAL",
        "definition": "Named mathematical concepts, theorems, formulas, constants, and numerical systems.",
    },
    {
        "type_name": "MEDICAL",
        "definition": "Named medical conditions, diseases, syndromes, treatments, procedures, drugs, anatomical parts, medical devices, and healthcare concepts/fields.",
    },
    {
        "type_name": "MILITARY",
        "definition": "Named military units, organizations, equipment, weapons, ranks, bases, conflicts/wars, and defense-related concepts.",
    },
    {
        "type_name": "OCCUPATION",
        "definition": "Named jobs, professions, titles, ranks, and work-related roles.",
    },
    {
        "type_name": "ORGANISM",
        "definition": "Named individual living beings (plants, animals, fungi, microorganisms), species, breeds, and higher biological taxa.",
    },
    {
        "type_name": "ORGANIZATION",
        "definition": "Named companies, institutions (educational, governmental, religious), agencies, political parties, associations, non-profits, teams, and other formal groups.",
    },
    {
        "type_name": "PERSON",
        "definition": "Named individual human beings (real or fictional), including their full names, titles, or aliases.",
    },
    {
        "type_name": "PERSONAL_CONTACT",
        "definition": "Contact information (email addresses, phone numbers), URLs, user IDs, and personal identifiers (Note: Data-like type).",
    },
    {
        "type_name": "POLITICAL",
        "definition": "Named political concepts, ideologies, systems, movements, parties (can also be ORG), and governance elements.",
    },
    {
        "type_name": "PRODUCT",
        "definition": "Named commercially available products, goods, services, brands, models, and manufactured items (including hardware, vehicles, but excluding software, creative works, food).",
    },
    {
        "type_name": "QUANTITY",
        "definition": "Measurements, numerical values with units, percentages, scores, and quantitative expressions (Note: Data-like type).",
    },
    {
        "type_name": "RELIGION",
        "definition": "Named religions, religious beliefs, practices, deities, figures, denominations, organizations (can also be ORG), sacred texts, and sacred places.",
    },
    {
        "type_name": "SCIENTIFIC",
        "definition": "Named scientific concepts, theories, laws, principles, methods, fields (excluding purely Biological, Medical, Mathematical covered elsewhere), and units of measure names.",
    },
    {
        "type_name": "SOCIAL_AND_BEHAVIORAL",
        "definition": "Named social behaviors, psychological concepts/conditions (non-medical), sociological theories, named social groups (distinct from DEMOGRAPHIC), and behavioral patterns.",
    },
    {
        "type_name": "SOFTWARE",
        "definition": "Named software applications, programs, operating systems, programming languages, frameworks, algorithms, websites, and digital systems/formats.",
    },
    {
        "type_name": "TIME",
        "definition": "Specific times of day, durations, frequencies, and recurring temporal expressions (e.g., 'daily', '9 AM').",
    },
]


# Helper functions
def get_entity_type_names() -> List[str]:
    """Return a list of all entity type names."""
    return [et["type_name"].strip().upper() for et in ENTITY_TYPES]


def get_entity_type_definition(type_name: str) -> str:
    """Return the definition for a specific entity type."""
    for et in ENTITY_TYPES:
        if et["type_name"] == type_name:
            return et["definition"]
    return ""


def format_entity_types_for_prompt() -> str:
    """Format entity types as a JSON string for inclusion in prompts."""
    import json

    return json.dumps(ENTITY_TYPES, indent=2)

class RelationTypeDefinition(TypedDict):
    type_name: str
    definition: str
    
RELATION_TYPES: List[RelationTypeDefinition] = [
    # We reccomend to include verb/action in relationship name
    # Additionally, the definition should be clear and include the possible entity types
    # the model should join
        # A definition "bootstrapped" or coming out of an LLM can prove more effective
       
    # Example relations for testing on Skincare data  
    {
        "type_name": "produces",
        "definition": "Links a person to a pruduct he/she makes.",
    },
    {
        "type_name": "is_ingridient_of",
        "definition": "Indicates a specific active ingredient present in a product.",
    },
    {
        "type_name": "has_shade",
        "definition": "Connects a product to a specific color variation.",
    },
]

# Helper functions
def get_relation_type_names() -> List[str]:
    """Return a list of all relation type names."""
    return [rt["type_name"].strip().upper() for rt in RELATION_TYPES]


def get_relation_type_definition(type_name: str) -> str:
    """Return the definition for a specific relation type."""
    for rt in RELATION_TYPES:
        if rt["type_name"] == type_name:
            return rt["definition"]
    return ""


def format_relation_types_for_prompt() -> str:
    """Format relation types as a JSON string for inclusion in prompts."""
    import json

    return json.dumps(RELATION_TYPES, indent=2)


class EntityType(BaseModel):
    """Model for hierarchical entity type classification."""

    BroadType: Optional[str] = Field(
        ...,
        description="The general type of the entity (e.g., person, organization, location)",
    )
    SpecificType: Optional[str] = Field(
        None,
        description="The specific type of the entity (e.g., politician, company, city)",
    )
    FineType: Optional[str] = Field(
        None,
        description="The sub-specific type of the entity (e.g., president, tech_company, capital_city)",
    )

    @field_validator("BroadType")
    @classmethod
    def validate_broad_type(cls, v):
        valid_types = get_entity_type_names()
        if v.strip().upper() not in valid_types:
            raise ValueError(f"Invalid broad type: {v}")
        return v


class Entity(BaseModel):
    """Model for a single named entity with its hierarchical type classification."""

    NamedEntity: str = Field(..., description="The actual text/name of the entity")
    NamedEntityDescription: Optional[str] = Field(
        ...,
        description="A contextual description of the detected named entity in the context in which it was detected",
    )
    CanonicalNamedEntity: str = Field(
        ...,
        description="The canonical name of the entity",
    )
    isNamedEntity: bool = Field(
        ...,
        description="Indicates whether the recognized entity is actually a named entity",
    )
    NamedEntityType: EntityType = Field(
        ..., description="The hierarchical type classification of the entity"
    )


class EntityResponse(BaseModel):
    """Model for the complete entity recognition response."""

    output: List[Entity] = Field(
        ...,
        description="List of identified entities with their hierarchical classifications",
    )

    @classmethod
    def from_json(cls, json_data: dict) -> "EntityResponse":
        """Create an EntityResponse instance from JSON data."""
        return cls.model_validate(json_data)

    def to_json(self) -> dict:
        """Convert the response to a JSON-compatible dictionary."""
        return self.model_dump()


class Relation(BaseModel):
    """Model for a single relation between entities."""

    Subject: str = Field(..., description="The subject entity name")
    Object: str = Field(..., description="The object entity name")
    SubjectType: Optional[EntityType] = Field(
        None, description="The type of the subject entity"
    )
    SubjectDescription: Optional[str] = Field(
        None, description="The description of the subject entity"
    )
    ObjectType: Optional[EntityType] = Field(
        None, description="The type of the object entity"
    )
    ObjectDescription: Optional[str] = Field(
        None, description="The description of the object entity"
    )
    RelationExists: bool | None = Field(
        ...,
        description="Indicates whether the relationship exists or not",
    )
    RelationLabel: str | None = Field(
        ..., description="The relationship between the entities"
    )
    Evidence: str | None = Field(
        ..., description="The text evidence supporting this relationship"
    )


class RelationResponse(BaseModel):
    """Model for the complete relation extraction response."""

    output: List[Relation] = Field(..., description="List of identified relations")

    @classmethod
    def from_json(cls, json_data: dict) -> "RelationResponse":
        """Create a RelationResponse instance from JSON data."""
        return cls.model_validate(json_data)

    def to_json(self) -> dict:
        """Convert the response to a JSON-compatible dictionary."""
        return self.model_dump()

if __name__ == "__main__":
    print(len(get_entity_type_names()))
    print(get_entity_type_definition("ACADEMIC"))
    print(format_entity_types_for_prompt())
    print(get_relation_type_names())
    print(get_relation_type_definition("produces"))
    print(format_relation_types_for_prompt())
    print(EntityType.model_json_schema())
    print(EntityResponse.model_json_schema())
    print(RelationResponse.model_json_schema())