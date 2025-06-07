mport os
import re
import json
import logging
import string
import csv
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Union, Optional, Tuple, Set, Callable
from enum import Enum
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optional dependencies with fallbacks
try:
    import spacy
    from spacy.tokens import Doc, Span, Token
    from spacy.matcher import Matcher, PhraseMatcher
    from spacy.language import Language
except ImportError:
    logger.warning("spaCy not installed. Core NER functionality will be limited.")
    spacy = None

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
except ImportError:
    logger.warning("NLTK not installed. Some NER functionality will be limited.")
    nltk = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    logger.warning("Transformers not installed. Advanced NER models will be unavailable.")
    pipeline = None

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
except ImportError:
    logger.warning("scikit-learn not installed. ML-based extraction will be unavailable.")
    np = None

try:
    import pandas as pd
except ImportError:
    logger.warning("Pandas not installed. Export functionality will be limited.")
    pd = None

try:
    import networkx as nx
except ImportError:
    logger.warning("NetworkX not installed. Relationship visualization will be limited.")
    nx = None


class EntityType(Enum):
    """Standard entity types"""
    PERSON = "PERSON"
    ORGANIZATION = "ORG"
    LOCATION = "LOC"
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    FACILITY = "FAC"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    CUSTOM = "CUSTOM"
    
    @classmethod
    def from_spacy(cls, spacy_label: str) -> 'EntityType':
        """Convert spaCy entity label to EntityType"""
        mapping = {
            "PERSON": cls.PERSON,
            "ORG": cls.ORGANIZATION,
            "GPE": cls.LOCATION,
            "LOC": cls.LOCATION,
            "DATE": cls.DATE,
            "TIME": cls.TIME,
            "MONEY": cls.MONEY,
            "PERCENT": cls.PERCENT,
            "FAC": cls.FACILITY,
            "PRODUCT": cls.PRODUCT,
            "EVENT": cls.EVENT,
            "WORK_OF_ART": cls.WORK_OF_ART,
            "LAW": cls.LAW,
            "LANGUAGE": cls.LANGUAGE,
        }
        return mapping.get(spacy_label, cls.CUSTOM)
    
    @classmethod
    def to_spacy(cls, entity_type: 'EntityType') -> str:
        """Convert EntityType to spaCy entity label"""
        mapping = {
            cls.PERSON: "PERSON",
            cls.ORGANIZATION: "ORG",
            cls.LOCATION: "LOC",
            cls.DATE: "DATE",
            cls.TIME: "TIME",
            cls.MONEY: "MONEY",
            cls.PERCENT: "PERCENT",
            cls.FACILITY: "FAC",
            cls.PRODUCT: "PRODUCT",
            cls.EVENT: "EVENT",
            cls.WORK_OF_ART: "WORK_OF_ART",
            cls.LAW: "LAW",
            cls.LANGUAGE: "LANGUAGE",
            cls.CUSTOM: "CUSTOM",
        }
        return mapping.get(entity_type, "CUSTOM")


@dataclass
class Entity:
    """Class for extracted entities"""
    text: str
    type: EntityType
    start_char: int
    end_char: int
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            "text": self.text,
            "type": self.type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """Create entity from dictionary"""
        return cls(
            text=data["text"],
            type=EntityType(data["type"]),
            start_char=data["start_char"],
            end_char=data["end_char"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )
    
    @classmethod
    def from_spacy(cls, ent: 'Span') -> 'Entity':
        """Create entity from spaCy entity span"""
        return cls(
            text=ent.text,
            type=EntityType.from_spacy(ent.label_),
            start_char=ent.start_char,
            end_char=ent.end_char,
            confidence=getattr(ent, "_.confidence", 1.0),
            metadata={"label_": ent.label_}
        )


@dataclass
class EntityRelationship:
    """Class for entity relationships"""
    source: Entity
    target: Entity
    relation_type: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relationship to dictionary"""
        return {
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityRelationship':
        """Create relationship from dictionary"""
        return cls(
            source=Entity.from_dict(data["source"]),
            target=Entity.from_dict(data["target"]),
            relation_type=data["relation_type"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )


@dataclass
class ExtractionResult:
    """Class for entity extraction results"""
    text: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[EntityRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert extraction result to dictionary"""
        return {
            "text": self.text,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert extraction result to JSON string"""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create extraction result from dictionary"""
        return cls(
            text=data["text"],
            entities=[Entity.from_dict(e) for e in data.get("entities", [])],
            relationships=[EntityRelationship.from_dict(r) for r in data.get("relationships", [])],
            metadata=data.get("metadata", {})
        )
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities of a specific type"""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_entity_texts_by_type(self, entity_type: EntityType) -> List[str]:
        """Get entity texts of a specific type"""
        return [e.text for e in self.entities if e.type == entity_type]
    
    def get_entity_counts(self) -> Dict[EntityType, int]:
        """Get counts of entities by type"""
        counts = defaultdict(int)
        for entity in self.entities:
            counts[entity.type] += 1
        return dict(counts)
    
    def merge_with(self, other: 'ExtractionResult') -> 'ExtractionResult':
        """Merge with another extraction result"""
        if self.text != other.text:
            raise ValueError("Cannot merge extraction results with different texts")
        
        # Create a new result with merged entities and relationships
        merged = ExtractionResult(
            text=self.text,
            entities=list(self.entities),
            relationships=list(self.relationships),
            metadata=dict(self.metadata)
        )
        
        # Add entities from other result if they don't overlap with existing entities
        existing_spans = [(e.start_char, e.end_char) for e in self.entities]
        for entity in other.entities:
            if not any(start <= entity.start_char < end or start < entity.end_char <= end 
                      for start, end in existing_spans):
                merged.entities.append(entity)
        
        # Add relationships from other result
        merged.relationships.extend(other.relationships)
        
        # Merge metadata
        merged.metadata.update(other.metadata)
        
        return merged


class EntityExtractor(ABC):
    """Abstract base class for entity extractors"""
    
    @abstractmethod
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities from text"""
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by this extractor"""
        pass


class SpacyEntityExtractor(EntityExtractor):
    """Entity extractor using spaCy"""
    
    def __init__(self, model: str = "en_core_web_sm", disable: List[str] = None):
        """Initialize spaCy entity extractor"""
        if spacy is None:
            raise ImportError("spaCy is required for SpacyEntityExtractor")
        
        self.model_name = model
        self.disable = disable or []
        
        try:
            self.nlp = spacy.load(model, disable=self.disable)
        except:
            logger.warning(f"spaCy model {model} not found. You may need to download it with: python -m spacy download {model}")
            self.nlp = None
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using spaCy"""
        if self.nlp is None:
            return ExtractionResult(text=text, metadata={"error": "spaCy model not available"})
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = [Entity.from_spacy(ent) for ent in doc.ents]
        
        # Create extraction result
        result = ExtractionResult(
            text=text,
            entities=entities,
            metadata={"model": self.model_name}
        )
        
        return result
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by spaCy model"""
        if self.nlp is None:
            return []
        
        # Get entity types from spaCy pipeline
        entity_types = set()
        if self.nlp.has_pipe("ner"):
            ner = self.nlp.get_pipe("ner")
            for label in ner.labels:
                try:
                    entity_types.add(EntityType.from_spacy(label))
                except:
                    pass
        
        return list(entity_types)


class NLTKEntityExtractor(EntityExtractor):
    """Entity extractor using NLTK"""
    
    def __init__(self):
        """Initialize NLTK entity extractor"""
        if nltk is None:
            raise ImportError("NLTK is required for NLTKEntityExtractor")
        
        # Download required NLTK resources if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using NLTK"""
        # Tokenize, POS tag, and chunk
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        ne_tree = ne_chunk(pos_tags)
        
        # Extract named entities
        entities = []
        for i, chunk in enumerate(ne_tree):
            if hasattr(chunk, 'label'):
                # This is a named entity
                entity_text = ' '.join(c[0] for c in chunk)
                
                # Calculate character offsets (approximate)
                start_char = text.find(entity_text)
                if start_char == -1:
                    # If exact match not found, try case-insensitive
                    start_char = text.lower().find(entity_text.lower())
                
                if start_char != -1:
                    end_char = start_char + len(entity_text)
                    
                    # Map NLTK entity type to our EntityType
                    entity_type = self._map_nltk_entity_type(chunk.label())
                    
                    entities.append(Entity(
                        text=entity_text,
                        type=entity_type,
                        start_char=start_char,
                        end_char=end_char,
                        confidence=0.8,  # NLTK doesn't provide confidence scores
                        metadata={"nltk_label": chunk.label()}
                    ))
        
        # Create extraction result
        result = ExtractionResult(
            text=text,
            entities=entities,
            metadata={"extractor": "nltk"}
        )
        
        return result
    
    def _map_nltk_entity_type(self, nltk_label: str) -> EntityType:
        """Map NLTK entity label to EntityType"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,  # Geo-Political Entity
            "LOCATION": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "FACILITY": EntityType.FACILITY,
        }
        return mapping.get(nltk_label, EntityType.CUSTOM)
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by NLTK"""
        return [
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.DATE,
            EntityType.TIME,
            EntityType.MONEY,
            EntityType.PERCENT,
            EntityType.FACILITY,
        ]


class TransformerEntityExtractor(EntityExtractor):
    """Entity extractor using Hugging Face Transformers"""
    
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        """Initialize transformer entity extractor"""
        if pipeline is None:
            raise ImportError("Transformers is required for TransformerEntityExtractor")
        
        self.model_name = model_name
        
        try:
            self.ner_pipeline = pipeline("ner", model=model_name, aggregation_strategy="simple")
        except:
            logger.warning(f"Failed to load transformer model {model_name}")
            self.ner_pipeline = None
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using transformer model"""
        if self.ner_pipeline is None:
            return ExtractionResult(text=text, metadata={"error": "Transformer model not available"})
        
        try:
            # Extract entities using transformer pipeline
            ner_results = self.ner_pipeline(text)
            
            # Convert to our Entity format
            entities = []
            for result in ner_results:
                entity_type = self._map_transformer_entity_type(result["entity_group"])
                
                entities.append(Entity(
                    text=result["word"],
                    type=entity_type,
                    start_char=result["start"],
                    end_char=result["end"],
                    confidence=result["score"],
                    metadata={"label": result["entity_group"]}
                ))
            
            # Create extraction result
            result = ExtractionResult(
                text=text,
                entities=entities,
                metadata={"model": self.model_name}
            )
            
            return result
        
        except Exception as e:
            logger.error(f"Error in transformer extraction: {str(e)}")
            return ExtractionResult(text=text, metadata={"error": str(e)})
    
    def _map_transformer_entity_type(self, transformer_label: str) -> EntityType:
        """Map transformer entity label to EntityType"""
        # Common mapping for CoNLL-2003 format used by many transformer models
        mapping = {
            "PER": EntityType.PERSON,
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "LOC": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "GPE": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.TIME,
            "MONEY": EntityType.MONEY,
            "PERCENT": EntityType.PERCENT,
            "FAC": EntityType.FACILITY,
            "FACILITY": EntityType.FACILITY,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.WORK_OF_ART,
            "LAW": EntityType.LAW,
            "LANGUAGE": EntityType.LANGUAGE,
            "MISC": EntityType.CUSTOM,
        }
        return mapping.get(transformer_label, EntityType.CUSTOM)
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by transformer model"""
        # This depends on the specific model, but most NER models support these types
        return [
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.CUSTOM,  # For MISC
        ]


class RegexEntityExtractor(EntityExtractor):
    """Entity extractor using regular expressions"""
    
    def __init__(self, patterns: Dict[str, List[str]] = None):
        """Initialize regex entity extractor"""
        self.patterns = patterns or {}
        self.compiled_patterns = {}
        
        # Compile patterns
        for entity_type, pattern_list in self.patterns.items():
            self.compiled_patterns[entity_type] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
    
    def add_pattern(self, entity_type: str, pattern: str) -> None:
        """Add a regex pattern for an entity type"""
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
            self.compiled_patterns[entity_type] = []
        
        self.patterns[entity_type].append(pattern)
        self.compiled_patterns[entity_type].append(re.compile(pattern, re.IGNORECASE))
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using regex patterns"""
        entities = []
        
        # Apply each pattern
        for entity_type, compiled_patterns in self.compiled_patterns.items():
            for pattern in compiled_patterns:
                for match in pattern.finditer(text):
                    # Create entity
                    try:
                        entity_type_enum = EntityType(entity_type)
                    except ValueError:
                        entity_type_enum = EntityType.CUSTOM
                    
                    entities.append(Entity(
                        text=match.group(),
                        type=entity_type_enum,
                        start_char=match.start(),
                        end_char=match.end(),
                        confidence=1.0,  # Regex matches are deterministic
                        metadata={"pattern": pattern.pattern}
                    ))
        
        # Create extraction result
        result = ExtractionResult(
            text=text,
            entities=entities,
            metadata={"extractor": "regex"}
        )
        
        return result
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by regex patterns"""
        supported_types = []
        for entity_type in self.patterns.keys():
            try:
                supported_types.append(EntityType(entity_type))
            except ValueError:
                # Custom entity type
                supported_types.append(EntityType.CUSTOM)
        
        return supported_types


class DictionaryEntityExtractor(EntityExtractor):
    """Entity extractor using dictionaries/gazetteers"""
    
    def __init__(self, dictionaries: Dict[str, List[str]] = None):
        """Initialize dictionary entity extractor"""
        self.dictionaries = dictionaries or {}
        self.matchers = {}
        
        # Initialize spaCy for efficient dictionary matching
        if spacy is not None:
            try:
                self.nlp = spacy.blank("en")
                
                # Create matchers for each entity type
                for entity_type, terms in self.dictionaries.items():
                    self._create_matcher(entity_type, terms)
            except:
                logger.warning("Failed to initialize spaCy for dictionary matching")
                self.nlp = None
        else:
            self.nlp = None
    
    def _create_matcher(self, entity_type: str, terms: List[str]) -> None:
        """Create a phrase matcher for an entity type"""
        if self.nlp is None:
            return
        
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(term) for term in terms]
        matcher.add(entity_type, patterns)
        self.matchers[entity_type] = matcher
    
    def add_terms(self, entity_type: str, terms: List[str]) -> None:
        """Add terms to a dictionary"""
        if entity_type not in self.dictionaries:
            self.dictionaries[entity_type] = []
        
        self.dictionaries[entity_type].extend(terms)
        
        # Update matcher
        if self.nlp is not None:
            self._create_matcher(entity_type, self.dictionaries[entity_type])
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using dictionaries"""
        entities = []
        
        if self.nlp is not None:
            # Use spaCy for efficient dictionary matching
            doc = self.nlp(text)
            
            for entity_type, matcher in self.matchers.items():
                matches = matcher(doc)
                
                for match_id, start, end in matches:
                    span = doc[start:end]
                    
                    try:
                        entity_type_enum = EntityType(entity_type)
                    except ValueError:
                        entity_type_enum = EntityType.CUSTOM
                    
                    entities.append(Entity(
                        text=span.text,
                        type=entity_type_enum,
                        start_char=span.start_char,
                        end_char=span.end_char,
                        confidence=1.0,  # Dictionary matches are deterministic
                        metadata={"dictionary": entity_type}
                    ))
        else:
            # Fallback to simple string matching
            for entity_type, terms in self.dictionaries.items():
                for term in terms:
                    # Find all occurrences of the term
                    start = 0
                    while True:
                        start = text.lower().find(term.lower(), start)
                        if start == -1:
                            break
                        
                        try:
                            entity_type_enum = EntityType(entity_type)
                        except ValueError:
                            entity_type_enum = EntityType.CUSTOM
                        
                        entities.append(Entity(
                            text=text[start:start+len(term)],
                            type=entity_type_enum,
                            start_char=start,
                            end_char=start+len(term),
                            confidence=1.0,
                            metadata={"dictionary": entity_type}
                        ))
                        
                        start += len(term)
        
        # Create extraction result
        result = ExtractionResult(
            text=text,
            entities=entities,
            metadata={"extractor": "dictionary"}
        )
        
        return result
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by dictionaries"""
        supported_types = []
        for entity_type in self.dictionaries.keys():
            try:
                supported_types.append(EntityType(entity_type))
            except ValueError:
                # Custom entity type
                supported_types.append(EntityType.CUSTOM)
        
        return supported_types


class MLEntityExtractor(EntityExtractor):
    """Entity extractor using machine learning"""
    
    def __init__(self, model_type: str = "random_forest"):
        """Initialize ML entity extractor"""
        if np is None:
            raise ImportError("scikit-learn is required for MLEntityExtractor")
        
        self.model_type = model_type
        self.models = {}
        self.vectorizers = {}
        self.entity_types = set()
    
    def train(self, texts: List[str], annotations: List[List[Dict[str, Any]]]) -> None:
        """Train ML models for entity extraction
        
        Args:
            texts: List of training texts
            annotations: List of entity annotations for each text
                Each annotation is a dict with keys: text, type, start_char, end_char
        """
        # Extract features and labels for each entity type
        entity_examples = defaultdict(list)
        non_entity_examples = defaultdict(list)
        
        for text, text_annotations in zip(texts, annotations):
            # Extract entity spans
            entity_spans = []
            for annotation in text_annotations:
                entity_spans.append((annotation["start_char"], annotation["end_char"]))
                self.entity_types.add(annotation["type"])
            
            # Tokenize text (simple whitespace tokenization for demonstration)
            tokens = text.split()
            token_spans = []
            start = 0
            for token in tokens:
                token_start = text.find(token, start)
                token_end = token_start + len(token)
                token_spans.append((token_start, token_end, token))
                start = token_end
            
            # Create examples for each token
            for token_start, token_end, token in token_spans:
                # Check if token is part of an entity
                is_entity = False
                entity_type = None
                
                for annotation in text_annotations:
                    if (token_start >= annotation["start_char"] and token_end <= annotation["end_char"]):
                        is_entity = True
                        entity_type = annotation["type"]
                        break
                
                # Create features for token
                features = self._extract_features(text, token, token_start, token_end)
                
                if is_entity:
                    entity_examples[entity_type].append(features)
                else:
                    # Add to non-entity examples for each entity type
                    for entity_type in self.entity_types:
                        non_entity_examples[entity_type].append(features)
        
        # Train models for each entity type
        for entity_type in self.entity_types:
            # Prepare training data
            X = entity_examples[entity_type] + non_entity_examples[entity_type]
            y = [1] * len(entity_examples[entity_type]) + [0] * len(non_entity_examples[entity_type])
            
            # Create vectorizer
            vectorizer = TfidfVectorizer()
            X_vec = vectorizer.fit_transform([" ".join(x) for x in X])
            
            # Create and train model
            if self.model_type == "random_forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = LogisticRegression(random_state=42)
            
            model.fit(X_vec, y)
            
            # Save model and vectorizer
            self.models[entity_type] = model
            self.vectorizers[entity_type] = vectorizer
    
    def _extract_features(self, text: str, token: str, token_start: int, token_end: int) -> List[str]:
        """Extract features for a token"""
        features = []
        
        # Token itself
        features.append(f"TOKEN={token}")
        
        # Token lowercase
        features.append(f"TOKEN_LOWER={token.lower()}")
        
        # Token shape (e.g., Xxx, XXX, 123)
        shape = ""
        for char in token:
            if char.isupper():
                shape += "X"
            elif char.islower():
                shape += "x"
            elif char.isdigit():
                shape += "d"
            else:
                shape += char
        features.append(f"SHAPE={shape}")
        
        # Is title case
        features.append(f"IS_TITLE={token.istitle()}")
        
        # Is all uppercase
        features.append(f"IS_UPPER={token.isupper()}")
        
        # Is all lowercase
        features.append(f"IS_LOWER={token.islower()}")
        
        # Is digit
        features.append(f"IS_DIGIT={token.isdigit()}")
        
        # Length
        features.append(f"LENGTH={len(token)}")
        
        # Position in text
        features.append(f"POSITION={token_start / len(text)}")
        
        return features
    
    def extract(self, text: str) -> ExtractionResult:
        """Extract entities using ML models"""
        if not self.models:
            return ExtractionResult(text=text, metadata={"error": "ML models not trained"})
        
        entities = []
        
        # Tokenize text (simple whitespace tokenization for demonstration)
        tokens = text.split()
        token_spans = []
        start = 0
        for token in tokens:
            token_start = text.find(token, start)
            token_end = token_start + len(token)
            token_spans.append((token_start, token_end, token))
            start = token_end
        
        # Predict entities for each token
        for token_start, token_end, token in token_spans:
            # Extract features
            features = self._extract_features(text, token, token_start, token_end)
            
            # Predict for each entity type
            for entity_type in self.models:
                # Vectorize features
                X_vec = self.vectorizers[entity_type].transform([" ".join(features)])
                
                # Predict
                prediction = self.models[entity_type].predict(X_vec)[0]
                
                if prediction == 1:
                    # Token is predicted as entity
                    try:
                        entity_type_enum = EntityType(entity_type)
                    except ValueError:
                        entity_type_enum = EntityType.CUSTOM
                    
                    # Get prediction probability
                    prob = self.models[entity_type].predict_proba(X_vec)[0][1]
                    
                    entities.append(Entity(
                        text=token,
                        type=entity_type_enum,
                        start_char=token_start,
                        end_char=token_end,
                        confidence=prob,
                        metadata={"ml_model": self.model_type}
                    ))
        
        # Create extraction result
        result = ExtractionResult(
            text=text,
            entities=entities,
            metadata={"extractor": "ml", "model_type": self.model_type}
        )
        
        return result
    
    def save(self, path: str) -> None:
        """Save ML models and vectorizers"""
        data = {
            "models": self.models,
            "vectorizers": self.vectorizers,
            "entity_types": list(self.entity_types),
            "model_type": self.model_type
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, path: str) -> None:
        """Load ML models and vectorizers"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.models = data["models"]
        self.vectorizers = data["vectorizers"]
        self.entity_types = set(data["entity_types"])
        self.model_type = data["model_type"]
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get entity types supported by ML models"""
        supported_types = []
        for entity_type in self.entity_types:
            try:
                supported_types.append(EntityType(entity_type))
            except ValueError:
                # Custom entity type
                supported_types.append(EntityType.CUSTOM)
        
        return supported_types


class EntityLinker:
    """Entity linking and resolution"""
    
    def __init__(self, knowledge_base: Dict[str, Dict[str, Any]] = None):
        """Initialize entity linker"""
        self.knowledge_base = knowledge_base or {}
    
    def add_entity(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """Add entity to knowledge base"""
        self.knowledge_base[entity_id] = entity_data
    
    def link_entities(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """Link extracted entities to knowledge base"""
        if not self.knowledge_base:
            return extraction_result
        
        # Create a new result with linked entities
        result = ExtractionResult(
            text=extraction_result.text,
            entities=list(extraction_result.entities),
            relationships=list(extraction_result.relationships),
            metadata=dict(extraction_result.metadata)
        )
        
        # Link entities
        for i, entity in enumerate(result.entities):
            # Look for matches in knowledge base
            matches = []
            for entity_id, entity_data in self.knowledge_base.items():
                # Check for exact match
                if entity.text.lower() == entity_data.get("name", "").lower():
                    matches.append((entity_id, entity_data, 1.0))
                # Check for partial match
                elif entity.text.lower() in entity_data.get("aliases", []):
                    matches.append((entity_id, entity_data, 0.9))
                # Check for fuzzy match
                elif self._fuzzy_match(entity.text, entity_data.get("name", "")):
                    matches.append((entity_id, entity_data, 0.7))
            
            # Sort matches by score
            matches.sort(key=lambda x: x[2], reverse=True)
            
            # Update entity metadata with link information
            if matches:
                entity_id, entity_data, score = matches[0]
                result.entities[i].metadata["linked_entity_id"] = entity_id
                result.entities[i].metadata["linked_entity_data"] = entity_data
                result.entities[i].metadata["link_score"] = score
        
        return result
    
    def _fuzzy_match(self, text1: str, text2: str) -> bool:
        """Simple fuzzy matching"""
        # Convert to lowercase
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Check if one is contained in the other
        if text1 in text2 or text2 in text1:
            return True
        
        # Check for high character overlap
        chars1 = set(text1)
        chars2 = set(text2)
        overlap = len(chars1.intersection(chars2)) / max(len(chars1), len(chars2))
        
        return overlap > 0.8


class RelationshipExtractor:
    """Extract relationships between entities"""
    
    def __init__(self, relation_patterns: Dict[str, List[str]] = None):
        """Initialize relationship extractor"""
        self.relation_patterns = relation_patterns or {}
        self.compiled_patterns = {}
        
        # Compile patterns
        for relation_type, pattern_list in self.relation_patterns.items():
            self.compiled_patterns[relation_type] = [re.compile(pattern, re.IGNORECASE) for pattern in pattern_list]
    
    def add_pattern(self, relation_type: str, pattern: str) -> None:
        """Add a regex pattern for a relationship type"""
        if relation_type not in self.relation_patterns:
            self.relation_patterns[relation_type] = []
            self.compiled_patterns[relation_type] = []
        
        self.relation_patterns[relation_type].append(pattern)
        self.compiled_patterns[relation_type].append(re.compile(pattern, re.IGNORECASE))
    
    def extract_relationships(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """Extract relationships between entities"""
        # Create a new result with relationships
        result = ExtractionResult(
            text=extraction_result.text,
            entities=list(extraction_result.entities),
            relationships=list(extraction_result.relationships),
            metadata=dict(extraction_result.metadata)
        )
        
        # Extract relationships using patterns
        for relation_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(extraction_result.text):
                    # Find entities that overlap with the match
                    match_start = match.start()
                    match_end = match.end()
                    
                    # Find entities before and after the match
                    entities_before = [e for e in extraction_result.entities if e.end_char <= match_start]
                    entities_after = [e for e in extraction_result.entities if e.start_char >= match_end]
                    
                    # Find closest entities
                    if entities_before and entities_after:
                        source = max(entities_before, key=lambda e: e.end_char)
                        target = min(entities_after, key=lambda e: e.start_char)
                        
                        # Create relationship
                        relationship = EntityRelationship(
                            source=source,
                            target=target,
                            relation_type=relation_type,
                            confidence=0.8,
                            metadata={"pattern": pattern.pattern, "match_text": match.group()}
                        )
                        
                        result.relationships.append(relationship)
        
        # Extract co-occurrence relationships
        self._extract_cooccurrence_relationships(result)
        
        return result
    
    def _extract_cooccurrence_relationships(self, result: ExtractionResult) -> None:
        """Extract co-occurrence relationships between entities"""
        # Group entities by sentence (simple approximation)
        sentences = result.text.split('.')
        sentence_spans = []
        start = 0
        for sentence in sentences:
            if sentence:
                end = start + len(sentence) + 1  # +1 for the period
                sentence_spans.append((start, end))
                start = end
        
        # Find entities in each sentence
        for start, end in sentence_spans:
            sentence_entities = [e for e in result.entities if e.start_char >= start and e.end_char <= end]
            
            # Create co-occurrence relationships
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    # Skip if entities are of the same type
                    if entity1.type == entity2.type:
                        continue
                    
                    # Create relationship
                    relationship = EntityRelationship(
                        source=entity1,
                        target=entity2,
                        relation_type="co-occurs-with",
                        confidence=0.6,
                        metadata={"relationship_type": "co-occurrence"}
                    )
                    
                    result.relationships.append(relationship)


class EntityVisualizer:
    """Visualize extracted entities and relationships"""
    
    def __init__(self):
        """Initialize entity visualizer"""
        pass
    
    def highlight_entities(self, extraction_result: ExtractionResult) -> str:
        """Highlight entities in text"""
        text = extraction_result.text
        
        # Sort entities by start position (reversed to avoid index issues)
        sorted_entities = sorted(extraction_result.entities, key=lambda e: e.start_char, reverse=True)
        
        # Highlight each entity
        for entity in sorted_entities:
            highlight = f"[{entity.text}]({entity.type.value})"
            text = text[:entity.start_char] + highlight + text[entity.end_char:]
        
        return text
    
    def to_html(self, extraction_result: ExtractionResult) -> str:
        """Convert extraction result to HTML with highlighted entities"""
        text = extraction_result.text
        
        # Sort entities by start position (reversed to avoid index issues)
        sorted_entities = sorted(extraction_result.entities, key=lambda e: e.start_char, reverse=True)
        
        # Create a color map for entity types
        color_map = {
            EntityType.PERSON: "#ffcccc",
            EntityType.ORGANIZATION: "#ccffcc",
            EntityType.LOCATION: "#ccccff",
            EntityType.DATE: "#ffffcc",
            EntityType.TIME: "#ffccff",
            EntityType.MONEY: "#ccffff",
            EntityType.PERCENT: "#ffccaa",
            EntityType.FACILITY: "#aaccff",
            EntityType.PRODUCT: "#ffaacc",
            EntityType.EVENT: "#aaffcc",
            EntityType.WORK_OF_ART: "#ccaaff",
            EntityType.LAW: "#ffaaaa",
            EntityType.LANGUAGE: "#aaffaa",
            EntityType.CUSTOM: "#aaaaff",
        }
        
        # Highlight each entity
        for entity in sorted_entities:
            color = color_map.get(entity.type, "#cccccc")
            highlight = f'<span style="background-color: {color};" title="{entity.type.value}">{entity.text}</span>'
            text = text[:entity.start_char] + highlight + text[entity.end_char:]
        
        # Wrap in HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Entity Extraction Result</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .legend {{ margin-top: 20px; }}
                .legend-item {{ display: inline-block; margin-right: 15px; }}
                .legend-color {{ display: inline-block; width: 20px; height: 20px; margin-right: 5px; vertical-align: middle; }}
            </style>
        </head>
        <body>
            <div>{text}</div>
            <div class="legend">
                <h3>Entity Types</h3>
                {"".join([f'<div class="legend-item"><span class="legend-color" style="background-color: {color};"></span>{entity_type.value}</div>' for entity_type, color in color_map.items() if entity_type in {e.type for e in extraction_result.entities}])}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def to_graph(self, extraction_result: ExtractionResult) -> Optional[Any]:
        """Convert extraction result to a graph visualization"""
        if nx is None:
            return None
        
        # Create graph
        G = nx.Graph()
        
        # Add entities as nodes
        for entity in extraction_result.entities:
            G.add_node(entity.text, type=entity.type.value, confidence=entity.confidence)
        
        # Add relationships as edges
        for relationship in extraction_result.relationships:
            G.add_edge(
                relationship.source.text,
                relationship.target.text,
                type=relationship.relation_type,
                confidence=relationship.confidence
            )
        
        return G


class EntityExporter:
    """Export extraction results to various formats"""
    
    def __init__(self):
        """Initialize entity exporter"""
        pass
    
    def to_json(self, extraction_result: ExtractionResult) -> str:
        """Export extraction result to JSON"""
        return extraction_result.to_json()
    
    def to_csv(self, extraction_result: ExtractionResult) -> str:
        """Export entities to CSV"""
        if not extraction_result.entities:
            return ""
        
        # Create CSV string
        csv_str = "text,type,start_char,end_char,confidence\n"
        
        for entity in extraction_result.entities:
            csv_str += f'"{entity.text}",{entity.type.value},{entity.start_char},{entity.end_char},{entity.confidence}\n'
        
        return csv_str
    
    def to_dataframe(self, extraction_result: ExtractionResult) -> Optional[Any]:
        """Export entities to pandas DataFrame"""
        if pd is None:
            return None
        
        if not extraction_result.entities:
            return pd.DataFrame()
        
        # Create DataFrame
        data = []
        for entity in extraction_result.entities:
            data.append({
                "text": entity.text,
                "type": entity.type.value,
                "start_char": entity.start_char,
                "end_char": entity.end_char,
                "confidence": entity.confidence
            })
        
        return pd.DataFrame(data)
    
    def save_to_file(self, extraction_result: ExtractionResult, file_path: str, format: str = "json") -> None:
        """Save extraction result to file"""
        format = format.lower()
        
        if format == "json":
            with open(file_path, 'w') as f:
                f.write(self.to_json(extraction_result))
        
        elif format == "csv":
            with open(file_path, 'w') as f:
                f.write(self.to_csv(extraction_result))
        
        elif format == "html":
            visualizer = EntityVisualizer()
            with open(file_path, 'w') as f:
                f.write(visualizer.to_html(extraction_result))
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class EntityExtraction:
    """Main entity extraction class"""
    
    def __init__(self, use_spacy: bool = True, use_transformers: bool = False, 
                 spacy_model: str = "en_core_web_sm", transformer_model: str = "dslim/bert-base-NER"):
        """Initialize entity extraction"""
        self.extractors = []
        
        # Initialize extractors
        if use_spacy and spacy is not None:
            try:
                self.extractors.append(SpacyEntityExtractor(model=spacy_model))
            except:
                logger.warning(f"Failed to initialize spaCy extractor with model {spacy_model}")
        
        if use_transformers and pipeline is not None:
            try:
                self.extractors.append(TransformerEntityExtractor(model_name=transformer_model))
            except:
                logger.warning(f"Failed to initialize transformer extractor with model {transformer_model}")
        
        # Initialize NLTK extractor as fallback
        if nltk is not None:
            try:
                self.extractors.append(NLTKEntityExtractor())
            except:
                logger.warning("Failed to initialize NLTK extractor")
        
        # Initialize regex extractor with common patterns
        self.regex_extractor = RegexEntityExtractor({
            "PERSON": [r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"],
            "EMAIL": [r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"],
            "PHONE": [r"\b\+?[0-9]{1,3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"],
            "URL": [r"\bhttps?://[^\s]+\b"],
            "IP_ADDRESS": [r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"]
        })
        self.extractors.append(self.regex_extractor)
        
        # Initialize dictionary extractor
        self.dict_extractor = DictionaryEntityExtractor()
        self.extractors.append(self.dict_extractor)
        
        # Initialize ML extractor
        if np is not None:
            try:
                self.ml_extractor = MLEntityExtractor()
                self.extractors.append(self.ml_extractor)
            except:
                logger.warning("Failed to initialize ML extractor")
                self.ml_extractor = None
        else:
            self.ml_extractor = None
        
        # Initialize entity linker
        self.entity_linker = EntityLinker()
        
        # Initialize relationship extractor
        self.relationship_extractor = RelationshipExtractor({
            "works-for": [r"\bworks for\b", r"\bemployed by\b", r"\bis at\b"],
            "located-in": [r"\blocated in\b", r"\bbased in\b", r"\bheadquartered in\b"],
            "founded-by": [r"\bfounded by\b", r"\bcreated by\b", r"\bestablished by\b"]
        })
        
        # Initialize visualizer and exporter
        self.visualizer = EntityVisualizer()
        self.exporter = EntityExporter()
    
    def extract(self, text: str, extractors: List[str] = None, link_entities: bool = True, 
               extract_relationships: bool = True) -> ExtractionResult:
        """Extract entities from text"""
        # Select extractors to use
        if extractors:
            selected_extractors = []
            for extractor in self.extractors:
                extractor_name = extractor.__class__.__name__
                if any(name.lower() in extractor_name.lower() for name in extractors):
                    selected_extractors.append(extractor)
        else:
            selected_extractors = self.extractors
        
        if not selected_extractors:
            return ExtractionResult(text=text, metadata={"error": "No extractors available"})
        
        # Extract entities using each extractor
        results = []
        for extractor in selected_extractors:
            try:
                result = extractor.extract(text)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in {extractor.__class__.__name__}: {str(e)}")
        
        # Merge results
        merged_result = results[0]
        for result in results[1:]:
            merged_result = merged_result.merge_with(result)
        
        # Link entities if requested
        if link_entities:
            merged_result = self.entity_linker.link_entities(merged_result)
        
        # Extract relationships if requested
        if extract_relationships:
            merged_result = self.relationship_extractor.extract_relationships(merged_result)
        
        return merged_result
    
    def add_regex_pattern(self, entity_type: str, pattern: str) -> None:
        """Add a regex pattern for entity extraction"""
        self.regex_extractor.add_pattern(entity_type, pattern)
    
    def add_dictionary_terms(self, entity_type: str, terms: List[str]) -> None:
        """Add dictionary terms for entity extraction"""
        self.dict_extractor.add_terms(entity_type, terms)
    
    def add_relationship_pattern(self, relation_type: str, pattern: str) -> None:
        """Add a pattern for relationship extraction"""
        self.relationship_extractor.add_pattern(relation_type, pattern)
    
    def add_entity_to_kb(self, entity_id: str, entity_data: Dict[str, Any]) -> None:
        """Add entity to knowledge base for linking"""
        self.entity_linker.add_entity(entity_id, entity_data)
    
    def train_ml_extractor(self, texts: List[str], annotations: List[List[Dict[str, Any]]]) -> None:
        """Train ML extractor with annotated examples"""
        if self.ml_extractor is None:
            logger.error("ML extractor not available")
            return
        
        self.ml_extractor.train(texts, annotations)
    
    def save_ml_model(self, path: str) -> None:
        """Save ML model to file"""
        if self.ml_extractor is None:
            logger.error("ML extractor not available")
            return
        
        self.ml_extractor.save(path)
    
    def load_ml_model(self, path: str) -> None:
        """Load ML model from file"""
        if self.ml_extractor is None:
            logger.error("ML extractor not available")
            return
        
        self.ml_extractor.load(path)
    
    def visualize(self, extraction_result: ExtractionResult, format: str = "text") -> str:
        """Visualize extraction result"""
        format = format.lower()
        
        if format == "text":
            return self.visualizer.highlight_entities(extraction_result)
        
        elif format == "html":
            return self.visualizer.to_html(extraction_result)
        
        elif format == "graph":
            return self.visualizer.to_graph(extraction_result)
        
        else:
            raise ValueError(f"Unsupported visualization format: {format}")
    
    def export(self, extraction_result: ExtractionResult, format: str = "json") -> Any:
        """Export extraction result"""
        format = format.lower()
        
        if format == "json":
            return self.exporter.to_json(extraction_result)
        
        elif format == "csv":
            return self.exporter.to_csv(extraction_result)
        
        elif format == "dataframe":
            return self.exporter.to_dataframe(extraction_result)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def save(self, extraction_result: ExtractionResult, file_path: str, format: str = "json") -> None:
        """Save extraction result to file"""
        self.exporter.save_to_file(extraction_result, file_path, format)
    
    def get_supported_entity_types(self) -> List[EntityType]:
        """Get all supported entity types"""
        supported_types = set()
        for extractor in self.extractors:
            supported_types.update(extractor.get_supported_entity_types())
        
        return list(supported_types)


# Example usage
if __name__ == "__main__":
    # Initialize entity extraction
    extractor = EntityExtraction(use_spacy=True, use_transformers=False)
    
    # Extract entities from text
    text = "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976. The company is headquartered in Cupertino, California."
    result = extractor.extract(text)
    
    # Print entities
    print("Extracted Entities:")
    for entity in result.entities:
        print(f"- {entity.text} ({entity.type.value})")
    
    # Print relationships
    print("\nExtracted Relationships:")
    for relationship in result.relationships:
        print(f"- {relationship.source.text} {relationship.relation_type} {relationship.target.text}")
    
    # Visualize entities
    highlighted = extractor.visualize(result)
    print(f"\nHighlighted Text:\n{highlighted}")
    
    # Export to JSON
    json_output = extractor.export(result)
    print(f"\nJSON Output:\n{json_output}")

