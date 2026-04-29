import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

def add_custom_climate_rules(nlp):
    """
    Challenge Level 3: Custom NER with EntityRuler.
    Adds 15+ climate-specific patterns to the pipeline.
    """
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        
        patterns = [
            {"label": "CLIMATE_EVENT", "pattern": "COP28"},
            {"label": "CLIMATE_EVENT", "pattern": "COP27"},
            {"label": "CLIMATE_EVENT", "pattern": "Paris Agreement"},
            {"label": "POLICY", "pattern": "Net Zero"},
            {"label": "POLICY", "pattern": "Carbon Tax"},
            {"label": "POLICY", "pattern": "Green New Deal"},
            {"label": "REPORT", "pattern": "AR6"},
            {"label": "REPORT", "pattern": "IPCC Report"},
            {"label": "REPORT", "pattern": "Special Report on Global Warming"},
            {"label": "THRESHOLD", "pattern": "1.5 degrees"},
            {"label": "THRESHOLD", "pattern": "2 degrees"},
            {"label": "ORG", "pattern": "UNFCCC"},
            {"label": "GAS", "pattern": "Methane"},
            {"label": "GAS", "pattern": "Carbon Dioxide"},
            {"label": "CLIMATE_EVENT", "pattern": "Kyoto Protocol"}
        ]
        ruler.add_patterns(patterns)
    return nlp

def export_knowledge_graph(stats):
    """
    Challenge Level 2: Knowledge Graph Export.
    Creates a CSV of entity relationships with weights and levels.
    """
    if stats is None or 'co_occurrence' not in stats:
        return
        
    df = stats['co_occurrence'].copy()
    
    def get_relationship_level(count):
        if count >= 10: return "Strong"
        if count >= 5: return "Moderate"
        return "Weak"
        
    df['weight'] = df['co_count']
    df['relationship_level'] = df['co_count'].apply(get_relationship_level)
    
    df.to_csv("entity_relationships.csv", index=False)
    print("-> Success: Knowledge Graph exported to entity_relationships.csv")

def analyze_temporal_trends(entity_df):
    """
    Challenge Level 1: Temporal Trends Analysis.
    Extracts years from DATE entities to analyze reporting focus.
    """
    dates = entity_df[entity_df['entity_label'] == 'DATE'].copy()
    dates['year'] = dates['entity_text'].str.extract(r'(\d{4})')
    trends = dates.dropna(subset=['year'])['year'].value_counts().sort_index()
    return trends