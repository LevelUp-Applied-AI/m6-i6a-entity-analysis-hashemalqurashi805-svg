import unicodedata
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from itertools import combinations

# Import functions from extensions.py
from extensions import add_custom_climate_rules, export_knowledge_graph, analyze_temporal_trends

def load_corpus(filepath="data/climate_articles.csv"):
    """Task 1: Load CSV dataset."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def preprocess_corpus(df):
    """Task 1: NFC Normalization for text consistency."""
    if df is None: return None
    processed_df = df.copy()
    processed_df['processed_text'] = processed_df['text'].apply(
        lambda x: unicodedata.normalize('NFC', str(x))
    )
    return processed_df

def run_ner_pipeline(df, nlp):
    """Task 2: Efficient batch processing for English entities."""
    en_df = df[df['language'] == 'en'].copy()
    entity_data = []
    
    for doc, text_id in zip(nlp.pipe(en_df['processed_text']), en_df['id']):
        for ent in doc.ents:
            entity_data.append({
                'text_id': text_id,
                'entity_text': ent.text,
                'entity_label': ent.label_,
                'start_char': ent.start_char,
                'end_char': ent.end_char
            })
    return pd.DataFrame(entity_data)

def aggregate_entity_stats(entity_df, articles_df):
    """Task 3: Compute all required statistics."""
    if entity_df.empty: return None

    # Top 20 Entities
    top_20 = entity_df.groupby(['entity_text', 'entity_label']).size().reset_index(name='count')
    top_20 = top_20.sort_values(by='count', ascending=False).head(20)

    # Label Counts
    label_counts = entity_df['entity_label'].value_counts().to_dict()

    # Co-occurrence Logic
    co_counter = Counter()
    for _, group in entity_df.groupby('text_id'):
        unique_ents = sorted(list(set(group['entity_text'])))
        if len(unique_ents) >= 2:
            for pair in combinations(unique_ents, 2):
                co_counter[pair] += 1
    
    co_occurrence = pd.DataFrame([
        {'entity_a': a, 'entity_b': b, 'co_count': count}
        for (a, b), count in co_counter.items()
    ]).sort_values(by='co_count', ascending=False).head(50)

    # Per-Category Stats
    merged = entity_df.merge(articles_df[['id', 'category']], left_on='text_id', right_on='id')
    per_category = merged.groupby(['category', 'entity_label']).size().reset_index(name='count')

    return {
        'top_entities': top_20,
        'label_counts': label_counts,
        'co_occurrence': co_occurrence,
        'per_category': per_category
    }

def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Task 4: Horizontal bar chart of top entities."""
    df = stats['top_entities']
    plt.figure(figsize=(12, 8))
    plt.barh(df['entity_text'] + " (" + df['entity_label'] + ")", df['count'], color='skyblue')
    plt.gca().invert_yaxis()
    plt.title("Top 20 Extracted Entities by Frequency")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(stats):
    """Task 5: Final text report summary."""
    top_5 = stats['top_entities'].head(5)
    report = ["\n--- FINAL ENTITY ANALYSIS REPORT ---", f"Global Labels: {stats['label_counts']}", "\nTop 5 Entities:"]
    for _, row in top_5.iterrows():
        report.append(f"- {row['entity_text']} ({row['entity_label']}): {row['count']} times")
    return "\n".join(report)

if __name__ == "__main__":
    # 1. Initialize NLP with Challenge Level 3
    nlp = spacy.load("en_core_web_sm")
    nlp = add_custom_climate_rules(nlp) 

    # 2. Execute Core Pipeline
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        entities = run_ner_pipeline(corpus, nlp)
        
        if not entities.empty:
            # 3. Analytics
            stats = aggregate_entity_stats(entities, corpus)
            
            # 4. Visualizations & Challenge Exports
            visualize_entity_distribution(stats)
            export_knowledge_graph(stats) # Challenge Level 2
            
            # 5. Challenge Level 1: Temporal Analysis
            trends = analyze_temporal_trends(entities)
            print("\n--- Temporal Trends (Years) ---")
            print(trends)

            # 6. Final Report
            print(generate_report(stats))
            print("\nAll tasks and challenge extensions completed successfully!")