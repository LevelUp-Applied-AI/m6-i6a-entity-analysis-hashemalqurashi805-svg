content = """# Integration 6A — Entity Analysis Pipeline

Module 6 Week A integration task for AI.SPIRE Applied AI & ML Systems.

## Project Overview
This project implements a robust, modular NLP pipeline designed to process climate-related articles. It extracts key entities (Named Entity Recognition), computes detailed statistics, and generates visualizations to uncover insights within the climate change discourse. The implementation includes advanced architectural patterns like dependency injection and modular extensions to handle domain-specific challenges.

## Core Features
- **Data Preprocessing:** Implements language-aware cleaning and Unicode NFC normalization to ensure consistency across diverse datasets.
- **Efficient NER Pipeline:** Utilizes spaCy's batch processing (`nlp.pipe`) for high-performance entity extraction from English text.
- **Statistical Aggregation:** Computes top 20 entities, global label distributions, and entity co-occurrence pairs for relationship analysis.
- **Visual Analytics:** Generates horizontal bar charts of top entities for immediate visual impact.
- **Structured Reporting:** Produces human-readable summaries of the corpus findings.

## Challenge Extensions (Advanced)
To provide deeper analytical value, the following challenge levels were implemented via a dedicated `extensions.py` module:

### Level 1: Temporal Analysis
- **Temporal Trend Extraction:** Automatically extracts years from `DATE` entities using optimized Regex patterns.
- **Reporting Focus:** Analyzes frequency trends across specific years (e.g., 2030, 2050) to identify the temporal focus of the corpus.

### Level 2: Knowledge Graph Mapping
- **Relationship Weighting:** Maps entity co-occurrence into a structured format with weighted relationship levels (Strong, Moderate, Weak).
- **CSV Export:** Generates `entity_relationships.csv`, ready for advanced graph visualization in tools like Gephi or Cytoscape.

### Level 3: Domain-Specific NER (EntityRuler)
- **Custom Patterns:** Added over 15 specialized patterns to recognize climate-specific entities that standard models often miss (e.g., "COP28", "Net Zero", "Paris Agreement").
- **Extended Schema:** Introduced new labels like `CLIMATE_EVENT`, `POLICY`, `REPORT`, and `THRESHOLD`.

## Project Structure
- `entity_analysis.py`: The main execution pipeline and core logic.
- `extensions.py`: Modular extensions for advanced analysis and custom rules.
- `entity_distribution.png`: Visual representation of top entities.
- `entity_relationships.csv`: Data export for Knowledge Graph analysis.

## Setup & Execution

1. **Clone the repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm