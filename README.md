# Project Readme: Text Data Labeling and Evaluation with LLM

## Overview

Welcome to the Text Data Labeling and Evaluation project! This project utilizes a Language Model (LLM) to label text data for various entities based on a predefined contract dataset. The entities, including document name, party name, governing law, agreement date, effective date, and expiration date, can be customized and configured in a settings file. The labeling process involves engineering prompts for the LLM, which outputs are then parsed to extract information for the specified entities. The labeled data is further processed to meet a specific format for extensive evaluation using defined Key Performance Indicators (KPIs).

## Components

### 1. Language Model (LLM)

- **Usage**: Labels text data based on predefined entities.
- **Configuration**: Entities are defined in a config file for customization.
- **Prompt Engineering**: Crafted prompts to instruct the LLM for entity labeling.

### 2. Data Parsing

- **Role**: Extracts labeled information for the defined entities from the LLM output.

### 3. Data Formatting

- **Purpose**: Ensures the labeled data conforms to a specific format for evaluation.

### 4. Evaluation

- **Metrics**: Utilizes extensive Key Performance Indicators (KPIs) for evaluation.
- **Similarity Metric**: Employs TF-IDF vectorization and embedding for similarity evaluation.



