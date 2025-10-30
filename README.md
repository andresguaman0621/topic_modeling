# Topic Modeling Using Document Networks

## Project Overview

This project implements a topic modeling system using document networks based on user-selected words describing mobile app usability. The analysis compares two similarity measures (Cosine and Jaccard) to identify which produces better topics through community detection.

## Files Included

1. **AG.ipynb** - The main Jupyter Notebook containing all code, analysis, and documentation
2. **AG.html** - HTML export of the notebook for easy viewing without Jupyter
3. **convert_to_html.py** - Python script used to convert the notebook to HTML
4. **README.md** - This file

## Dataset

The project uses the words_ux.csv dataset from:
https://raw.githubusercontent.com/marsgr6/estadistica-ux/main/data/words_ux.csv

The dataset contains a single column (Word) with user-selected words describing mobile app usability.

## Requirements

To run the notebook, you'll need the following Python packages:

```
pandas
numpy
matplotlib
seaborn
networkx
scikit-learn
scipy
python-louvain
```

Install all requirements with:

```bash
pip install pandas numpy matplotlib seaborn networkx scikit-learn scipy python-louvain
```

## How to Use

### Option 1: View the HTML File (No Installation Required)

Simply open **AG.html** in any web browser to view the complete analysis with all code, outputs, and visualizations.

### Option 2: Run the Jupyter Notebook

1. Install Jupyter if you haven't already:
   ```bash
   pip install jupyter
   ```

2. Navigate to the project directory:
   ```bash
   cd C:\Users\a.guaman\Downloads\IA_2
   ```

3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open **AG.ipynb** in the Jupyter interface

5. Run all cells sequentially (Cell > Run All) or execute them one by one

## Project Structure

The notebook is organized into the following sections:

### 1. Data Preparation (15%)
- Load the dataset from the URL
- Explore data structure
- Preprocess (handle missing values, normalize text)
- Create document structure

### 2. Document-Term Matrix (DTM) Construction (20%)
- Build binary DTM using CountVectorizer
- Analyze word frequencies and distribution
- Display DTM structure

### 3. Similarity Measures (20%)
- Compute Cosine Similarity matrix
- Compute Jaccard Similarity matrix
- Compare similarity distributions
- Analyze similarity statistics

### 4. Document Network Construction and Trimming (20%)
- Build networks from similarity matrices
- Apply threshold-based edge trimming
- Analyze network statistics (density, degree distribution)
- Compare network characteristics

### 5. Community Detection and Topic Identification (15%)
- Apply Louvain community detection algorithm
- Extract representative words for each community
- Label topics based on dominant words
- Display topics for both similarity measures

### 6. Evaluation and Decision (10%)
- Calculate modularity scores
- Measure internal coherence
- Assess topic diversity
- Compute overall evaluation scores
- Make data-driven decision on best similarity measure

### 7. Visualization
- Network visualizations with communities
- Degree distributions
- Similarity distributions
- Comparison charts

### 8. Conclusions
- Summary of findings
- Final recommendation
- Limitations and future work

## Key Findings

The notebook provides a comprehensive comparison of Cosine Similarity and Jaccard Similarity for document network-based topic modeling. The evaluation is based on:

- **Modularity**: Quality of community structure
- **Internal Coherence**: Similarity within topics
- **Topic Diversity**: Distinctness between topics
- **Interpretability**: Clarity and meaningfulness of topics

The final decision is made using a weighted combination of these metrics, with detailed justification provided in the notebook.

## Methodology

1. **Document Creation**: Words are grouped into documents (4 words per document by default)
2. **Binary DTM**: Each document is represented by presence/absence of words
3. **Similarity Calculation**: Two measures capture different aspects of document similarity
4. **Network Construction**: Documents become nodes, similarities become weighted edges
5. **Edge Trimming**: 70th percentile threshold maintains network density
6. **Community Detection**: Louvain algorithm identifies topic clusters
7. **Evaluation**: Multi-metric assessment determines superior method

## Customization

You can modify the following parameters in the notebook:

- `words_per_document`: Number of words per document (default: 4)
- Similarity threshold percentile (default: 70th percentile)
- Number of top words to display per topic (default: 15)
- Evaluation metric weights (default: modularity=0.3, coherence=0.4, diversity=0.3)

## References

- Text Analysis - Word's Graph: https://si2lab-udla.shinyapps.io/usability/
- Topic Coherence Paper: https://ieeexplore.ieee.org/iel7/6287639/8948470/09003400.pdf
- Distance Measures in ML: https://www.linkedin.com/pulse/understanding-different-distance-measures-tiago-davi-1f/

## Author

A. Guaman

## Date

October 30, 2025

## Notes

- The notebook includes detailed comments explaining each step
- All visualizations are embedded in the output
- Statistical measures support the final decision
- The methodology follows best practices for network-based topic modeling

For questions or issues, please refer to the inline documentation in the notebook.
