# Concept Map

This project aims to create knowledge graphs that for a given subject/topic/concept, language and level of education. The nodes are hierarchical - the subject, concepts and micro-concepts belonging to the various concepts. The edges between the same-level nodes dictate the order in which they should be learnt. The edges between different-level nodes dictate the parent-child relationship between the nodes.

## Installation

For creating knowledge graphs the following python libraries are required:
- openai, numpy, tenacity

For interactive hierarchical graphs you will need:
- dash
- dash\-cytoscape

For standard graph visualization you will need:
- gephi

All the required libraries (gephi not included) can be installed using the following command:
```bash
pip install requirements.txt
```

Also, you need to set up config.ini file with the following structure:
```ini
[DEFAULT]
OPENAI_KEY= <your_openai_key>
GPT_MODEL= <your_gpt_model>
```

## Usage

### Creating a knowledge graph

To create a multi-level hierarchical knowledge graph you need to execute the following command:

```bash
python3 src/create_graph.py
```

This will create a .graphml file that can be visualized by gephi and also a .pkl file that can be visualized using the interactive hierarchical graph.

### Visualizing the knowledge graph

To visualize the knowledge graph you have two options:

1. Interactive hierarchical graph
```bash 
python3 src/dash_scape.py -f <path_to_pkl_graph_file>
```

2. Standard graph visualization - Gephi

### Example

A knowledge graph for the subject "Mathematics" in English at the "8th grade" level.

[View the knowledge graph here](docs/example_graph.pdf)

![](docs/example_graph.gif)



