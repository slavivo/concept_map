# concept_map

This project aims to create knowledge graphs that for a given subject/field, language and level of study. The nodes are hierarchical - the subject, concepts and micro-concepts belonging to the various concepts. The edges between the same-level nodes dictate the order in which they should be learnt. The edges between different-level nodes dictate the parent-child relationship between the nodes.

## Installation

For creating knowledge graphs the following libraries are required:
- openai, numpy, tenacity
For interactive hierarchical graphs you will need:
- dash
- dash-cytoscape
For standard graph visualization you will need:
- gephi

All the required libraries (gephi not included) can be installed using the following command:
```bash
pip install requirements.txt
```

## Usage

### Creating a knowledge graph

To create a knowledge graph you have two options:

1. Single-level simple graph
```bash
python3 src/create_graph_v2.py
```

2. Multi-level hierarchical graph
```bash
python3 src/create_graph.py
```

The first options will create .graphml file that can be visualized using Gephi. The second option will also create a .pkl file that can be visualized using the interactive hierarchical graph.

### Visualizing the knowledge graph

To visualize the knowledge graph you have two options:

1. Interactive hierarchical graph
```bash 
python3 src/dash_scape.py -f <path_to_pkl_graph_file>
```

2. Standard graph visualization - Gephi

### Example

A knowledge graph for the subject "Mathematics" in English at the "8th grade" level.

[View the knowledge graph](docs/third_graph.pdf)

<video width="320" height="240" controls>
  <source src="docs/graph_video.webm" type="video/webm">
  Your browser does not support the video tag.
</video>



