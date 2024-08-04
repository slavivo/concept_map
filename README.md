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

Also, you need to set up config.ini file in the src directory with the following structure:
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

Note that in the interactibe hierarchical graph you have the option of regenerating subgraphs and saving them. But in order to also generate gra requirements you also need to include the path to a .csv file (like docs/inf_rvp.csv) containing the requirements. For example:
```bash
python3 src/dash_scape.py -f <path_to_pkl_graph_file> -r <path_to_csv_file>
```

### Example

A knowledge graph for the subject "Informatics" in Czech at the "9th grade" level.

Color coding is done based on the RVP (Rámcový vzdělávací program) for the Czech Republic. The colors are as follows:
- Green - 6th grade
- Yellow - 7th grade
- Orange - 8th grade
- Red - 9th grade
- Gray - optional

![](docs/example_graph.gif)



