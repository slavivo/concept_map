import configparser
import argparse
import openai
from utils import RequestParams, chat_completion_request
import xml.etree.ElementTree as ET

config = configparser.ConfigParser()
config.read('src/config.ini')

OPENAI_KEY = config['DEFAULT']['OPENAI_KEY']
GPT_MODEL = config['DEFAULT']['GPT_MODEL']

def parse_output(output):
    nodes = []
    edges = []
    for line in output.strip().split(';'):
        if line:
            if line.startswith('r'):
                parts = line.strip('"').split('|')
                edges.append((parts[0], parts[1], parts[2], parts[3], parts[4]))
            elif line.startswith('c'):
                parts = line.strip('"').split('|')
                nodes.append((parts[0], parts[1], parts[2]))
            else:
                continue
    return nodes, edges


def main():
    client = openai.Client(api_key=OPENAI_KEY)

    prompt = open('src/entity_extraction.txt', 'r').read()
    msg = open('src/example.txt', 'r').read()
    messages = [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': msg}]
    params = RequestParams(client, messages=messages, model=GPT_MODEL, max_tokens=4096, temperature=0.2, top_p=0.1)
    response = chat_completion_request(params)

    response = response.choices[0].message.content.replace('\n', '')
    print(response)
    nodes, edges = parse_output(response)
    nodes.insert(0, ("concept", "8th grade math", "major-concept"))
    print(nodes)
    print(edges)

    
    # Create the root element
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")

    # Create keys for node and edge data
    ET.SubElement(graphml, "key", id="d0", **{"for": "node", "attr.name": "type", "attr.type": "string"})
    ET.SubElement(graphml, "key", id="name", **{"for": "edge", "attr.name": "name", "attr.type": "string"})
    ET.SubElement(graphml, "key", id="weight", **{"for": "edge", "attr.name": "weight", "attr.type": "integer"})

    # Create the graph element
    graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

    # Add nodes
    node_elements = {}
    for node in nodes:
        concept, name, type_ = node
        node_element = ET.SubElement(graph, "node", id=name)
        data_element = ET.SubElement(node_element, "data", key="d0")
        data_element.text = type_
        node_elements[name] = node_element

    # Add edges
    for edge in edges:
        relationship, source, target, type_, weight = edge
        edge_element = ET.SubElement(graph, "edge", source=source, target=target)
        data_name = ET.SubElement(edge_element, "data", key="name")
        data_name.text = type_
        data_weight = ET.SubElement(edge_element, "data", key="weight")
        data_weight.text = weight

    # Convert the tree to a string
    tree = ET.ElementTree(graphml)
    tree.write("hierarchical_graph_with_intra_level.graphml", encoding="utf-8", xml_declaration=True)

if __name__ == '__main__':
    main()