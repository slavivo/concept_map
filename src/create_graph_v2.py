import configparser
import argparse
import openai
from utils import RequestParams, chat_completion_request
import xml.etree.ElementTree as ET
import concurrent.futures
import logging

config = configparser.ConfigParser()
config.read("src/config.ini")

OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]

client = openai.Client(api_key=OPENAI_KEY)

logging.basicConfig(level=logging.INFO)


def parse_line(line, nodes, edges, type_, size):
    if line.startswith("r"):
        parts = line.strip('"').split("|")
        if len(parts) == 4:
            edges.append((parts[0], parts[1], parts[2], "sibling", parts[3]))
        else:
            logging.info(f"Invalid edge: {line}")
    elif line.startswith("c"):
        parts = line.strip('"').split("|")
        if len(parts) == 2:
            nodes.append((parts[0], parts[1], type_, size))
        else:
            logging.info(f"Invalid node: {line}")
    else:
        return


def parse_output(output, type_):
    nodes = []
    edges = []
    size = 0
    if type_ == "micro-concept":
        size = 3
    elif type_ == "concept":
        size = 6
    elif type_ == "major-concept":
        size = 15

    for line in output.strip().split(";"):
        if line:
            parse_line(line, nodes, edges, type_, size)

    return nodes, edges


def add_edges(nodes, edges, source_node, target_type):
    for node in nodes:
        if node[2] == target_type:
            edges.append(("r", source_node, node[1], "parent-child", "10"))
    return edges


def process_message(msg):
    logging.info(f"Processing message: {msg}")
    prompt = open("docs/second_level.txt", "r").read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Math for eights grade: {msg}"},
    ]
    params = RequestParams(
        client,
        messages=messages,
        model=GPT_MODEL,
        max_tokens=4096,
        temperature=0.2,
        top_p=0.1,
    )
    response = chat_completion_request(params)

    response = response.choices[0].message.content.replace("\n", "")
    nodes_, edges_ = parse_output(response, "micro-concept")
    edges_ = add_edges(nodes_, edges_, msg, "micro-concept")
    return nodes_, edges_


def create_tree(nodes, edges):
    # Create the root element
    graphml = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")

    # Create keys for node and edge data
    ET.SubElement(
        graphml,
        "key",
        id="d0",
        **{"for": "node", "attr.name": "type", "attr.type": "string"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="size_",
        **{"for": "node", "attr.name": "size_", "attr.type": "integer"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="name",
        **{"for": "edge", "attr.name": "name", "attr.type": "string"},
    )
    ET.SubElement(
        graphml,
        "key",
        id="weight",
        **{"for": "edge", "attr.name": "weight", "attr.type": "integer"},
    )

    # Create the graph element
    graph = ET.SubElement(graphml, "graph", id="G", edgedefault="directed")

    # Add nodes
    node_elements = {}
    for node in nodes:
        concept, name, type_, size = node
        node_element = ET.SubElement(graph, "node", id=name)
        data_element = ET.SubElement(node_element, "data", key="d0")
        data_element.text = type_
        data_element = ET.SubElement(node_element, "data", key="label")
        data_element.text = name
        data_size = ET.SubElement(node_element, "data", key="size_")
        data_size.text = str(size)
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
    tree.write("docs/concept_map.graphml", encoding="utf-8", xml_declaration=True)


def main():
    prompt = open("docs/graph_create.txt", "r").read()
    msg = open("docs/example.txt", "r").read()
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": msg}]
    params = RequestParams(
        client,
        messages=messages,
        model=GPT_MODEL,
        max_tokens=4096,
        temperature=0.2,
        top_p=0.1,
    )
    response = chat_completion_request(params)

    response = response.choices[0].message.content.replace("\n", "")
    nodes, edges = parse_output(response, "concept")

    logging.info(f"Nodes: {nodes}")
    logging.info(f"Edges: {edges}")

    create_tree(nodes, edges)

if __name__ == "__main__":
    main()
