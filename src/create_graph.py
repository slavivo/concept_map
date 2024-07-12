import configparser
import argparse
import openai
from utils import RequestParams, chat_completion_request
import xml.etree.ElementTree as ET
import concurrent.futures
import logging
import pickle
from functools import partial
import datetime

config = configparser.ConfigParser()
config.read("src/config.ini")

OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]

client = openai.Client(api_key=OPENAI_KEY)

logging.basicConfig(level=logging.INFO)


def parse_line(line, nodes, edges, type_, size, parent):
    if line.startswith("r"):
        parts = line.strip('"').split("|")
        if len(parts) == 4:
            edges.append((parts[1].lower().replace(" ", "_") + parent, parts[2].lower().replace(" ", "_") + parent, "sibling", parts[3]))
        else:
            logging.info(f"Invalid edge: {line}")
    elif line.startswith("c"):
        parts = line.strip('"').split("|")
        if len(parts) == 2:
            nodes.append((parts[1].lower().replace(" ", "_") + parent, parts[1], type_, size))
        else:
            logging.info(f"Invalid node: {line}")
    else:
        return


def parse_output(output, type_, parent=''):
    nodes = []
    edges = []
    size = 0
    if type_ == "micro-concept":
        size = 3
    elif type_ == "concept":
        size = 6
    elif type_ == "major-concept":
        size = 15
    if parent:
        parent = "__" + parent.lower().replace(" ", "_")

    for line in output.strip().split(";"):
        if line:
            parse_line(line, nodes, edges, type_, size, parent)

    return nodes, edges


def add_edges(nodes, edges, source_node_id, target_type):
    for node in nodes:
        if node[2] == target_type:
            edges.append((source_node_id, node[0], "parent-child", "10"))
    return edges


def process_message(concept, study_level, language):
    logging.info(f"Processing message: {concept[1]}")
    prompt = open("docs/second_level.txt", "r").read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Concept level: {study_level}\nLanguage: {language}\nConcept: {concept[1]}"},
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
    nodes_, edges_ = parse_output(response, "micro-concept", concept[1])
    edges_ = add_edges(nodes_, edges_, concept[0], "micro-concept")
    return nodes_, edges_


def create_graphml_tree(nodes, edges):
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
        id_, name, type_, size = node
        node_element = ET.SubElement(graph, "node", id=id_)
        data_element = ET.SubElement(node_element, "data", key="d0")
        data_element.text = type_
        data_element = ET.SubElement(node_element, "data", key="label")
        data_element.text = name
        data_size = ET.SubElement(node_element, "data", key="size_")
        data_size.text = str(size)
        node_elements[name] = node_element

    # Add edges
    for edge in edges:
        source, target, type_, weight = edge
        edge_element = ET.SubElement(graph, "edge", source=source, target=target)
        data_name = ET.SubElement(edge_element, "data", key="name")
        data_name.text = type_
        data_weight = ET.SubElement(edge_element, "data", key="weight")
        data_weight.text = weight

    # Convert the tree to a string
    tree = ET.ElementTree(graphml)
    tree.write(f"docs/graph_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.graphml", encoding="utf-8", xml_declaration=True)

def create_dashscape_tree(nodes, edges):
    major_nodes = [n for n in nodes if n[2] == "major-concept" or n[2] == "concept"]
    dashscape_major_nodes = [{'data': {'id': n[0], 'label': n[1]}} for n in major_nodes]
    major_nodes = set([n[0] for n in major_nodes])
    major_edges = [e for e in edges if e[0] in major_nodes and e[1] in major_nodes]
    major_edges = set(major_edges)
    dashscape_major_edges = [{'data': {'source': e[0], 'target': e[1]}} for e in major_edges]

    edge_major_to_micro = [e for e in edges if e[0] in major_nodes and e[1] not in major_nodes]

    subgraphs = {}
    for node in major_nodes:
        tmp_nodes = set()
        subgraphs[node] = {
            'nodes': [],
            'edges': []
        }
        for edge in edge_major_to_micro:
            if edge[0] == node:
                tmp_nodes.add(edge[1])
                subgraphs[node]['nodes'].append({'data': {'id': edge[1], 'label': edge[1].split('__')[0].replace('_', ' ').capitalize()}})
        subgraphs[node]['edges'] = [{'data': {'source': edge[0], 'target': edge[1]}} for edge in edges if edge[0] in tmp_nodes and edge[1] in tmp_nodes]

    with open (f'docs/graph_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl', 'wb') as f:
        pickle.dump((dashscape_major_nodes, dashscape_major_edges, subgraphs), f)


def main():
    subject = input("Enter subject: ")
    study_level = input("Enter study level: ")
    language = input("Enter the language: ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nooutput", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # First level
    prompt = open("docs/first_level.txt", "r").read()
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"Study level: {study_level}\nLanguage: {language}\nSubject: {subject}"}]
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
    nodes, edges = parse_output(response, "concept", f'{study_level} {subject}')
    major_label = f'{study_level} {subject}'
    nodes.insert(0, (major_label.lower().replace(" ", "_"), major_label, "major-concept", 15))
    edges = add_edges(nodes, edges, nodes[0][0], "concept")

    logging.debug(f"First level nodes: {nodes}\n")
    logging.debug(f"First level edges: {edges}\n")

    # Second level
    prompt = open("docs/second_level.txt", "r").read()
    concepts = [n for n in nodes if n[2] == "concept"]

    process_message_partial = partial(process_message, study_level=study_level, language=language)

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_message_partial, concepts)

        for nodes_, edges_ in results:
            nodes.extend(nodes_)
            edges.extend(edges_)

    logging.debug(f"Second level nodes: {nodes}\n")
    logging.debug(f"Second level edges: {edges}\n")

    # Create .graphml and dashscape tree
    if not args.nooutput:
        create_dashscape_tree(nodes, edges)
        create_graphml_tree(nodes, edges)


if __name__ == "__main__":
    main()
