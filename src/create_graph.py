import configparser
import argparse
import openai
import concurrent.futures
import logging
from functools import partial
from utils import (
    Node,
    Edge,
    create_dashscape_tree,
    create_graphml_tree,
    RequestParams,
    chat_completion_request,
)
    

# Read the config file
config = configparser.ConfigParser()
config.read("src/config.ini")
OPENAI_KEY = config["DEFAULT"]["OPENAI_KEY"]
GPT_MODEL = config["DEFAULT"]["GPT_MODEL"]

# Create the OpenAI client
client = openai.Client(api_key=OPENAI_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)

def parse_segment(segment, type_, size, parent) -> tuple:
    '''
    This function parses a segment of the output from the openAI model and returns either a node or an edge.

    Parameters:
    segment (str): The segment to parse.
    type_ (str): The type of the node.
    size (int): The size of the node.
    parent (str): The parent node.

    Returns:
    tuple: A tuple containing the node and the edge.
    '''
    if segment.startswith("r"):
        parts = segment.strip('"').split("|")
        if len(parts) == 4:
            return None, Edge(parts[1].lower().replace(" ", "_") + parent, parts[2].lower().replace(" ", "_") + parent, "sibling", parts[3])
        else:
            logging.info(f"Invalid edge: {segment}")
    elif segment.startswith("c"):
        parts = segment.strip('"').split("|")
        if len(parts) == 2:
            return Node(parts[1].lower().replace(" ", "_") + parent, parts[1], type_, size), None
        else:
            logging.info(f"Invalid node: {segment}")
    else:
        return None, None

def parse_output(output, type_, parent='') -> tuple:
    '''
    This function parses the output from the openAI model and returns a list of nodes and a list of edges.

    Parameters:
    output (str): The output from the openAI model.
    type_ (str): The type of the nodes.
    parent (str): The parent node.

    Returns:
    tuple: A tuple containing the list of nodes and the list of edges.
    '''
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

    for segment in output.strip().split(";"):
        if segment:
            node, edge = parse_segment(segment, type_, size, parent)
            nodes.append(node) if node else None
            edges.append(edge) if edge else None

    return nodes, edges

def add_edges(nodes, edges, source_node_id, target_type) -> list:
    '''
    This function adds edges between the source node and the target nodes of the specified type.

    Parameters:
    nodes (list): The list of nodes.
    edges (list): The list of edges.
    source_node_id (str): The source node id.
    target_type (str): The target node type.

    Returns:
    list: The list of edges.
    '''
    for node in nodes:
        if node.type == target_type:
            edges.append(Edge(source_node_id, node.id, "parent-child", "10"))
    return edges

def process_message(concept, study_level, language) -> tuple:
    '''
    This function processes a message from the openAI model and returns the nodes and edges.

    Parameters:
    concept (tuple): The concept to process.
    study_level (str): The study level.
    language (str): The output language.

    Returns:
    tuple: A tuple containing the nodes and edges.
    '''
    logging.info(f"Processing message: {concept.label}")
    prompt = open("docs/second_level.txt", "r").read()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Concept level: {study_level}\nLanguage: {language}\nConcept: {concept.label}"},
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
    nodes_, edges_ = parse_output(response, "micro-concept", concept.id)
    edges_ = add_edges(nodes_, edges_, concept.id, "micro-concept")
    return nodes_, edges_

def main():
    '''
    This is the main function.

    It prompts the user for the subject, study level, and language, and then creates a graph from the output of the openAI model.
    '''
    subject = input("Enter subject/concept/topic: ")
    study_level = input("Enter education level: ")
    language = input("Enter output language: ")

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

    # Parse the output and add the major node
    nodes, edges = parse_output(response, "concept", f'{study_level} {subject}')
    major_label = f'{study_level} {subject}'
    nodes.insert(0, Node(major_label.lower().replace(" ", "_"), major_label, "major-concept", 15))

    # Add edges from the major node to the first level nodes
    edges = add_edges(nodes, edges, nodes[0].id, "concept")

    logging.debug(f"First level nodes: {nodes}\n")
    logging.debug(f"First level edges: {edges}\n")

    # Second level
    prompt = open("docs/second_level.txt", "r").read()
    concepts = [n for n in nodes if n.type == "concept"]

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
