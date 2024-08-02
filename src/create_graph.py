import configparser
import argparse
import openai
import concurrent.futures
import logging
from functools import partial
from typing import List, Tuple, Optional
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

def parse_segment(segment: str, type_: str, size: int, parent: str) -> Tuple[Optional[Node], Optional[Edge]]:
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
    return None, None

def parse_output(output: str, type_: str, parent: str = '') -> Tuple[List[Node], List[Edge]]:
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
            if node:
                nodes.append(node)
            if edge:
                edges.append(edge)

    return nodes, edges

def add_edges(nodes: List[Node], edges: List[Edge], target_node_id: str, source_type: str) -> List[Edge]:
    '''
    This function adds edges between the source node and the target nodes of the specified type.

    Parameters:
    nodes (list): The list of nodes.
    edges (list): The list of edges.
    target_node_id (str): The target node id.
    source_type (str): The source nodes type.

    Returns:
    list: The list of edges.
    '''
    for node in nodes:
        if node.type == source_type:
            edges.append(Edge(node.id, target_node_id, "parent-child", "10"))
    return edges

def process_message(concept_idx: int, concepts: list[Node], subject: str, study_level: str, language: str, additional_instructions: str) -> Tuple[List[Node], List[Edge]]:
    '''
    This function processes a message from the openAI model and returns the nodes and edges.

    Parameters:
    concept_idx (int): The index of the concept to process.
    concepts (Node): The list of concepts.
    subject (str): The subject.
    study_level (str): The study level.
    language (str): The output language.
    additional_instructions (str): Additional instructions.

    Returns:
    tuple: A tuple containing the nodes and edges.
    '''
    concept = concepts[concept_idx]
    not_included_concepts = concepts[:concept_idx] + concepts[concept_idx + 1:]
    not_included_concepts = [c.label for c in not_included_concepts]
    logging.info(f"Processing message: {concept.label}")
    prompt = open("docs/second_level.txt", "r").read()
    text = f"Given concept: {subject} - {concept.label}\nEducational level: {study_level}\nOutput language: {language}\nConcepts to not be included: {', '.join(not_included_concepts)}"
    if additional_instructions:
        text += f"\nAdditional instructions: {additional_instructions}"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
    ]
    logging.debug(f"Processing second-level user message: {text}")
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

def main() -> None:
    '''
    This is the main function.

    It prompts the user for the subject, study level, and language, and then creates a graph from the output of the openAI model.
    '''
    subject = input("Enter subject/concept/topic: ")
    study_level = input("Enter education level: ")
    language = input("Enter output language: ")
    additional_instructions = input("Enter additional instructions for construction of first-level concepts: ")
    micro_additional_instructions = input("Enter additional instructions for construction of second-level concepts: ")

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--nooutput", action="store_true")
    args = parser.parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # First level
    prompt = open("docs/first_level.txt", "r").read()
    text = f"Subject: {subject}\nEducational level: {study_level}\nOutput language: {language}"
    if additional_instructions:
        text += f"\nAdditional instructions: {additional_instructions}"
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": text}]
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

    process_message_partial = partial(process_message, concepts=concepts, subject=subject, study_level=study_level, language=language, additional_instructions=micro_additional_instructions)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_message_partial, range(len(concepts)))

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
