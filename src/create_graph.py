import configparser
import argparse
import openai
import concurrent.futures
import logging
import pandas as pd
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

def process_message(concept_idx: int, concepts: list, subject: str, study_level: str, language: str, additional_instructions: str, add_parent_edges: bool = True) -> Tuple[List[Node], List[Edge]]:
    '''
    This function processes a message from the openAI model and returns the nodes and edges.

    Parameters:
    concept_idx (int): The index of the concept to process.
    concepts: The list of concepts.
    subject (str): The subject.
    study_level (str): The study level.
    language (str): The output language.
    additional_instructions (str): Additional instructions.
    add_parent_edges (bool): Whether to add parent edges.

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

    while True:
        response = chat_completion_request(params)
        response = response.choices[0].message.content
        if response.endswith("#END#"):
            response = response[:-5]
            break
        logging.info("Second level response overflowed the token limit, retrying...")

    response = response.replace("\n", "")
    nodes_, edges_ = parse_output(response, "micro-concept", concept.id)
    if add_parent_edges:
        edges_ = add_edges(nodes_, edges_, concept.id, "micro-concept")
    return nodes_, edges_

def grade_subgraph(subgraph: Tuple[str, List[Node]] , prompt: str, language: str, grade_content: dict) -> None:
    '''
    This function adds grade requirements to the nodes.
    '''
    logging.info(f"Processing grade requirements for {subgraph[0]}")

    subject, nodes = subgraph
    node_labels = [n.label for n in nodes]

    text = f"""Language: {language}\n\nConcepts about {subject}: {node_labels}\n\nCurriculum outcomes: {grade_content}"""

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text},
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
    response = response.choices[0].message.content

    lines = response.split('\n')
    i = 0
    for node, line in zip(nodes, lines):
        label, req = line.split(': ')
        if label == node.label:
            node.requirement = int(req)
        else:
            logging.info(f"Label mismatch: {label} != {node.label}")

def add_requirements(language: str, requirements_path: str, nodes: List[Node], edges: List[Edge]) -> List[Node]:
    '''
    This function adds grade requirements to the nodes.

    Parameters:
    language (str): The output language.
    requirements_path (str): The path to the csv requirements file.
    nodes (list): The list of nodes.
    edges (list): The list of edges.

    Returns:
    list: The list of nodes.
    '''
    subgraphs = {}
    curr_concept = None
    for node in nodes:
        if node.type != 'micro-concept':
            continue
        node_concept = node.id.split('__')[1]
        if curr_concept != node_concept:
            curr_concept = node_concept
            subgraphs[curr_concept] = []
        subgraphs[curr_concept].append(node)

    prompt = open("docs/grade_requirements.txt", "r").read()
    curr_outcomes = pd.read_csv(requirements_path)
    curr_outcomes['doporucena_trida'] = curr_outcomes['doporucena_trida'].astype(int)
    curr_outcomes['vystup'] = curr_outcomes['vystup'].astype(str)
    curr_outcomes = curr_outcomes[['doporucena_trida', 'vystup']].apply(tuple, axis=1).tolist()
    grade_content = {}

    for grade, content in curr_outcomes:
        if grade not in grade_content:
            grade_content[grade] = []
        grade_content[grade].append(content)

    grade_subgraph_partial = partial(grade_subgraph, prompt=prompt, language=language, grade_content=grade_content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(grade_subgraph_partial, subgraphs.items())

    # node lookup dict
    node_dict = {node.id: node for node in nodes if node.type == 'micro-concept'}

    # edge lookup dicts
    edge_source_dict = {}
    edge_target_dict = {}
    for edge in edges:
        if edge.source not in node_dict or edge.target not in node_dict:
            continue
        if edge.source not in edge_source_dict:
            edge_source_dict[edge.source] = []
        edge_source_dict[edge.source].append(edge)
        if edge.target not in edge_target_dict:
            edge_target_dict[edge.target] = []
        edge_target_dict[edge.target].append(edge)

    def recursive_propagate(node, active, child_req, visited):
        if node.id in visited:
            return node.requirement
        visited.add(node.id)
        parents = edge_target_dict.get(node.id, [])
        parents = [node_dict[edge.source] for edge in parents]
        if not active and node.requirement != -1:
            active = True
        if active and node.requirement == -1:
            node.requirement = child_req
        for parent in parents:
            req = recursive_propagate(parent, active, node.requirement, visited)
            if active:
                node.requirement = max(node.requirement, req)
        return node.requirement

    # Propagate requirements to the parent nodes 
    for subgraph in subgraphs.values():
        # get leaf nodes
        root_nodes = [node for node in subgraph if node.id not in edge_source_dict and node.type == 'micro-concept']

        for node in root_nodes:
            visited = set(node.id)
            _ = recursive_propagate(node, node.requirement != -1, node.requirement, visited) 

    return nodes

def main() -> None:
    '''
    This is the main function.

    It prompts the user for the subject, study level, and language, and then creates a graph from the output of the openAI model.
    '''
    subject = input("Enter subject/concept/topic: ")
    study_level = input("Enter education level: ")
    language = input("Enter output language: ")
    requirements_path = input("Enter the path to a csv file containing mandatory curriculum (optional): ")
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

    while True:
        response = chat_completion_request(params)
        response = response.choices[0].message.content
        if response.endswith("#END#"):
            response = response[:-5]
            break
        logging.info("First level response overflowed the token limit, retrying...")

    response = response.replace("\n", "")

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

    # Add grade requirements
    if requirements_path:
        nodes = add_requirements(language, requirements_path, nodes, edges)

    # Create .graphml and dashscape tree
    if not args.nooutput:
        create_dashscape_tree(nodes, edges, study_level, language)
        create_graphml_tree(nodes, edges)

if __name__ == "__main__":
    main()
