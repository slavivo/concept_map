import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import pickle
import argparse
import create_graph
from utils import Node

app = dash.Dash(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="The file to load the graph from (.pkl file)", required=True)
parser.add_argument("-p", "--port", help="The port to run the server on", default=8050)
args = parser.parse_args()

with open(args.file, 'rb') as f:
    major_node, graph, subgraphs = pickle.load(f)
    concepts = [graph[n['data']['id']]['nodes'] for n in major_node]
    concepts = [c['data'] for conc in concepts for c in conc]
    concepts = [Node(c['id'], c['label'], 'concept', 10) for c in concepts]

cyto.load_extra_layouts()

app.layout = html.Div([
    html.Button("Back", id='back-button', n_clicks=0),
    html.Button("Regenerate current subgraph", id='regenerate-button', n_clicks=0),
    html.Button("Save graph", id='save-button', n_clicks=0),
    html.Div(id="dynamic-input-area"),
    cyto.Cytoscape(
        id='cytoscape',
        elements=major_node,
        layout={
            'name': 'dagre',
        },
        style={'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'},
        stylesheet=[
            {'selector': '[requirement = ""]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px'}},
            {'selector': '[requirement = "o"]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px'}},
            {'selector': '[requirement = "6."]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px', 'background-color': '#90EE90'}},
            {'selector': '[requirement = "7."]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px', 'background-color': '#FFFFE0'}},
            {'selector': '[requirement = "8."]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px', 'background-color': '#FFA07A'}},
            {'selector': '[requirement = "9."]', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1', 'text-wrap': 'wrap', 'text-max-width': '30px', 'background-color': '#FF6347'}},
            {'selector': 'edge', 'style': {'curve-style': 'bezier', 'width': '0.5px', 'opacity': '0.5', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.5, 'target-arrow-color': '#000', 'line-color': '#000'}}
        ]
    ),
    dcc.Store(id='current-view', data={'level': 'major', 'parent': ''}),
    dcc.Store(id='current-major', data={'id': '', 'label': '', 'study_level': '', 'language': ''}),
    dcc.Store(id='next-elements', data=[]),
    dcc.Store(id='transition-phase', data='idle'),
    dcc.Interval(id='transition-interval', interval=100, n_intervals=0, max_intervals=10, disabled=True)
])

def regenerate_subgraph(current_view, current_major):
    # Get the index of the current concept (subgraph)
    for i, c in enumerate(concepts):
        if c.id == current_view['parent']:
            index = i
            break

    # Generate the subgraph
    nodes, edges = create_graph.process_message(
        index, 
        concepts, 
        current_major['label'], 
        current_major['study_level'], 
        current_major['language'], 
        "", 
        False
    )

    # Update the subgraphs
    subgraphs[current_view['parent']] = {
        'nodes': [{'data': {'id': n.id, 'label': n.label}} for n in nodes], 
        'edges': [{'data': {'source': e.source, 'target': e.target}} for e in edges]
    }

    # Get the new elements
    new_elements = subgraphs[current_view['parent']]['nodes'] + subgraphs[current_view['parent']]['edges']
    return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, dash.no_update, dash.no_update, 'fade-out', 0, False, new_elements


@app.callback(
    [Output('cytoscape', 'elements'),
     Output('cytoscape', 'style'),
     Output('current-view', 'data'),
     Output('current-major', 'data'),
     Output('transition-phase', 'data'),
     Output('transition-interval', 'n_intervals'),
     Output('transition-interval', 'disabled'),
     Output('next-elements', 'data')],
    [Input('cytoscape', 'tapNodeData'),
     Input('back-button', 'n_clicks'),
     Input('regenerate-button', 'n_clicks'),
     Input('save-button', 'n_clicks'),
     Input('transition-interval', 'n_intervals')],
    [State('current-view', 'data'),
     State('current-major', 'data'),
     State('next-elements', 'data'),
     State('transition-phase', 'data')]
)
def handle_cytoscape_interaction(node_data, back_clicks, regen_clicks, save_clicks, n_intervals, current_view, current_major, next_elements, transition_phase):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'save-button':
        with open(args.file, 'wb') as f:
            pickle.dump((major_node, graph, subgraphs), f)
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0, True, dash.no_update

    if trigger == 'regenerate-button':
        if current_view['level'] == 'micro':
            return regenerate_subgraph(current_view, current_major)
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, 0, True, dash.no_update

    if trigger == 'transition-interval':
        if transition_phase == 'fade-out':
            if n_intervals == 5:
                return next_elements, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, dash.no_update, dash.no_update, 'fade-in', 0, False, dash.no_update
        elif transition_phase == 'fade-in':
            if n_intervals == 5:
                return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'}, dash.no_update, dash.no_update, 'idle', 0, True, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, n_intervals + 1, False, dash.no_update

    elif trigger == 'back-button' and current_view['level'] != 'major':
        if current_view['level'] == 'standard':
            new_elements = major_node
            next_view = {'level': 'major', 'parent': ''}
        else:
            new_elements = graph[current_major['id']]
            next_view = {'level': 'standard', 'parent': ''}
        return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_view, dash.no_update, 'fade-out', 0, False, new_elements
    
    elif trigger == 'cytoscape' and node_data:
        node_id = node_data['id']

        if current_view['level'] == 'major':
            next_view = {'level': 'standard', 'parent': node_data['id']}
        else:
            next_view = {'level': 'micro', 'parent': node_data['id']}

        if node_id in graph:
            new_elements = graph[node_id]
            return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_view, node_data, 'fade-out', 0, False, new_elements
        elif node_id in subgraphs:
            new_elements = subgraphs[node_id]['nodes'] + subgraphs[node_id]['edges']
            return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_view, dash.no_update, 'fade-out', 0, False, new_elements
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, 'idle', 0, True, dash.no_update

if __name__ == '__main__':
    app.run_server(port=args.port, debug=True)
