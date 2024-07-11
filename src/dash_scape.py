import dash
from dash import dcc, html, Input, Output, State
import dash_cytoscape as cyto
import pickle
import argparse

app = dash.Dash(__name__)

# add argument -f to specify the file

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="The file to load the graph from (.pkl file)", required=True)
args = parser.parse_args()

with open(args.file, 'rb') as f:
    main_nodes, main_edges, subgraphs = pickle.load(f)

app.layout = html.Div([
    html.Button("Back", id='back-button', n_clicks=0),
    cyto.Cytoscape(
        id='cytoscape',
        elements=main_nodes + main_edges,
        layout={
            'name': 'cose',
            'idealEdgeLength': 100,
            'nodeOverlap': 10,
            'nodeRepulsion': 80000,
        },
        style={'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'},
        stylesheet=[
            {'selector': 'node', 'style': {'label': 'data(label)', 'width': '10px', 'height': '10px', 'font-size': '10px', 'opacity': '1'}},
            {'selector': 'edge', 'style': {'curve-style': 'bezier', 'width': '0.5px', 'opacity': '0.3', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.3, 'target-arrow-color': '#000', 'line-color': '#000'}}
        ]
    ),
    dcc.Store(id='current-node', data='main'),
    dcc.Store(id='next-elements', data=[]),
    dcc.Store(id='transition-phase', data='idle'),
    dcc.Interval(id='transition-interval', interval=100, n_intervals=0, max_intervals=10, disabled=True)
])

@app.callback(
    [Output('cytoscape', 'elements'),
     Output('cytoscape', 'style'),
     Output('current-node', 'data'),
     Output('transition-phase', 'data'),
     Output('transition-interval', 'n_intervals'),
     Output('transition-interval', 'disabled'),
     Output('next-elements', 'data')],
    [Input('cytoscape', 'tapNodeData'),
     Input('back-button', 'n_clicks'),
     Input('transition-interval', 'n_intervals')],
    [State('current-node', 'data'),
     State('next-elements', 'data'),
     State('transition-phase', 'data')]
)
def handle_cytoscape_interaction(node_data, n_clicks, n_intervals, current_node, next_elements, transition_phase):
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'transition-interval':
        if transition_phase == 'fade-out':
            if n_intervals == 5:
                return next_elements, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, dash.no_update, 'fade-in', 0, False, dash.no_update
        elif transition_phase == 'fade-in':
            if n_intervals == 5:
                return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '1', 'transition': 'opacity 0.5s ease'}, dash.no_update, 'idle', 0, True, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, n_intervals + 1, False, dash.no_update

    if trigger == 'back-button' and current_node != 'main':
        new_elements = main_nodes + main_edges
        next_node = 'main'
        return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, next_node, 'fade-out', 0, False, new_elements
    elif trigger == 'cytoscape' and node_data:
        new_node = node_data['id']
        if new_node in subgraphs:
            new_elements = subgraphs[new_node]['nodes'] + subgraphs[new_node]['edges']
            return dash.no_update, {'width': '100%', 'height': '400px', 'opacity': '0', 'transition': 'opacity 0.5s ease'}, new_node, 'fade-out', 0, False, new_elements
    return dash.no_update, dash.no_update, dash.no_update, 'idle', 0, True, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)
