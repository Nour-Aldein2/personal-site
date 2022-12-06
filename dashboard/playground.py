import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

graph = dcc.Graph(figure={
    'data': [
        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': 'SF'},
        {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
    ],
    'layout': {
        'title': 'Dash Data Visualization',
    },
}, responsive=True, className="h-100")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.Div("stuff", className="bg-secondary h-100"), width=2),
        dbc.Col([
            dbc.Row([
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")], width=4)
            ]),
            dbc.Row([
                dbc.Col([graph]),
            ]),
            dbc.Row([
                dbc.Col([graph]),
            ]),
        ], width=5),
        dbc.Col([
            dbc.Row([
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
                dbc.Col([html.Div("header", className="bg-primary")]),
            ]),
            dbc.Row([
                dbc.Col([graph]),
            ], className="h-100"),
        ], width=5),
    ])
], fluid=True)

app.run_server(debug=True)