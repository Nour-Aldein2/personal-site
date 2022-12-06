import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np

import plotly.express as px

import text_processing

markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

# Python
print("Start app!")
df_train = text_processing.load_and_process_data("train.csv")
print("Got train data")
df_test = text_processing.load_and_process_data("test.csv")
print("Got test data")

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[BS])
print("created the app")

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Explaining USE with LIME"), width=12)
    ]),
    dbc.Row([
        dcc.Dropdown(
            df_test.select_dtypes(include=np.number).columns[1:],  # Don't include ID
            "char_count",
            id="feature-selector"
        ),
    ]),
    # ----------------------- Training Data
    dbc.Row([
        dbc.Col(
            dbc.Container(
                dbc.Card(
                    html.Div(
                        dbc.Table.from_dataframe(
                            df_train[["id", "text", "target"]]),
                        style={"maxHeight": "450px", "overflow": "scroll"},
                    ), body=True,
                ), className="p-5"), width={"size": 6, "order": "first"}
        ),
        dbc.Col(
            html.Div([
                dcc.Graph(id="train-hist")
            ]), width={"size": 6, "order": "last"},
        )
    ]),
    # ----------------------- Testing Data
    dbc.Row([
        dbc.Col(
            dbc.Container(
                dbc.Card(
                    html.Div(
                        dbc.Table.from_dataframe(df_test[["id", "text"]]),
                        style={"maxHeight": "450px", "overflow": "scroll"},
                    ), body=True,
                ), className="p-5"), width={"size": 6, "order": "first"}
        ),
        dbc.Col(
            html.Div([
                dcc.Graph(id="test-hist")
            ]), width={"size": 6, "order": "last"},
        )
    ]),
], className="m-5", style={'backgroundColor': 'var(--bs-light)', })

@app.callback(
    Output(component_id='train-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_train = px.histogram(df_train, x=feature, marginal="box",
                             color_discrete_sequence=['turquoise'])
    fig_train.update_layout(height=600)
    return fig_train


@app.callback(
    Output(component_id='test-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_test = px.histogram(df_test, x=feature, marginal="box",
                            color_discrete_sequence=['indianred'])
    fig_test.update_layout(height=600)
    return fig_test


app.run(debug=True)
