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
df_train = text_processing.load_and_process_data("train.csv")
df_test = text_processing.load_and_process_data("test.csv")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children={

    html.H1(
        children='Explaining USE Predictions with LIME',
        style={
            "textAlign": "center",
            "color": "#2BBBB"  # colors["text"]
        }
    ),

    # html.H2("Overview"),
    # dcc.Markdown(children=markdown_text),
    # dcc.Dropdown(
    #     df_test.select_dtypes(include=np.number).columns[1:],  # Don't include ID
    #     "char_count",
    #     id="feature-selector"
    # ),

    # # ----------------------- Training Data
    # html.Div([
    #     dbc.Row([
    #         dbc.Col(
    #             dbc.Container(
    #                 dbc.Card(
    #                     html.Div(
    #                         dbc.Table.from_dataframe(
    #                             df_train[["id", "text", "target"]]),
    #                         style={"maxHeight": "450px", "overflow": "scroll"},
    #                     ),
    #                     body=True,
    #                 ),
    #                 className="p-5",
    #             ), width={"size": 6, "order": "first"}
    #         ),
    #         dbc.Col(
    #             html.Div([
    #                 dcc.Graph(id="train-hist")
    #             ]), width={"size": 6, "order": "last"},
    #         )
    #     ]),
    # ],
    # ),

    # # ----------------------- Testing Data
    # html.Div([
    #     dbc.Row([
    #         dbc.Col(
    #             dbc.Container(
    #                 dbc.Card(
    #                     html.Div(
    #                         dbc.Table.from_dataframe(df_test[["id", "text"]]),
    #                         style={"maxHeight": "450px", "overflow": "scroll"},
    #                     ),
    #                     body=True,
    #                 ),
    #                 className="p-5",
    #             ), width={"size": 6, "order": "first"}
    #         ),
    #         dbc.Col(
    #             html.Div([
    #                 dcc.Graph(id="test-hist")
    #             ]), width={"size": 6, "order": "last"},
    #         )
    #     ]),
    # ],
    # ),


    # Input your text
    html.Div([
        html.Label('Text Input'),
        dcc.Input(value='MTL', type='text'),
    ]),
}, style={"padding": "5%"})


# @app.callback(
# #     Output(component_id='train-hist', component_property='figure'),
# #     Input(component_id='feature-selector', component_property='value')
# # )
# # def update_figure(feature):
# #     fig_train = px.histogram(df_train, x=feature, marginal="box",
# #                              color_discrete_sequence=['turquoise'])
# #     fig_train.update_layout(height=600)
# #     return fig_train


# # @app.callback(
# #     Output(component_id='test-hist', component_property='figure'),
# #     Input(component_id='feature-selector', component_property='value')
# # )
# # def update_figure(feature):
# #     fig_test = px.histogram(df_test, x=feature, marginal="box",
# #                             color_discrete_sequence=['indianred'])
# #     fig_test.update_layout(height=600)
# #     return fig_test




if __name__ == "__main__":
    app.run(debug=True)
