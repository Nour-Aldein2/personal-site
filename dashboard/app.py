import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd

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

## Python 
df_train = text_processing.load_and_process_data("train.csv")
df_test = text_processing.load_and_process_data("test.csv")


# Dashboard
def generate_table(dataframe, max_rows=10):
    return dash.dash_table.DataTable(
        data=dataframe.to_dict('records'),
        columns=[{'id': c, 'name': c} for c in dataframe.columns],

        style_cell_conditional=[
            {
                'if': {'column_id': c},
                'textAlign': 'left',
            } for c in ['id', 'text']
        ],
        style_data={
            'color': 'black',
            'textAlign': 'left',
            'overflow': 'auto',
            'backgroundColor': 'white',
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'textAlign': 'left',
                'backgroundColor': 'rgb(245,245,245)',
            }
        ],
        style_header={
            'backgroundColor': 'rgb(224,224,224)',
            'color': 'black',
            'textAlign': 'left',
            'fontWeight': 'bold',
            'position':'sticky',
            'top': '0'
        },
        style_table={
            'height': 400,
            'overflowY': 'scroll',
            'overflowX': 'scroll',
            'width': 'relative'
        }
    )


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


feature = "char_count"

fig_train = px.histogram(df_train, x=feature, marginal="box", color_discrete_sequence=['turquoise'])
fig_train.update_layout(height=600)
fig_test = px.histogram(df_test, x=feature, marginal="box", color_discrete_sequence=['indianred'])
fig_test.update_layout(height=600)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}


app.layout = html.Div(children=[

    html.H1(
        children='Explaining USE Predictions with LIME',
        style={
            "textAlign": "center",
            "color": "#2BBBB" #colors["text"]
        }
    ),

    html.H2("Overview"),
    dcc.Markdown(children=markdown_text),

    # Training Data
    html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Container(
                    dbc.Card(
                        html.Div(
                            dbc.Table.from_dataframe(df_train[["id","text","target"]]),
                            style={"maxHeight": "450px", "overflow": "scroll"},
                        ),
                        body=True,
                    ), 
                    className="p-5",
                ), width={"size":6, "order":"first"}
            ),
            dbc.Col(
                html.Div([
                    dcc.Graph(
                        id="Train Data",
                        figure=fig_train,
                        # style={'padding':10}
                    )
                ]), width={"size":6, "order":"last"},
            )
        ]),
    ],
    ),


    html.Div([
        dbc.Row([
            dbc.Col(
                dbc.Container(
                    dbc.Card(
                        html.Div(
                            dbc.Table.from_dataframe(df_test[["id","text"]]),
                            style={"maxHeight": "450px", "overflow": "scroll"},
                        ),
                        body=True,
                    ), 
                    className="p-5",
                ), width={"size":6, "order":"first"}
            ),
            dbc.Col(
                html.Div([
                    dcc.Graph(
                        id="Test Data",
                        figure=fig_test,
                        # style={'padding':10}
                    )
                ]), width={"size":6, "order":"last"},
            )
        ]),
    ],
    ),


    


# Input yout text
    html.Div([
        html.Label('Text Input'),
        dcc.Input(value='MTL', type='text'),
    ]),

], style={"padding":"5%"})

if __name__ == "__main__":
    app.run(debug=True)