import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_addons as tfa
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

import plotly.express as px
from plotly.io.json import to_json_plotly

import text_processing
import visualizations as vis

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
    ## TODO: Add functionality to retrieve the data if there is an interesting point
    ## TODO: Make the histograms visually more appealing
    ## TODO: Add titles for each section of the dashboard (training, test, n-gram)
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
    # ----------------------- N-Gram
    dbc.Row([
        dbc.Col([
            dcc.Markdown('''
                N-Gram takes sequence data (one word or more) as input, it then creates probablitiy distrubution of \
                the all the possible items, and then make a prediction based on the likelihood of each item. \
                In addiction to next-word prediction, N-Grams have other applications, such as language identification, \
                information retrieval, and predictions in DNA sequencing.\
                [[1]](https://deepai.org/machine-learning-glossary-and-terms/n-gram)
            ''', style={'padding': '0 3%', 'textAlign': 'justify'}),
            html.H6(["Choose The Class:"], style={'padding': '0 3%'}),
            dbc.RadioItems(
                options=[
                    {"label": "Disaster", "value": 1},
                    {"label": "Not Disaster", "value": 0},
                    # {"label": "Disabled Option", "value": 3, "disabled": True},
                ],
                value=1,
                id="n-gram-class",
                switch=True,
                inline=True,
                style={'padding': '0 3%'}
            ),
            html.H6(["Number of Phrases:"], style={'padding': '0 3%'}),
            dcc.Slider(3, 20, 1,
                       value=5,
                       id="phrases-num"),
            dbc.Row([
                dbc.Col([
                    html.H6(["Lower Boundary:"], style={'padding': '0 7%'}),
                    dcc.Slider(1, 5, 1,
                               value=1,
                               id="lower-bound"),
                ]),
                dbc.Col([
                    html.H6(["Upper Boundary:"], style={'padding': '0 7%'}),
                    dcc.Slider(1, 7, 1,
                               value=1,
                               id="upper-bound")
                ]),
            ], style={'padding': '1% 0'})
        ]),
        dbc.Col(
            dcc.Graph(id="n-gram")
        )
    ]),

    dbc.Row([
        dbc.Button(['Train USE Model',
                    # dbc.Spinner(size='sm', color="danger", type="grow"), " Training..."
                    dcc.Store(data=[], id="model", storage_type='local')],
                   id='train_model',
                   n_clicks=0,
                   color="primary",
                   className="me-1"
                   ),
    ], style={'padding': '2%'}),
    # dbc.Row([
    #     dbc.Spinner([html.Div(id='training-in-progress')], color="danger", type="grow")
    # ], style={'padding': '2%'})
], className=["pt-5", "p-5"], style={'backgroundColor': 'var(--bs-light)', })


# Dataframes
@app.callback(
    Output(component_id='train-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_train = px.histogram(df_train, x=feature, marginal="box",
                             color_discrete_sequence=['turquoise'])
    fig_train.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)')
    return fig_train


@app.callback(
    Output(component_id='test-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_test = px.histogram(df_test, x=feature, marginal="box",
                            color_discrete_sequence=['indianred'])
    fig_test.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)')
    return fig_test


# N-Gram
@app.callback(
    Output(component_id='n-gram', component_property='figure'),
    [Input(component_id='phrases-num', component_property='value'),
     Input(component_id='lower-bound', component_property='value'),
     Input(component_id='upper-bound', component_property='value'),
     Input(component_id='n-gram-class', component_property='value')]
)
def create_n_gram(phrases_num, lower, upper, class_name):
    target = text_processing.get_top_count_vectorizer(df_train,
                                                      df_train["text"],
                                                      ngram=(lower, upper),
                                                      n=phrases_num) # target_0, target_1
    ## TODO: Fix condition for if the upper bound is larger the lower bound
    fig = px.bar(target[class_name],
                 labels={"index": "Phrase", "value": "Count"})
    fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)')
    if class_name == 0:
        fig.update_layout(title="Class 0: Not a Disaster", title_font_color="green", title_x=0.5)
    else:
        fig.update_layout(title="Class 1: Disaster", title_font_color="red", title_x=0.5)

    return fig

# USE Model
def prepare_data(df_train, df_test):
    # Split data into training and validation datasets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df_train["text"].to_numpy(),
                                                                                df_train["target"].to_numpy(),
                                                                                test_size=0.1,
                                                                                random_state=7)
    return train_sentences, val_sentences, train_labels, val_labels

@app.callback(
    Output(component_id='model', component_property='data'),
    Input(component_id='train_model', component_property='n_clicks')
)
def create_model(n_clicks):
    # Callbacks
    early_stop = EarlyStopping(monitor="val_loss",
                               verbose=1,
                               patience=3)
    print("loading model!")
    # Create a keras layer using USE pretrained layer from TF Hub
    sentence_encoder_use = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                          input_shape=[],
                                          dtype=tf.string,
                                          trainable=False,
                                          name="USE")
    print("USE loaded")
    # Create model using the Sequential API
    model = tf.keras.models.Sequential([
        sentence_encoder_use,
        layers.Dense(64, activation="relu"),
        layers.Dense(2, activation="softmax", name="output_layer")
    ], name="baseline_USE")

    # Compile the model
    model.compile(loss="sparse_categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])

    # Get data
    train_sentences, val_sentences, train_labels, val_labels = prepare_data(df_train, df_test)
    # Fit the baseline model
    if n_clicks > 0:
        history = model.fit(train_sentences,
                          train_labels,
                          epochs=30,
                          validation_data=(val_sentences, val_labels),
                          callbacks=[early_stop])

    return model.to_json()


app.run(debug=True)

## TODO: run the app with multiple workers so the callbacks can b executed in parallel
