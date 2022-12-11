import dash
import pandas as pd
from dash import html, dcc, Input, Output, State
from dash_iconify import DashIconify
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import lime
from lime.lime_text import LimeTextExplainer

import plotly.express as px
import plotly.graph_objects as go

import text_processing
import visualizations as vis

weights = None
markdown_text = '''
### Dash and Markdown

Dash apps can be written in Markdown.
Dash uses the [CommonMark](http://commonmark.org/)
specification of Markdown.
Check out their [60 Second Markdown Tutorial](http://commonmark.org/help/)
if this is your first introduction to Markdown!
'''

# Python
# print("Start app!")
df_train = text_processing.load_and_process_data("train.csv")
# print("Got train data")
df_test = text_processing.load_and_process_data("test.csv")
# print("Got test data")
class_names = ["Not Disaster", "Disaster"]
explainer = LimeTextExplainer(class_names=class_names, verbose=False)

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"
app = dash.Dash(external_stylesheets=["assets/typography.css", BS])
# print("created the app")

app.layout = dmc.NotificationsProvider(html.Div([dbc.Container([
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
                ), className=["pt-4", "mt-4"], fluid=True), width={"size": 5, "order": "first"}
        ),
        dbc.Col(
            html.Div([
                dcc.Graph(id="train-hist")
            ]), width={"size": 7, "order": "last"},
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
                ), className=["pt-4", "mt-4"], fluid=True), width={"size": 5, "order": "first"}
        ),
        dbc.Col(
            html.Div([
                dcc.Graph(id="test-hist")
            ]), width={"size": 7, "order": "last"},
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
                ], width={"order": "last"}),
            ], style={'padding': '1% 0'})
        ]),
        dbc.Col([
            html.Div(id="notify-container"),
            dcc.Graph(id="n-gram"),
        ], width={"order": "first"})
    ], style={'paddingTop': '5%'}),
    # ----------------------- Model
    dbc.Row([
        dmc.Button(['Train USE Model'],
                   id='train_model',
                   n_clicks=0,
                   disabled=False,
                   # color="primary",
                   # outline=True,
                   variant="gradient",
                   radius="xl",
                   gradient={"from": "teal", "to": "lime", "deg": 105}
                   ),
        dcc.Store(data=[], id='model', storage_type='local')
    ], style={'padding': '2%'}),
    # ----------------------- Model Predictions
    dbc.Row([
        dmc.Button(['Make Predictions'],
                   id='predict',
                   n_clicks=0,
                   disabled=False,
                   variant="gradient",
                   radius="xl",
                   gradient={"from": "teal", "to": "blue", "deg": 60},
                   ),
        dbc.Tooltip(
            "Make predictions for the validation data.",
            target=f"predict",
            placement="top",
        ),
        dcc.Store(data=[], id="predictions", storage_type='session')
    ], id='make-prediction', style={'display': 'none'}),
    # ----------------------- Understand Model
    dbc.Row([
        dmc.Chips(
            id='analyze-model',
            data=[
                {'value': 'history_df', 'label': 'History'},
                {'value': 'loss_graph', 'label': 'Loss Graph'},
                {'value': 'accuracy_graph', 'label': 'Accuracy Graph'},
                {'value': 'conf-mat', 'label': "Confusion Matrix"}
            ],
            value=None,
            variant='filled',
            spacing='xl',
            size='lg',
            radius='xl',
            color='green'
        )
    ], id='understand-model', style={'display': 'none'}),

    html.Div(id='display-training-results'),
    # ----------------------- LIME
    dbc.Row([
        html.H3("Explain The Predictions of USE Model"),
        dmc.Button(['Explain Random Prediction'],
                   id='lime-val-button',
                   n_clicks=0,
                   variant="gradient",
                   radius="md",
                   gradient={"from": "orange", "to": "lime", "deg": 105}
                   ),
        html.Div(id='show-lime-val'),
        html.Div(id="show-pred-truth")

    ], id='explain-preds', style={'display': 'none'})
    # ----------------------- Get any Tweet

    ## TODO: Choose a random sample of the val_data, make prediction, then explain that prediction
    ## TODO: Let the user enter a text, make prediction, then explain it
], className=["pt-5", "p-5", "main-container"], style={'backgroundColor': 'var(--bs-light)', })
]))


# ==========================================
# ==============================
# ===================


# Dataframes
@app.callback(
    Output(component_id='train-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_train = px.histogram(df_train, x=feature, marginal="box",
                             color_discrete_sequence=['#417767'], template='plotly_white')
    fig_train.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, b=0, t=10))
    fig_train.update_yaxes(visible=False, showticklabels=False)
    fig_train.update_xaxes(visible=False)
    return fig_train


@app.callback(
    Output(component_id='test-hist', component_property='figure'),
    Input(component_id='feature-selector', component_property='value')
)
def update_figure(feature):
    fig_test = px.histogram(df_test, x=feature, marginal="box",
                            color_discrete_sequence=['#774151'])
    fig_test.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                           margin=dict(l=0, r=0, b=15, t=0))
    fig_test.update_yaxes(visible=False, showticklabels=False)
    return fig_test


# N-Gram update notification
@app.callback(
    Output(component_id='notify-container', component_property='children'),
    [
        Input(component_id='phrases-num', component_property='value'),
        Input(component_id='lower-bound', component_property='value'),
        Input(component_id='upper-bound', component_property='value'),
        Input(component_id='n-gram-class', component_property='value')
    ]
)
def create_n_gram(phrases_num, lower, upper, class_name):
    notification = dbc.Alert(
        "Graph is updating, please wait.",
        id="alert-auto",
        is_open=True,
        color="warning",
        duration=4000,
    )
    return notification

# N-Gram
@app.callback(
    [
        Output(component_id='n-gram', component_property='figure'),
        Output(component_id='upper-bound', component_property='disabled'),
        Output(component_id='upper-bound', component_property='value'),
    ],
    [
        Input(component_id='phrases-num', component_property='value'),
        Input(component_id='lower-bound', component_property='value'),
        Input(component_id='upper-bound', component_property='value'),
        Input(component_id='n-gram-class', component_property='value')
    ]
)
def create_n_gram(phrases_num, lower, upper, class_name):
    if lower > upper:
        upper = lower
        target = text_processing.get_top_count_vectorizer(df_train,
                                                          df_train["text"],
                                                          ngram=(lower, upper),
                                                          n=phrases_num)  # target_0, target_1
        ## TODO: Fix condition for if the upper bound is larger the lower bound
        fig = px.bar(target[class_name],
                     labels={"index": "Phrase", "value": "Count"},
                     orientation='h',
                     color_discrete_sequence=['#2A5470'])
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, b=15, t=0))
        fig['layout']['yaxis']['autorange'] = "reversed"

        if class_name == 0:
            fig.update_layout(title="Class 0: Not a Disaster", title_font_color="green", title_x=0.5)
        else:
            fig.update_layout(title="Class 1: Disaster", title_font_color="red", title_x=0.5)
        return fig, True, upper
    else:
        target = text_processing.get_top_count_vectorizer(df_train,
                                                          df_train["text"],
                                                          ngram=(lower, upper),
                                                          n=phrases_num)  # target_0, target_1
        ## TODO: Fix condition for if the upper bound is larger the lower bound
        fig = px.bar(target[class_name],
                     labels={"index": "Phrase", "value": "Count"},
                     orientation='h',
                     color_discrete_sequence=['#2A5470'])
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, b=15, t=0))
        fig['layout']['yaxis']['autorange'] = "reversed"

        if class_name == 0:
            fig.update_layout(title="Class 0: Not a Disaster", title_font_color="green", title_x=0.5)
        else:
            fig.update_layout(title="Class 1: Disaster", title_font_color="red", title_x=0.5)
        return fig, False, upper


# USE Model
def prepare_data(df_train, df_test):
    # Split data into training and validation datasets
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(df_train["text"].to_numpy(),
                                                                                df_train["target"].to_numpy(),
                                                                                test_size=0.1,
                                                                                random_state=7)
    return train_sentences, val_sentences, train_labels, val_labels


@app.callback(
    [
        Output(component_id='model', component_property='data'),
        Output(component_id='train_model', component_property='disabled'),
        Output(component_id='train_model', component_property='children'),
        Output(component_id='train_model', component_property='variant'),
        Output(component_id='make-prediction', component_property='style')
    ],
    Input(component_id='train_model', component_property='n_clicks'),
    prevent_initial_call=True

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
        global weights
        weights = model.get_weights()
        return [model.to_json(), history.history], True, "Done!", 'outline', {'padding': '2%', 'display': 'block'}
    else:
        return [model.to_json()], True, "Done!", 'outline', {'padding': '2%', 'display': 'block'}


# Make Predictions
@app.callback(
    [
        Output(component_id='predictions', component_property='data'),
        Output(component_id='understand-model', component_property='style'),
        Output(component_id='explain-preds', component_property='style'),
        Output(component_id='predict', component_property='disabled'),
    ],
    [
        Input(component_id='predict', component_property='n_clicks'),
        Input(component_id='model', component_property='data'),
    ],
    prevent_initial_call=True
)
def make_prediction(n_clicks, model_data):
    if n_clicks > 0:
        train_sentences, val_sentences, train_labels, val_labels = prepare_data(df_train, df_test)
        model = tf.keras.models.model_from_json(model_data[0], custom_objects={'KerasLayer': hub.KerasLayer})
        model.set_weights(weights)
        pred_probs = model.predict(val_sentences)
        preds = tf.argmax(pred_probs, axis=1).numpy()
        return {'predictions': preds.tolist()}, {'padding': '2%', 'display': 'block'}, {'padding': '2%',
                                                                                        'display': 'block'}, True
    else:
        return (), {'display': 'none'}, {'display': 'none'}, False


# Analyze Model
@app.callback(
    Output(component_id='display-training-results', component_property='children'),
    [
        Input(component_id='analyze-model', component_property='value'),
        Input(component_id='model', component_property='data'),
        Input(component_id='predictions', component_property='data')
    ],
    prevent_initial_call=True
)
def analyize_model(value, model_data, preds):
    train_sentences, val_sentences, train_labels, val_labels = prepare_data(df_train, df_test)

    if value == 'history_df':
        history = pd.DataFrame(model_data[1])
        return dbc.Table.from_dataframe(history)
    elif value == 'loss_graph':
        history = pd.DataFrame(model_data[1])
        fig = px.line(history, y=['loss', 'val_loss'], x=history.index,
                      title="Loss",
                      labels={"index": "Epoch",
                              "value": "Loss",
                              "variable": "Function"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        return dcc.Graph(figure=fig)
    elif value == 'accuracy_graph':
        history = pd.DataFrame(model_data[1])
        fig = px.line(history, y=['accuracy', 'val_accuracy'], x=history.index,
                      title="Accuracy",
                      labels={"index": "Epoch",
                              "value": "Accuracy",
                              "variable": "Function"})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        return dcc.Graph(figure=fig)
    elif value == 'conf-mat':
        cm_df = pd.DataFrame(data=confusion_matrix(y_true=val_labels, y_pred=np.array(preds["predictions"])),
                             columns=class_names,
                             index=class_names
                             )

        fig = px.imshow(cm_df,
                        text_auto=True,
                        color_continuous_scale=px.colors.sequential.Blues)
        fig.update_layout(xaxis=dict(tickfont=dict(size=10), tickmode="linear"),
                          yaxis=dict(tickfont=dict(size=10), tickmode="linear"),
                          title="Tweets Classification Confusion Matrix",
                          paper_bgcolor='rgba(0,0,0,0)'
                          )
        fig.update_traces(hovertemplate="<br>".join([
            "Predicted label: %{x}",
            "True label: %{y}",
            "Preds count: %{z}"
        ])
        )
        return dcc.Graph(figure=fig)


# Explain Predictions on Val
@app.callback(
    [
        Output(component_id='show-lime-val', component_property='children'),
        Output(component_id='show-pred-truth', component_property='children')
    ],
    [
        Input(component_id='lime-val-button', component_property='n_clicks'),
        Input(component_id='model', component_property='data'),
        Input(component_id='predictions', component_property='data')
    ],
    prevent_initial_call=True
)
def explain_val(n_clicks, model_data, preds):
    if n_clicks > 0:
        train_sentences, val_sentences, train_labels, val_labels = prepare_data(df_train, df_test)
        model = tf.keras.models.model_from_json(model_data[0], custom_objects={'KerasLayer': hub.KerasLayer})
        model.set_weights(weights)
        preds = preds["predictions"]
        rng = np.random.RandomState(n_clicks + 132)
        idx = rng.choice(range(len(preds)))
        explain_pred = explainer.explain_instance(val_sentences[idx],
                                                  classifier_fn=model.predict,
                                                  labels=[val_labels[idx]])
        if val_labels[idx] == preds[idx]:
            return html.Iframe(
                srcDoc=explain_pred.as_html(),
                width='100%',
                height='400px',
                style={'border': '2px #d3d3d3 solid'},
            ), dmc.Notification(
                id="better-notify",
                title=f"Prediction: {class_names[preds[idx]]}",
                message=[f"True Label: {class_names[val_labels[idx]]}"],
                action="show",
                color='green',
                icon=[DashIconify(icon="akar-icons:circle-check")],
            )
        else:
            return html.Iframe(
                srcDoc=explain_pred.as_html(),
                width='100%',
                height='400px',
                style={'border': '2px #d3d3d3 solid'},
            ), dmc.Notification(
                id="better-notify",
                title=f"Prediction: {class_names[preds[idx]]}",
                message=[f"True Label: {class_names[val_labels[idx]]}"],
                action="show",
                color='red',
                icon=[DashIconify(icon="akar-icons:circle-x")],
            )
    else:
        return "", ""


app.run(host="0.0.0.0", port="8050")

## TODO: run the app with multiple workers so the callbacks can b executed in parallel
