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
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


import lime
from lime.lime_text import LimeTextExplainer

import plotly.express as px

import text_processing

weights = None
md_project_intro = '''
The goal of this dashboard is to provide an insight on how does an LSTM layer make predictions for \
classification tasks. The task used in this dashboard is the famouse Kaggle cometition; classifying tweets \
into disaster and not disaster. \n We start by vectorizing the text data using \
`tf.keras.layers.TextVectorization` where we choose the size of the courpus is set to be 10000, and the \
maximum length of the sequence length is 15 tokens. Finally, we fit the vectorizer to the training data \
before we are able to use in the model. The next step is to prepare an embedding layer using \
`tf.keras.layers.Embedding`.
The user is able to view the data in tabular form, and see the distribution of different features extracted \
from the text.
'''
md_n_gram = '''
N-Gram takes sequence data (one word or more) as input, it then creates probablitiy distrubution of \
the all the possible items, and then make a prediction based on the likelihood of each item. \
In addiction to next-word prediction, N-Grams have other applications, such as language identification, \
information retrieval, and predictions in DNA sequencing.\
[[1]](https://deepai.org/machine-learning-glossary-and-terms/n-gram)
'''
md_model_description = '''
We Design the model used in this dashboard using a vectorizer,an embedding layer, and LSTM layer, and an \
output layer with two units and `softmax` as its activation function. Notice that we are using two units \
and softmax instead of one layer and sigmoid activation function to be able to use the model to predictions \
and compare them in a LIME instance. As a consequence of using classification layer in this way, the \
loss function will Sparse Categorical Crossentropy instead of Binary Crossentropy.

You will be able to train the model once, and then explore it's predictions, and the training process \
after clicking on **Train LSTM Button**. In other words, this button will be disabled and a new one will \
appear allowing the use to make predictions on the validation data. Once you click on 'Make Predictions' \
button, more buttons will appear, each one will allow you to explore what the button's labels indicates.

The last button that will appear is **Explain Predictions with LIME**, which you will be able to click on as \
many times as you need. When you click on this button, the model will make a prediction and compare it with \
the ground truth label. A notification will appear at the lower left corner of the window showing if the \
model made a correct prediction or if it's miss classified it. Underneath the button you will see which words\
affected the decision of the model, which will help you decide if you can trust such a model. Every time \
you click on this button, a random prediction will be generated.
'''

# Python
# print("Start app!")
df_train = text_processing.load_and_process_data("train.csv")
# print("Got train data")
df_test = text_processing.load_and_process_data("test.csv")
dropdown_ids_lst = df_test.select_dtypes(include=np.number).columns[1:].to_list()
dropdown_ids = [{"label": " ".join(i.split("_")).title(), "value": i} for i in dropdown_ids_lst]
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
        dcc.Markdown(md_project_intro),
        dcc.Dropdown(
            dropdown_ids,  # Don't include ID
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
            html.Div([
                dmc.Button(
                    "Show Five Random Tweets",
                    id="show-random-train",
                    color="gray",
                    fullWidth=True
                ),
                html.Div(id='train-texts', style={'padding': '6% 0'})
            ], style={"size": 6, 'padding': '3% 0'}),
        ),
        dbc.Col(
            html.Div([
                dcc.Graph(id="train-hist")
            ]), width={"size": 6, "order": "last"},
        )
    ]),
    # ----------------------- N-Gram
    dbc.Row([
        dbc.Col([
            dcc.Markdown(md_n_gram, style={'padding': '0 3%', 'textAlign': 'justify'}),
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
            dmc.LoadingOverlay(
                dcc.Graph(id="n-gram"),
                loaderProps={"variant": "bars", "color": "blue", "size": 100}
            ),
        ], width={"order": "first"})
    ], style={'paddingTop': '5%'}),
    # ----------------------- Model
    dbc.Row([
        dcc.Markdown(md_model_description),
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
                   loading=True,
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
                             color_discrete_sequence=['#417767'],
                             labels={feature: " ".join(feature.split("_")).title(), "count": "Count"})
    fig_train.update_layout(height=600, paper_bgcolor='rgba(0,0,0,0)', # plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, b=0, t=10),
                            bargap=0.2,)
    fig_train.update_yaxes(ticklabelposition="inside top")
    return fig_train


@app.callback(
    [
        Output(component_id="train-texts", component_property="children"),
        Output(component_id="show-random-train", component_property="loading")
    ],
    Input(component_id="show-random-train", component_property="n_clicks"),
    prevent_initial_call=True
)
def get_train_texts(n_clicks):
    if n_clicks > 0:
        num = 5
        samples = []
        tweets = df_train.sample(n=num)
        for i in range(num):
            samples.append(f"**Random tweet {i+1}**: {tweets['text'].iloc[i]}")
        markdown_train = "\n\n".join(samples)
        return dcc.Markdown(markdown_train), False
    else:
        return "", False


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
        fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', #plot_bgcolor='rgba(0,0,0,0)',
                          margin=dict(l=0, r=0, b=15, t=0))
        fig['layout']['yaxis']['autorange'] = "reversed"

        if class_name == 0:
            fig.update_layout(title="Class 0: Not a Disaster", title_font_color="green", title_x=0.5)
        else:
            fig.update_layout(title="Class 1: Disaster", title_font_color="red", title_x=0.5)
        return fig, False, upper


# LSTM Model
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
    print("Preparing data")
    train_sentences, val_sentences, train_labels, val_labels = prepare_data(df_train, df_test)
    print("Vectorizing text")
    # Setup text vectorization variables
    max_vocab_len = 10000  # max number of words to have in our vocabulary
    max_length = 15  # max length our sequence will be (e.g. how many words from a tweet does a model see?)

    text_vectorizer = TextVectorization(max_tokens=max_vocab_len,
                                        output_mode="int",
                                        output_sequence_length=max_length)
    # Fit the text vectroizer to the training text
    text_vectorizer.adapt(train_sentences)
    # Embedding
    print("Getting Embeddings")
    embedding = tf.keras.layers.Embedding(input_dim=max_vocab_len,  # set input shape
                                 embeddings_initializer="uniform",
                                 output_dim=128,  # The output shape: GPUs work better with numbers divisable by 8
                                 input_length=max_length  # how long is each input
                                 )
    print("Creating model")
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = embedding(x)
    x = tf.keras.layers.LSTM(64)(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.models.Model(inputs, outputs)
    if n_clicks > 0:
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=["accuracy"])
        history = model.fit(
            train_sentences, train_labels,
            epochs=3,
            batch_size=32,
            validation_data=(val_sentences, val_labels)
        )
        print("Getting weights...")
        global weights
        weights = model.get_weights()
        history = history.history
        return [model.to_json(), history], True, "Done!", 'outline', {'padding': '2%', 'display': 'block'}
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
        Output(component_id='show-pred-truth', component_property='children'),
        Output(component_id='lime-val-button', component_property='loading')
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
        model = tf.keras.models.model_from_json(model_data[0])
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
            ), False
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
            ), False
    else:
        return "", "", False


app.run(host="0.0.0.0", port="7777")