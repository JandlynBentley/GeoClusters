"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021
This program is the primary program for the GeoClusters Dash web app.
Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and (eventually) a clustering algorithm comparison component.
"""

import base64
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import pandas as pd
from sklearn.cluster import KMeans
import plotly.graph_objs as go
import itertools
from sklearn import mixture

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
markdown_text1 = '''
GeoClusters is a visual tool for geoscientists to view their data under various clustering algorithms in real time.

To use the clustering comparative tool:
&nbsp;

    * Drag and drop or select a preprocessed CSV or Excel file in the blue upload area. 
    * This file should have all preceding and trailing comments removed so only the column names and column data remain.
    * Choose from the drop down menus to compare clustering algorithms on your data.

The code for this open-source tool can be found on [Github](https://github.com/JandlynBentley/GeoClusters).
'''
axes_options = []
data = []


def main():
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    # Div 0: holds everything on the web page
    app.layout = html.Div([

        # Div 1: Holds Title, Sub-title, and instructions
        html.Div([
            html.H1(
                children='GeoClusters',
                style={
                    'color': 'green',
                    'textAlign': 'center'
                }
            ),
            html.H3(
                children='A Comparative Cluster Analysis Tool',
                style={
                    'textAlign': 'center'
                }
            ),
            html.Br(),

            # Instructions to the user displayed with markdown
            dcc.Markdown(
                children=markdown_text1,
                style={
                    'textAlign': 'left',
                    'width': '50%'
                }
            ),
            html.Br(),
        ]),  # Div 1 ends

        # ************************************************************************************************
        # Div 2: Holds The comparative tool
        html.Div([

            # Div 2.5: Holds all the interactive components and graph (left and right side)
            html.Div([

                # LEFT SIDE
                html.Div([
                    # container for the left interactive elements
                    html.Div(
                        id='output-dropdown-area1',
                        style={'width': '50%'}
                    )
                ]),
                html.Br(),

                # Container for the left graph
                html.Div([
                    html.Div(
                        id='output-graph-area1',
                        style={'width': '50%',
                               'display': 'inline-block'},
                    )
                ]),
                html.Br(),

                # RIGHT SIDE
                html.Div([
                    # container for the right interactive elements
                    html.Div(
                        id='output-dropdown-area2',
                        style={'width': '50%'}
                    )
                ]),
                html.Br(),

                # Container for the right graph
                html.Div([
                    html.Div(
                        id='output-graph-area2',
                        style={'width': '50%',
                               'display': 'inline-block'},
                    )
                ]),
                html.Br(),

            ],
                # Set all of the interactive components and graphs into two columns
                style={
                    'columnCount': 2,
                    'columnGap': '100px'
                }
            )

        ]),
        # ************************************************************************************************

        # Div 3: Holds the upload tool
        html.Div([
            # Container for file upload component
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files'),
                    html.Center()
                ]),
                style={
                    'width': '50%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '100px',
                    'textAlign': 'center',
                    'margin-top': '10px',
                    'margin-left': '25%',
                    'margin-bottom': '10px',
                    'backgroundColor': '#d6f5f5',
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Br()
        ])

    ])  # end of app layout

    # Given the data set from Upload, parse the data into a data frame and collect axis options
    # Also return axis dropdowns for x, y, and z
    @app.callback(Output('output-dropdown-area1', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_input_dropdown(list_of_contents, list_of_names, list_of_dates):

        if list_of_contents is not None:
            data_frame = [parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)]
            '''
            The parsing function takes in a data set, collects and stores the attributes in a global variable
            to be used for the axes' dropdown options, and returns the data set as a Pandas data frame.
            Confirmed that the parsing function returns a data frame as needed if it's indexed.
            '''

            # Left side
            alg1 = html.Div(alg_selection1())  # Algorithm selection dropdown
            graph2d3d_1 = html.Div(graph_2d3d_selection1())  # 2D or 3D graph selection radio buttons
            clusters1 = num_clusters_selection1()  # Number of predicted clusters via radio buttons
            axes1 = html.Div(axes_selection_xyz_1())

            # Right side
            alg2 = html.Div(alg_selection2())  # Algorithm selection dropdown
            graph2d3d_2 = html.Div(graph_2d3d_selection2())  # 2D or 3D graph selection radio buttons
            clusters2 = num_clusters_selection2()  # Number of predicted clusters via radio buttons
            axes2 = html.Div(axes_selection_xyz_2())

            children = [
                # LEFT SIDE -------
                html.H5(
                    children='--- First Algorithm ---',
                    style={
                        'textAlign': 'center'
                    }
                ),

                alg1,
                graph2d3d_1,
                clusters1,
                axes1,

                html.Br(),
                html.Br(),

                html.H5(
                    children='---------------------------------------------------------------------------------------',
                    style={
                        'textAlign': 'left'
                    }
                ),

                html.Br(),

                # RIGHT SIDE -------
                html.H5(
                    children='--- Second Algorithm ---',
                    style={
                        'textAlign': 'center'
                    }
                ),

                alg2,
                graph2d3d_2,
                clusters2,
                axes2,
            ]

            return children

    # LEFT GRAPH -----------------------------------------------------------------------------------------------

    # Once x, y, z axes have been chosen, output a scatter plot to graph 1
    @app.callback(Output('output-graph-area1', 'children'),
                  Input("dropdown_algorithm1", "value"),
                  Input("2d3d_graph1", "value"),
                  Input('dd_x_1', 'value'),
                  Input('dd_y_1', 'value'),
                  Input('dd_z_1', 'value'),
                  Input('clusters_selector1', 'value'))
    def update_output_graph1(algorithm, choice2d3d, x, y, z, clusters):

        print("Algorithm is: " + str(algorithm))
        print("Choice of 2D or 3D is: " + str(choice2d3d))
        print("x is: " + str(x))
        print("y is: " + str(y))
        print("z is: " + str(z))
        print("Number of clusters: " + str(clusters))

        alg_bool = algorithm is not None
        x_bool = x is not None
        y_bool = y is not None
        z_bool = z is not None

        # Make a 2D graph
        if alg_bool and (choice2d3d == '2D') and x_bool and y_bool:

            if algorithm == 'K-Means':
                print("TEST!!!!! A K-Means 2D Graph will be made.")
                fig = make_k_means_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    ),
                ]
            elif algorithm == 'GMM':
                print("TEST!!!!! A GMM 2D Graph will be made.")
                fig = make_gmm_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    )
                ]
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (1).")
                ]

        # Make a 3D graph
        elif alg_bool and (choice2d3d == '3D') and x_bool and y_bool and z_bool:

            print("A Graph for " + str(algorithm) + " will be made:")

            if algorithm == 'K-Means':
                print("A K-Means 3D Graph will be made.")
                fig = make_k_means_3d_graph(data[0], x, y, z, clusters)

                children = [
                    html.Br(),
                    html.H5("A K-Means 3D Graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    ),
                ]
            elif algorithm == 'GMM':
                print("TEST!!!!! GMM 3D Graph is not available yet.")
                fig = make_gmm_3d_graph(data[0], x, y, z, clusters)
                children = [
                    html.Br(),
                    html.H5("A GMM 3D Graph will be made (1)."),
                    dcc.Graph(
                        id='graph1',
                        figure=fig
                    )
                ]
            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 3D Graph is not available yet.")
                children = [
                    html.Br(),
                    html.H5("A 3D Graph will be made. (1).")
                ]
            # NO 3D Options Available:
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 3D Graph is not available.")
                children = [
                    html.Br(),
                    html.H5("A DBSCAN 3D Graph is not available. Please make another selection (1).")
                ]

        return children

    # RIGHT GRAPH -----------------------------------------------------------------------------------------------

    # Once x, y, z axes have been chosen, output a scatter plot to graph 1
    @app.callback(Output('output-graph-area2', 'children'),
                  Input("dropdown_algorithm2", "value"),
                  Input("2d3d_graph2", "value"),
                  Input('dd_x_2', 'value'),
                  Input('dd_y_2', 'value'),
                  Input('dd_z_2', 'value'),
                  Input('clusters_selector2', 'value'))
    def update_output_graph2(algorithm, choice2d3d, x, y, z, clusters):

        print("Algorithm right is: " + str(algorithm))
        print("Choice of 2D or 3D right is: " + str(choice2d3d))
        print("x right is: " + str(x))
        print("y right is: " + str(y))
        print("z right is: " + str(z))
        print("Number of clusters right: " + str(clusters))

        alg_bool = algorithm is not None
        x_bool = x is not None
        y_bool = y is not None
        z_bool = z is not None

        # Make a 2D graph
        if alg_bool and (choice2d3d == '2D') and x_bool and y_bool:

            if algorithm == 'K-Means':
                print("TEST!!!!! A K-Means 2D Graph will be made.")
                fig = make_k_means_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (2)."),
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    ),
                ]
            elif algorithm == 'GMM':
                print("TEST!!!!! A GMM 2D Graph will be made.")
                fig = make_gmm_2d_graph(data[0], x, y, clusters)
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (2)."),
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    )
                ]
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (2).")
                ]

            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 2D Graph not available yet.")
                children = [
                    html.Br(),
                    html.H5("A " + str(algorithm) + " graph will be made (2).")
                ]

        # Make a 3D graph
        elif alg_bool and (choice2d3d == '3D') and x_bool and y_bool and z_bool:

            print("A Graph for " + str(algorithm) + " will be made:")

            if algorithm == 'K-Means':
                print("A K-Means 3D Graph will be made.")
                fig = make_k_means_3d_graph(data[0], x, y, z, clusters)

                children = [
                    html.Br(),
                    html.H5("A K-Means 3D Graph will be made (2)."),
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    ),
                ]
            elif algorithm == 'GMM':
                print("TEST!!!!! GMM 3D Graph is not available yet.")
                fig = make_gmm_3d_graph(data[0], x, y, z, clusters)
                children = [
                    html.Br(),
                    html.H5("A GMM 3D Graph will be made (2)."),
                    dcc.Graph(
                        id='graph2',
                        figure=fig
                    )
                ]
            elif algorithm == 'Mean-Shift':
                print("TEST!!!!! Mean-Shift 3D Graph is not available yet.")
                children = [
                    html.Br(),
                    html.H5("A 3D Graph will be made. (2).")
                ]
            # NO 3D Options Available:
            elif algorithm == 'DBSCAN':
                print("TEST!!!!! DBSCAN 3D Graph is not available.")
                children = [
                    html.Br(),
                    html.H5("A DBSCAN 3D Graph is not available. Please make another selection (2).")
                ]

        return children

    # **************************************** Run the app ****************************************
    app.run_server(debug=True, dev_tools_ui=False)


# FUNCTIONS ---------------------------------------------------------------------------------------------------

# Given a file, parse the contents into a scatter plot
def parse_contents(contents, filename, date):
    columns = []
    data_types = []

    # Data is separated by a comma
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Make a data frame, df, from either csv or excel file
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            columns = df.columns
            data_types = df.dtypes

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            columns = df.columns
            data_types = df.dtypes

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # parse the numerical-based attributes from the first row
    if columns is not None:
        for i in range(len(columns)):
            if data_types[i] == 'int64' or data_types[i] == 'float64':
                axes_options.append(columns[i])

    data.append(df)
    return df


# Return a figure for the 3D version of K-Means
def make_k_means_3d_graph(df, x_axis, y_axis, z_axis, clusters):

    x = df[[x_axis, y_axis, z_axis]].values
    model = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)

    # 3D scatter plot using Plotly
    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))
    labels = model.labels_
    trace = go.Scatter3d(x=x[:, 0], y=x[:, 1], z=x[:, 2],
                         mode='markers',
                         marker=dict(color=labels, size=10, line=dict(color='black', width=10)))
    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800,
                       width=800)
    graph_data = [trace]
    fig_k_means = go.Figure(data=graph_data, layout=layout)

    return fig_k_means


# Return a figure for the 2D version of K-Means
def make_k_means_2d_graph(df, x_axis, y_axis, clusters):

    # make a data frame from the csv data
    x = df[[x_axis, y_axis]].values

    # 3D scatter plot using Plotly
    # model = KMeans(n_clusters=4, init="k-means++", max_iter=300, n_init=10, random_state=0)
    model = KMeans(n_clusters=clusters, init="k-means++", max_iter=300, n_init=10, random_state=0)
    y_clusters = model.fit_predict(x)
    labels = model.labels_

    layout = go.Layout(title="K-Means",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=600,
                       width=600)
    fig_k_means = go.Figure(layout=layout)

    fig_k_means.add_trace(go.Scatter(
        x=x[:, 0], y=x[:, 1],
        hovertext=[x_axis, y_axis],  # no attribute names displayed :(
        mode='markers',
        marker=dict(color=labels, size=10)))

    fig_k_means.update_layout(
        hoverlabel=dict(
            font_size=16,
            font_family="Rockwell",
        ))

    return fig_k_means


# Return a figure for the 2D version of GMM
def make_gmm_2d_graph(df, x_axis, y_axis, clusters):

    color_iter = itertools.cycle(['cornflowerblue', 'darkorange', 'red', 'teal', 'gold', 'violet'])
    X = df[[x_axis, y_axis]].values

    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(X)
    cluster_labels = gmm.predict(X)

    layout = go.Layout(title="GMM",
                       margin=dict(l=0, r=0),
                       xaxis=dict(title=x_axis),
                       yaxis=dict(title=y_axis),
                       height=600,
                       width=600)
    fig_gmm = go.Figure(layout=layout)

    def plot_gmm_results(X, Y_, means, covariances, title):
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            fig_gmm.add_trace(go.Scatter(
                x=X[Y_ == i, 0], y=X[Y_ == i, 1],
                mode='markers',
                marker=dict(color=color, size=10)))

    plot_gmm_results(X, cluster_labels, gmm.means_, gmm.covariances_, 'Gaussian Mixture Model with EM')
    return fig_gmm


# Return a figure for the 3D version of GMM
def make_gmm_3d_graph(df, x_axis, y_axis, z_axis, clusters):

    color_iter = itertools.cycle(['cornflowerblue', 'darkorange', 'red', 'teal', 'gold', 'violet'])
    X = df[[x_axis, y_axis, z_axis]].values

    scene = dict(xaxis=dict(title=x_axis + ' <---'), yaxis=dict(title=y_axis + ' --->'),
                 zaxis=dict(title=z_axis + ' <---'))

    # Fit a Gaussian mixture with EM
    # n_components is the number of mixture components
    # covariance_type = FULL: each component has its own general covariance matrix
    gmm = mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(X)
    Y_ = gmm.predict(X)
    means = gmm.means_
    covariances = gmm.covariances_

    layout = go.Layout(title='Gaussian Mixture Model with EM',
                       margin=dict(l=0, r=0),
                       scene=scene,
                       height=800,
                       width=800)
    fig_gmm = go.Figure(layout=layout)

    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        fig_gmm.add_trace(go.Scatter3d(
            x=X[Y_ == i, 0], y=X[Y_ == i, 1], z=X[Y_ == i, 2],
            mode='markers',
            marker=dict(color=color, size=10)))

    return fig_gmm


# LEFT COMPONENTS ---------------------------------------------------------------------------------------------------

# Return a Div container that contains algorithm selection dropdown
def alg_selection1():
    return html.Div([
        html.H5("Algorithm Selection"),
        dcc.Dropdown(
            id="dropdown_algorithm1",
            options=[{'label': 'K-Means', 'value': 'K-Means'},
                     {'label': 'GMM', 'value': 'GMM'},
                     {'label': 'DBSCAN', 'value': 'DBSCAN'},
                     {'label': 'Mean-Shift', 'value': 'Mean-Shift'}
                     ],
            placeholder='Select an algorithm',
            style={
                'width': '50%',
                'display': 'inline-block',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            }
        ),
    ]),


def num_clusters_selection1():
    return html.Div([
        html.H5("Select the number of predicted clusters (will be applied to K-Means and GMM only)"),
        dcc.RadioItems(
            id='clusters_selector1',
            options=[
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3},
                {'label': '4', 'value': 4},
                {'label': '5', 'value': 5},
                {'label': '6', 'value': 6},
                {'label': '7', 'value': 7},
                {'label': '8', 'value': 8}

            ],
            value=2,
            labelStyle={'display': 'inline-block'}
        ),
        html.Br()
    ])


# Return a Div container that contains 2D or 3D graph choice selection dropdown
def graph_2d3d_selection1():
    return html.Div([
        html.H5("Select a 2D or 3D graph"),
        dcc.RadioItems(
            id='2d3d_graph1',
            options=[
                {'label': '2D', 'value': '2D'},
                {'label': '3D', 'value': '3D'}
            ],
            value='2D',
            labelStyle={'display': 'inline-block'}),
        html.Br()
    ])


# Return a Div container that contains axis selection dropdowns for x, y, and z (3D)
def axes_selection_xyz_1():
    return html.Div([
        html.H5("X-Axis"),
        dcc.Dropdown(
            id="dd_x_1",
            placeholder='Select X-axis attribute 1',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            },
        ),
        html.H5("Y-Axis"),
        dcc.Dropdown(
            id="dd_y_1",
            placeholder='Select Y-axis attribute 1',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            },
        ),
        html.H5("Z-Axis"),
        dcc.Dropdown(
            id="dd_z_1",
            placeholder='Select Z-axis attribute 1',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            }
        ),
        html.Br()
    ]),


# RIGHT COMPONENTS ---------------------------------------------------------------------------------------------------

# Return a Div container that contains algorithm selection dropdown
def alg_selection2():
    return html.Div([
        html.H5("Algorithm Selection"),
        dcc.Dropdown(
            id="dropdown_algorithm2",
            options=[{'label': 'K-Means', 'value': 'K-Means'},
                     {'label': 'GMM', 'value': 'GMM'},
                     {'label': 'DBSCAN', 'value': 'DBSCAN'},
                     {'label': 'Mean-Shift', 'value': 'Mean-Shift'}
                     ],
            placeholder='Select an algorithm',
            style={
                'width': '50%',
                'display': 'inline-block',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            }
        ),
    ]),


def num_clusters_selection2():
    return html.Div([
        html.H5("Select the number of predicted clusters (will be applied to K-Means and GMM only)"),
        dcc.RadioItems(
            id='clusters_selector2',
            options=[
                {'label': '2', 'value': 2},
                {'label': '3', 'value': 3},
                {'label': '4', 'value': 4},
                {'label': '5', 'value': 5},
                {'label': '6', 'value': 6},
                {'label': '7', 'value': 7},
                {'label': '8', 'value': 8}

            ],
            value=2,
            labelStyle={'display': 'inline-block'}
        ),
        html.Br()
    ])


# Return a Div container that contains 2D or 3D graph choice selection dropdown
def graph_2d3d_selection2():
    return html.Div([
        html.H5("Select a 2D or 3D graph"),
        dcc.RadioItems(
            id='2d3d_graph2',
            options=[
                {'label': '2D', 'value': '2D'},
                {'label': '3D', 'value': '3D'}
            ],
            value='2D',
            labelStyle={'display': 'inline-block'}),
        html.Br()
    ])


# Return a Div container that contains axis selection dropdowns for x, y, and z (3D)
def axes_selection_xyz_2():
    return html.Div([
        html.H5("X-Axis"),
        dcc.Dropdown(
            id="dd_x_2",
            placeholder='Select X-axis attribute 2',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            },
        ),
        html.H5("Y-Axis"),
        dcc.Dropdown(
            id="dd_y_2",
            placeholder='Select Y-axis attribute 2',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            },
        ),
        html.H5("Z-Axis"),
        dcc.Dropdown(
            id="dd_z_2",
            placeholder='Select Z-axis attribute 2',
            options=[{'label': i, 'value': i} for i in axes_options],
            style={
                'width': '50%',
                'lineHeight': '30px',
                'borderWidth': '1px',
                'textAlign': 'left'
            }
        ),
        html.Br()
    ]),


if __name__ == '__main__':
    main()
