"""
Capstone Project: GeoClusters
Written by: Jandlyn Bentley, Bridgewater State University, 2021

# This program is the primary program for the GeoClusters Dash web app.

Here the Dash app is established, with basic instructions on how to use the open-source tool,
a data set file upload component, and (eventually) a clustering algorithm comparison component.
Currently, this program only takes in and parses a .csv or .xsl data set and produces a data table
in the web app.
"""

import base64
import datetime
import io
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

markdown_text = '''
---
### GeoClusters is a visual tool for geoscientists to view their data under various clustering algorithms in real time.

&nbsp
To begin, drag or drop a preprocessed CSV or Excel file into the upload area.
Your data set must include only the data tiles in the first row and columns of data.
Next, choose from the drop down menu to compare clustering algorithms.
A data table will also be provided at the bottom of the app for reference.

The code for this open-source tool can be found on [Github](https://github.com/JandlynBentley/GeoClusters).
'''


def main():

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([

        html.H1(
            children='GeoClusters',
            style={
                'textAlign': 'center'

            }
        ),

        # Instructions to the user displayed with markdown
        dcc.Markdown(
            children=markdown_text,
            style={
                'textAlign': 'center'
            }
        ),

        html.Br(),

        # Dash Upload component
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

        html.Br(),

        # Container for the data table
        html.Div(id='output-data-upload'),
    ])

    # Given the data from Upload, output a data table
    @app.callback(Output('output-data-upload', 'children'),
                  Input('upload-data', 'contents'),
                  State('upload-data', 'filename'),
                  State('upload-data', 'last_modified'))
    def update_output(list_of_contents, list_of_names, list_of_dates):
        if list_of_contents is not None:
            children = [
                # c = contents, n = filename, d = date
                parse_contents(c, n, d) for c, n, d in zip(list_of_contents, list_of_names, list_of_dates)
            ]
            return children

    # Run the app
    app.run_server(debug=True)


# Given a file, parse the contents into a DataTable
def parse_contents(contents, filename, date):

    # Data is separated by a comma
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Make a dataframe, df, from either csv or excel file
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            # df = pd.read_csv('file_location\filename.txt', delimiter = "\t")
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    # Display the data set's title and timestamp
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # display as a data table
        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # REMOVED THE 'RAW CONTENT PROVIDED BY WEB BROWSER' DEBUGGING CODE FROM ORIGINAL TUTORIAL
    ])


if __name__ == '__main__':
    main()







