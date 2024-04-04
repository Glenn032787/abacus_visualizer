import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.graph_objects as go
import itertools
import numpy as np 
import base64
import io
from plotly.subplots import make_subplots
import pyarrow.parquet as pq
import pyarrow.compute as pc
import json


with open('config.json', 'r') as f:
  data = json.load(f)
  CNV_link = data['copy_number_file_path']
  GP_link = data['gpfit_file_path']
#CNV_link = "/projects/steiflab/scratch/glchang/abacus_visualization/test/A95621B.copy_number.parquet"
#GP_link = "/projects/steiflab/scratch/glchang/abacus_visualization/test/A95621B.gp_fit.parquet"


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] # just styling

app = dash.Dash(__name__, external_stylesheets=external_stylesheets) # app

app.title = "Abacus Visualizer"

chromosomes_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y']
chromosome_sizes = [249250621, 243199373, 198022430, 191154276, 180915260, 171115067, 159138663, 146364022, 141213431, 135534747, 135006516, 133851895, 115169878, 107349540, 102531392, 90354753, 81195210, 78077248, 59128983, 63025520, 48129895, 51304566, 155270560, 59373566]


# Calculate cumulative sizes
cumulative_sizes = [0] + list(itertools.accumulate(chromosome_sizes)) 


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors)+1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)     
    nvals = [(v-bvals[0])/(bvals[-1]-bvals[0]) for v in bvals]  #normalized values
    
    dcolorscale = [] # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k+1], colors[k]]])
    return dcolorscale    

def parquetFilter(inequality, parquet, column, value):
    if inequality == "=":
        return pc.equal(parquet[column], value)
    elif inequality == ">":
        return pc.greater(parquet[column], value)
    elif inequality == "<":
        return pc.less(parquet[column], value)


table = pq.read_table(CNV_link)
table = table.sort_by([("cell_id", "ascending")])
df = table.to_pandas()

gpfit_table = pq.read_table(GP_link)
gpfit_table = gpfit_table.sort_by([("cell_id", "ascending")])

heatmap_df = df
heatmap_df['x_mapping'] = [start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(df['start'], df['chrom'])]

bvals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
color_schemes = ['#3498db', '#85c1e9', '#d3d3d3', '#fee2d5', '#fcc3ac', '#fc9f81', '#fb7c5c', '#f5543c', '#e32f27', '#c1151b', '#9d0d14', '#780428', '#530031', '#40092e', '#2d112b']
colorscale = discrete_colorscale(bvals, color_schemes)

bvals = np.array(bvals)
tickvals = [str(i) for i in range(15)] #position with respect to bvals where ticktext is displayed
ticktext = [f'<{bvals[1]}'] + [f'{bvals[k]}-{bvals[k+1]}' for k in range(1, len(bvals)-2)]+[f'>{bvals[-2]}']


heatmap_fig = go.Figure(data=go.Heatmap(
    z=df['copy_number'],
    x=heatmap_df['x_mapping'],
    y=df['cell_id'],
    colorscale=colorscale,
    zmin=0,  # Set the minimum value for the colorbar scale
    zmax=14,  # Set the maximum value for the colorbar scale
    colorbar=dict(title="Copy Number"),
))
heatmap_fig.update_yaxes(title_text="Cell ID")

heatmap_fig.update_layout(
    title='Heatmap',
    xaxis=dict(
        title='Chromosome',
        tickvals=cumulative_sizes,
        ticktext=chromosomes_names + [''],
        tickmode='array',
        ticklabelmode='period',
        ticklabelposition='outside',
        tickformat='d',
        range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
    )
)

x_options = df.columns
y_options = df.columns
z_options = df.columns
cell_id_options = df['cell_id'].unique().tolist()

# Create the Dash layout
app.layout = html.Div([
    html.H1("Abacus Visualizer", style={"background-color": "#fb7c5c", "color": "white", "padding": "13px"}),
    
    dcc.Loading(
        id="loading-component",
        type="circle",  # You can also use "circle" or "dot"
        children=[dcc.Graph(
        id='heatmap',
        figure=heatmap_fig,
        style={'overflow': 'scroll', 'height': '900px'}
    )]),
    html.Div(id='info-block', children=[], style={'width': '97%', 'text-align': 'right', 'font-size': '15px'}),
    html.Div([
        html.Div([
            html.Label(['Metadata:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Upload(
                id='upload-metadata',
                children=html.Button('Upload CSV File')
            ),
            html.Div(id='output-message'),
            html.Div(id='intermediate-data', style={'display': 'none'}),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['Parameter:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': x, 'value': x} for x in x_options],
                value=None,
                placeholder='Select a value'
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='fil-cond-dropdown-metadata',
                options=[
                    {'label': '>', 'value': '>'},
                    {'label': '=', 'value': '='},
                    {'label': '<', 'value': '<'}
                ],
                style={'width': '100%'},
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='value-dropdown',
                options=[],
                value=None,
                placeholder='Select a value'
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),

    
    html.Div([
        html.Div([
            html.Button('reset', id='reset-button', n_clicks=0),
            html.Button('apply', id='apply-button', n_clicks=0),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['Parameter:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='column-dropdown-sort',
                options=[{'label': x, 'value': x} for x in x_options],
                value=None,
                placeholder='Select a value'
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['sort by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='sort-cond-dropdown-metadata',
                options=[
                    {'label': 'accending', 'value': True},
                    {'label': 'decending', 'value': False}
                ],
                style={'width': '100%'},
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'})
    ]),
    html.Div([
        dcc.Graph(id='scatter', style={'width': '60%'}),
        dcc.Graph(id='multiplicity', style={'width': '40%'})
    ], style={'display': 'flex', 'width': '100%'}),
    html.Div([
        html.Div([
            html.Label(['Cell ID:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='cell-id-dropdown',
                options=[{'label': cell_id, 'value': cell_id} for cell_id in cell_id_options],
                style={'width': '100%'}
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['Parameter:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='x-axis-dropdown-scatter',
                options=[{'label': x, 'value': x} for x in x_options],
                style={'width': '100%'},
            ),
        ], style={'width': '13%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['filter by:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Dropdown(
                id='fil-cond-dropdown-scatter',
                options=[
                    {'label': '>', 'value': '>'},
                    {'label': '=', 'value': '='},
                    {'label': '<', 'value': '<'}
                ],
                style={'width': '100%'},
            ),
        ], style={'width': '3%', 'display': 'inline-block', 'vertical-align': 'top'}),
        html.Div([
            html.Label(['value:'], style={'font-weight': 'bold', "text-align": "left"}),
            dcc.Input(id='input-val-scatter', type='number', min=0, value=0)
        ], style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top'}),  
    ]),
    html.Div([     
        html.Button('apply', id='apply-button-scatter', n_clicks=0, style={'width': '105px'}),
    ], style={'margin-top': '10px', 'display': 'flex', 'flex-direction': 'row', 'align-items': 'center'})
], style={'display': 'flex', 'flex-direction': 'column', 'width': '100%'})

def get_coverage_order(df_group):
    '''Get the order of magnitude of the coverage depth for the panel.'''
    group_order = 1

    if len(df_group) > 0:
        if df_group['2'].max() > 0:
            group_order = int(np.log10(df_group['2'].max()))

    return group_order 

@app.callback(
    Output('intermediate-data', 'children'),
    Output('column-dropdown', 'options'),
    Output('column-dropdown-sort', 'options'),
    Input('upload-metadata', 'contents'),
    State('upload-metadata', 'filename'),
    prevent_initial_call=True
)
def update_dropdown(contents, filename):
    if contents is None:
        raise dash.exceptions.PreventUpdate

    # Read the uploaded file into a DataFrame
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df_meta = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    # Update options for column-dropdown based on DataFrame columns
    column_options = [{'label': col, 'value': col} for col in (df_meta.columns.union(x_options))]
    return df_meta.to_json(date_format='iso', orient='split'), column_options, column_options

# what is this function even doing ?? need to figure this out before moving forward with metadata filtering
@app.callback(
    Output('value-dropdown', 'options'),
    Input('column-dropdown', 'value'),
    State('intermediate-data', 'children'),
    prevent_initial_call=True
)
def update_value_dropdown(selected_column, jsonified_df):
    if selected_column is None:
        raise dash.exceptions.PreventUpdate
    
    if jsonified_df is None:
        return [{'label': str(val), 'value': val} for val in df[selected_column].unique()]

    # Read the DataFrame from the intermediate data
    df_meta = pd.read_json(jsonified_df, orient='split')

    if selected_column in df_meta.columns:
        # Get unique values for the selected column
        unique_values = df_meta[selected_column].unique() 
    else:
        unique_values = df[selected_column].unique()

    # Update options for value-dropdown based on unique values
    value_options = [{'label': str(val), 'value': val} for val in unique_values]

    return value_options


@app.callback(
    [Output('heatmap', 'figure'),
     Output('apply-button', 'n_clicks'),
     Output('reset-button', 'n_clicks')],
    [Input('apply-button', 'n_clicks'),
     Input('reset-button', 'n_clicks')],
    [State('upload-metadata', 'contents'),
    State('column-dropdown', 'value'),
    State('fil-cond-dropdown-metadata', 'value'),
    State('value-dropdown', 'value'),
    State('intermediate-data', 'children'), 
    State('column-dropdown-sort', 'value'),
    State('sort-cond-dropdown-metadata', 'value')]
)
def update_heatmap(n_clicks_apply, n_clicks_reset, uploaded_metadata, x_axis_val, cond_type, input_val, intermediate_data, column_sort_value, sort_cond_value):

    curr_heatmap_fig = go.Figure()

    if 'apply-button' == ctx.triggered_id:
        # # Map the dropdown value to the corresponding conditional operator
        # # Get the selected conditional operator from the dropdown value
        # conditional_operator = conditional_operators[cond_type]

        if intermediate_data is None or x_axis_val in x_options:
            filtered_table = pc.filter(table, parquetFilter(cond_type, table, x_axis_val, input_val))
            df = filtered_table.to_pandas()
        else:
            content_type, content_string = uploaded_metadata.split(',')
            decoded = base64.b64decode(content_string)
            
            # Read the file using pandas
            metadata_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Apply the selected filter to the metadata dataset
            filtered_data = metadata_df
            condition = None

            if cond_type == '=':
                    cond_type = '=='

            if input_val:
                if isinstance(input_val, (int, float)):
                    condition = f"{x_axis_val} {cond_type} {input_val}"
                else:
                    condition = f"{x_axis_val} {cond_type} '{input_val}'"
                filtered_data = filtered_data.query(condition)                

            if column_sort_value is not None and sort_cond_value is not None:
                sortedFilterData = filtered_data.sort_values(by=column_sort_value, ascending=sort_cond_value)
            else:
                sortedFilterData = filtered_data.sort_values(by="cell_id")
            cellIDorder = sortedFilterData['cell_id'].tolist()

            metadata_df_sel_cell_ids = filtered_data['cell_id'].unique()
            metadata_df_sel_cell_ids = metadata_df_sel_cell_ids.tolist()
            
            df = table.to_pandas()
            df = df[df['cell_id'].isin(metadata_df_sel_cell_ids)]
            df['cell_id'] = df['cell_id'].astype(pd.CategoricalDtype(categories=cellIDorder, ordered=True))
            df = df.sort_values(by='cell_id')

        df['x_mapping'] = [start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(df['start'], df['chrom'])]
        
        dd_heatmap_fig = go.Figure(data=go.Heatmap(
            z=df['copy_number'],
            x=df['x_mapping'],
            y=df['cell_id'],
            colorscale=colorscale,
            zmin=0,  # Set the minimum value for the colorbar scale
            zmax=14,  # Set the maximum value for the colorbar scale
            colorbar=dict(title="Copy Number"),
        ))
        dd_heatmap_fig.update_yaxes(title_text="Cell ID")
        dd_heatmap_fig.update_layout(
            title='Heatmap',
            xaxis=dict(
                title='Chromosome',
                tickvals=cumulative_sizes,
                ticktext=chromosomes_names + [''],
                tickmode='array',
                ticklabelmode='period',
                ticklabelposition='outside',
                tickformat='d',
                range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
            )
        )
        
        curr_heatmap_fig = dd_heatmap_fig
        n_clicks_apply = 0

    elif "reset-button" == ctx.triggered_id:
        curr_heatmap_fig = heatmap_fig
        n_clicks_reset = 0
    else:
        curr_heatmap_fig = heatmap_fig
    return curr_heatmap_fig, n_clicks_apply, n_clicks_reset

@app.callback(
    Output('info-block', 'children'),
    [Input('heatmap', 'clickData')]
)
def update_info_block(clickData):
    if clickData is not None:
        point_data = clickData['points'][0]
        chromosome = get_chrom_from_x_val(point_data['x'], chromosomes_names, cumulative_sizes)
        cell_id = point_data['y']
        copy_number = point_data['z']

        # Construct the text content for the info block
        text_content = f'''
            **Chromosome:** {chromosome} \t
            **Cell ID:** {cell_id} \t
            **Copy Number:** {copy_number}
        '''
        return dcc.Markdown(text_content)
    else:
        text_content = f'''
            **Chromosome:**  \t
            **Cell ID:**  \t
            **Copy Number:** 
        '''
        return text_content

def get_chrom_from_x_val(val, chrName, chrCumulativeSize):
    for i in range(len(chrCumulativeSize)):
        if val < chrCumulativeSize[i]:
            return chrName[i-1]
    raise ValueError

# Define the callback to show the scatter plot on hover
@app.callback(
    [Output('scatter', 'figure'),
    Output('apply-button-scatter', 'n_clicks')],
    [Input('heatmap', 'clickData'),
    Input('apply-button-scatter', 'n_clicks')],
    [State('cell-id-dropdown', 'value'),
     State('x-axis-dropdown-scatter', 'value'),
     State('fil-cond-dropdown-scatter', 'value'),
     State('input-val-scatter', 'value')]
)
def show_scatter(clickData, n_clicks_apply_scatter, cell_id, x_axis_val, cond_type, input_val):
    if clickData is not None and not n_clicks_apply_scatter:
        # Click on heatmap
        cell_id = clickData['points'][0]['y']
        filtered_table = pc.filter(table, parquetFilter("=", table, "cell_id", cell_id))
    elif "apply-button-scatter" == ctx.triggered_id and cell_id is not None:
        # Click on apply button
        if x_axis_val is None:
            filtered_table = pc.filter(table, parquetFilter("=", table, "cell_id", cell_id))
        else:
            condition = pc.and_(parquetFilter(cond_type, table, x_axis_val, input_val), parquetFilter("=", table, "cell_id", cell_id))
            filtered_table = pc.filter(table, condition)
    else:
        return go.Figure(), n_clicks_apply_scatter
    
    filtered_table = filtered_table.sort_by([("cell_id", "descending")])
    scatter_df = filtered_table.to_pandas()
    scatter_df['x_mapping'] = [start + cumulative_sizes[chromosomes_names.index(chrom)] for start, chrom in zip(scatter_df['start'], scatter_df['chrom'])]

    scatter_fig = go.Figure(data=go.Scattergl(
        x=scatter_df['x_mapping'],
        y=scatter_df['modal_corrected']*scatter_df['assignment'],
        mode='markers',
        marker=dict(
            size=5,
            color=scatter_df['copy_number'],
            colorscale=colorscale,
            cmin=0,
            cmax=14,
            showscale=True,
            colorbar={"title": "Copy Number"}, 
        ),
        text=chromosomes_names,
        hovertemplate='Position: %{x} <br>Scaled Sequencing Coverage: %{y}',
        showlegend=False 
    ))

    scatter_fig.update_layout(
        title='Scatter Plot',
        xaxis=dict(
            title='Chromosome',
            tickvals=cumulative_sizes,
            ticktext=chromosomes_names + [''],
            tickmode='array',
            ticklabelmode='period',
            ticklabelposition='outside',
            tickformat='d',
            range=[0, cumulative_sizes[-1] + chromosome_sizes[-1]], 
        ),
    )
    scatter_fig.update_yaxes(title_text="Scaled sequencing coverage")
    n_clicks_apply_scatter = 0
    return scatter_fig, n_clicks_apply_scatter
    
@app.callback(
    Output('multiplicity', 'figure'),
    [Input('heatmap', 'clickData'),
    Input('apply-button-scatter', 'n_clicks')],
    [State('cell-id-dropdown', 'value'),
     State('x-axis-dropdown-scatter', 'value'),
     State('fil-cond-dropdown-scatter', 'value'),
     State('input-val-scatter', 'value')]
)
def update_multiplicity(clickData, n_clicks_apply_scatter, cell_id, x_axis_val, cond_type, input_val):
    selected_cell_id = None
    if "apply-button-scatter" == ctx.triggered_id and cell_id is not None:
        selected_cell_id = cell_id
    elif clickData is not None:
        selected_cell_id = clickData['points'][0]['y']
    else:
        return go.Figure()
    
    filtered_gpfittable = pc.filter(gpfit_table, parquetFilter("=", gpfit_table, "cell_id", selected_cell_id))
    multiplicity_df = filtered_gpfittable.to_pandas()
    
    filtered_table = pc.filter(table, parquetFilter("=", table, "cell_id", selected_cell_id))
    df_coverage_order = filtered_table.to_pandas()
    df_coverage_order = df_coverage_order.groupby(['copy_number', 'assignment']).apply(get_coverage_order, include_groups=False).reset_index()    
    df_coverage_order.rename({0: 'order'}, axis=1, inplace=True)
    
    if max(df_coverage_order['order']) >= 6:
        scale_values = 1000000
        scale_label = 'Mb'
    elif max(df_coverage_order['order']) >= 3:
        scale_values = 1000
        scale_label = 'Kb'
    else:
        scale_values = 1
        scale_label = 'Bases'
    
    dfg1 = multiplicity_df[multiplicity_df['state'] == 2]
    dfg1 = dfg1[dfg1['training_cell_id'] != '']
    dfg1['color'] = ['#fcc3ac' if condition == 'G1' else '#d3d3d3' for condition in dfg1['ref_condition']]

    multiplicity_fig = make_subplots(rows=1, cols=2, subplot_titles=("State 2","State 4"))
    multiplicity_fig.add_trace(go.Scattergl(
        x=dfg1['total_coverage']/scale_values,
        y=dfg1['2_coverage']/scale_values,
        mode='markers',
        marker=dict(
            size=4,
            color=dfg1['color']
        ),
        text=dfg1['ref_condition'],
        showlegend=False), row=1, col=1)
   
    # MULTIPLICITY REF
    dfg1_ref = multiplicity_df[multiplicity_df['state'] == 4]
    dfg1_ref = dfg1_ref[dfg1_ref['training_cell_id'] != '']
    dfg1_ref['color'] = ['#fcc3ac' if condition == 'G1' else '#d3d3d3' for condition in dfg1_ref['ref_condition']]

    multiplicity_fig.add_trace(go.Scattergl(
        x=dfg1_ref['total_coverage']/scale_values,
        y=dfg1_ref['2_coverage']/scale_values,
        mode='markers',
        marker=dict(
            size=4,
            color=dfg1_ref['color']
        ),
        text=dfg1_ref['ref_condition'],
        showlegend=False
    ), row=1, col=2)

    # Add titles
    multiplicity_fig.update_layout(title_text="Ploidy Coverage Curves")
    multiplicity_fig.update_layout(coloraxis=dict(colorscale=colorscale))
    multiplicity_fig.update_yaxes(title_text=scale_label + " covered")
    multiplicity_fig.update_xaxes(title_text=scale_label + " sequenced")

    # Add number of binds
    for i in range(1,3):
        df = multiplicity_df[multiplicity_df['state'] == i*2]
        numBin = df['num_bins'].iloc[0]
        multiplicity_fig.add_annotation(
            xref="x domain", yref="y domain",
            x=0, y=1,
            text=f"{numBin} bins",
            showarrow=False,
            font=dict(
                size=16,
                color="#000000"),
            row=1, col=i
        )
    return multiplicity_fig


run = False
if __name__ == '__main__':
    if not run:
        run = True
        # main()
        # webbrowser.open_new_tab('http://127.0.0.1:8888/')
        print('finished main')
        app.run(debug=False, use_reloader=True)
    
    
