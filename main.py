#!/usr/bin/env python
# coding: utf-8

# In[34]:

import logging
import logging.handlers
import requests
import pandas as pd
import os
import networkx as nx
from matplotlib.pyplot import figure
from pyvis.network import Network
import igraph as ig # For Kamada Kawai Layout
import numpy as np

from bokeh.io import output_notebook, show, save
from bokeh.models import Arrow, NormalHead,VeeHead, CustomJS,Legend, LegendItem, CustomJSTransform, LabelSet,HoverTool, BoxSelectTool,TapTool,PointDrawTool,Range1d, Circle, Scatter, StaticLayoutProvider,ColumnDataSource, MultiLine, EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.plotting import figure ,from_networkx
from bokeh.transform import linear_cmap , transform
from bokeh.models.widgets import Div
from bokeh.layouts import layout, row
from networkx.drawing.nx_agraph import graphviz_layout
import warnings
warnings.filterwarnings("ignore")

print("Getting credentials ...")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger_file_handler = logging.handlers.RotatingFileHandler(
    "status.log",
    maxBytes=1024 * 1024,
    backupCount=1,
    encoding="utf8",
)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger_file_handler.setFormatter(formatter)
logger.addHandler(logger_file_handler)


try :
    airtable_api_key = os.environ['AIRTABLE_API_KEY']
    base_id = os.environ['BASE_ID'] # Epiverse TRACE
    api_status = "API keys ingested"
except KeyError:
    api_status = "Api keys not available"
    
    
logger.info(f"API status: {api_status}")
    

#airtable_api_key = os.environ['AIRTABLE_API_KEY']
#base_id = os.environ['BASE_I'] # Epiverse TRACE
table_id = "tbljWHt6xZCuLMmqL" # Table Id from https://airtable.com/appgKw8XxfNU3FeNH/api/docs#curl/table:trace%20ecosystem
table_id_dataflows = "tblAs8Ep64VPzMBVN"


color_dict = {"Maturing":"#1485c7",
              "Stable" : "#AEC800",
              "Experimental" : "#10BED2",
              "Concept" : "#EBE6E0"
               }


def dev_color_assign(node_value):
    node_color = str()
    if node_value in dev_c:
        node_color = color_dict['Concept']
    elif node_value in dev_e:
        node_color =color_dict['Experimental']
    elif node_value in dev_m:
        node_color =color_dict['Maturing']
    elif node_value in dev_s:
        node_color =color_dict['Stable']
    elif node_value in dev_ret:
        node_color ="#E94560"
    else :
        node_color ="#FFFAE7"
    return node_color


def flow_color_assign(edge_value):
    edge_color ="#FFFFFF"
    if edge_value == str('Concept'):
        edge_color = color_dict['Concept']
    elif edge_value == str("Experimental"):
        edge_color =color_dict['Experimental']
    elif edge_value == str("Maturing"):
        edge_color =color_dict['Maturing']
    elif edge_value ==str("Stable"):
        edge_color =color_dict['Stable']
    elif edge_value == str('None'):
        edge_color ="#EEEEEE"
        
    return edge_color


def get_airtable_data(url,headers):
    run = True
    airtable_records = []
    # Getting recorders considering pagination
    while run is True:
        response = requests.get(url,  headers=headers)
        airtable_response = response.json()
        airtable_records += (airtable_response['records'])
        if 'offset' in airtable_response:
            run = True
            params = (('offset', airtable_response['offset']),)
        else:
             run = False

    airtable_rows = [] 
    airtable_index = []
    for record in airtable_records:
        airtable_rows.append(record['fields'])
        airtable_index.append(record['id'])
        
    return pd.DataFrame(airtable_rows,airtable_index).reset_index()
    


# # Get Data from Airtable

# ## Get Dataflow table



url_dataflow= "https://api.airtable.com/v0/appgKw8XxfNU3FeNH/{}?view=Grid+view".format(table_id_dataflows)
headers = {"Authorization": "Bearer {}".format(airtable_api_key) }

logger.info("Getting data from EPIVERSE TRACE Airtables")
airtable_raw_dataframe_dataflow = get_airtable_data(url_dataflow,headers)
# Change Dtypes
airtable_raw_dataframe_dataflow = airtable_raw_dataframe_dataflow.astype({'From': 'str', 'To': 'str','Flow dev status':'str'})
# Replace extra characters from the columns and subset for requored columns
airtable_raw_dataframe_dataflow.loc[:,'From'] = airtable_raw_dataframe_dataflow['From'].str.replace('[','',regex = False).str.replace(']','',regex = False).str.replace("'",'',regex = False)
airtable_raw_dataframe_dataflow.loc[:,'To'] = airtable_raw_dataframe_dataflow['To'].str.replace('[','',regex = False).str.replace(']','',regex = False).str.replace("'",'',regex = False)
req_cols_dataflow = ['From','To', 'Flow dev status']
airtable_dataframe_dataflow = airtable_raw_dataframe_dataflow[req_cols_dataflow]
airtable_dataframe_dataflow.columns =['from_dataflow','to_dataflow','status_dataflow']


# ## Get Software Ecosystem table 


url= "https://api.airtable.com/v0/appgKw8XxfNU3FeNH/{}?view=Grid+view".format(table_id)

airtable_raw_dataframe = get_airtable_data(url,headers)
airtable_raw_dataframe.loc[:,'logo_ext'] = [i[0]['url'] if str(i)!="nan" else "none" for i in airtable_raw_dataframe['Logo'].to_list() ]

# Subsetting the Software ecosystem table to get list of Nodes
req_col= ['Software name','Flows into','index']
graph_df = airtable_raw_dataframe[req_col]

# To get all the nodes even when there are no connections
graph_df_explode_temp = graph_df.explode('Flows into') 

# Create a mapping dictionary for mappinga airtable ids with names 
mapping = dict(graph_df[['index','Software name']].values)
airtable_dataframe_dataflow.loc[:,'from_package'] = airtable_dataframe_dataflow.loc[:,'from_dataflow'].map(mapping)
airtable_dataframe_dataflow.loc[:,'to_package'] = airtable_dataframe_dataflow.loc[:,'to_dataflow'].map(mapping)
airtable_dataframe_dataflow = airtable_dataframe_dataflow[['from_package','to_package','status_dataflow']]



logger.info("Carrying out data wrangling steps...")

# Get names and flows of packages 
graph_df_explode = graph_df_explode_temp.merge(graph_df[['Software name','index']], left_on = ['Flows into'], right_on = ['index'], how ='left')
graph_df_explode = graph_df_explode.drop(['Flows into','index_x','index_y'], axis =1)
graph_df_explode.columns = ['package','flows_into']
graph_df_mapping = graph_df_explode.copy(deep=True)
graph_df_single_nodes = graph_df_explode[graph_df_explode['flows_into'].isna()]['package'].to_list()
graph_df_explode_raw = graph_df_explode.dropna()

# Attach flows from DataFlow table 
graph_df_explode = pd.DataFrame(np.vstack([graph_df_explode_raw, airtable_dataframe_dataflow[['from_package','to_package']]]), 
                                columns=graph_df_explode_raw.columns).drop_duplicates(['package','flows_into'],keep='first')

# Subsetting raw airtable 
req_col_metadata = ['Software name','Scope','Description','On CRAN','Dev status',
                      'License','Github project','Website','Maintainer email','logo_ext']
graph_df_metadata = airtable_raw_dataframe[req_col_metadata]
# Fillna with None
graph_df_metadata = graph_df_metadata.fillna("None")
graph_df_metadata.columns = ['package','scope','description','on_cran','dev_status',
                               'license','github','website','maintainer_email','logo_ext']

# Requising explode from earlier -- splits list in columns to multiple rows
graph_df_metadata_merged = graph_df_explode.merge(graph_df_metadata,left_on =['package'] ,right_on =['package'], how='left')
graph_df_metadata_merged.loc[:,'description'] = graph_df_metadata_merged['description'].str.replace('\n','')

# separate table for node popup 
graph_node_descr = graph_df_mapping.merge(graph_df_metadata,left_on =['package'] ,right_on =['package'], how='left')
graph_node_descr['description'] = graph_node_descr['description'].str.replace('\n','')
graph_node_descr['maintainer_email'] = [str(i).replace('[','').replace(']','').replace(' ','') for i in graph_node_descr['maintainer_email'].to_list()]
node_attr_dict_temp = graph_node_descr.drop_duplicates(['package'], keep='first')


data_sources = node_attr_dict_temp[node_attr_dict_temp['on_cran']=='Not applicable']
data_goods_list = data_sources['package'].to_list()

# development classification
dev_c,dev_e, dev_m,dev_s,dev_ret = [node_attr_dict_temp[node_attr_dict_temp['dev_status']==i]['package'].to_list()
                                    for i in ['Concept', 'Experimental', 'Maturing', 'Stable', 'To retire']]

        


# # Getting Graph Visualization

logger.info("Building graph ...")
G = nx.DiGraph()
# Ingesting data from dataframe directly
G = nx.from_pandas_edgelist(graph_df_explode, source='package',target= 'flows_into',create_using=nx.DiGraph())
# Add single nodes 
for single_node in graph_df_single_nodes:
    G.add_node(single_node)

# Adding Description to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.description])),
"Description")
# Adding Scope to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.scope])),
"Scope")
# Adding License to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.license])),
"License")
# Adding Github Link to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.github])),
"Link to Github")
# Adding Website Link to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.website])),
"Website")
# Adding CRAN binary to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.on_cran])),
"On Cran")
# Adding Maintainer Email to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.maintainer_email])),
"Maintainer Email")
# Adding Image URL to node attributes
nx.set_node_attributes(G,
dict(zip(node_attr_dict_temp.package,[str(s) for s in node_attr_dict_temp.logo_ext])),
"image_url")

# Adding attributes to edges by replaces 

for x,y,z in G.edges(data=True):
    status = airtable_dataframe_dataflow.loc[(airtable_dataframe_dataflow['from_package']==x)&(airtable_dataframe_dataflow['to_package']==y)]['status_dataflow'].to_list()
    G.add_edge(x, y, key=str(status)[2:-2])
    
# Making color palette for bokeh
node_cmap = dict(zip(node_attr_dict_temp.package,[dev_color_assign(str(s)) for s in node_attr_dict_temp.package]))
# Set node color using custom function 
colors = dict(zip(G.nodes,[dev_color_assign(str(s)) for s in G.nodes]))
nx.set_node_attributes(G, {k:v for k,v in colors.items()},'colors' )

# Set Edg Color
edge_attrs = {}
for start_node, end_node, edge_value in G.edges(data=True):
    edge_color = flow_color_assign(str(edge_value['key']))
    edge_attrs[(start_node, end_node)] = edge_color 
nx.set_edge_attributes(G, edge_attrs, "edge_color")

# Setting thin lines for white edges
edge_attrs_thickness = {}
for start_node, end_node, edge_value in G.edges(data=True):
    edge_color = flow_color_assign(str(edge_value['key']))
    if edge_color == "#FFFFFF":
        edge_attrs_thickness[(start_node, end_node)] = 6 
    else:
        edge_attrs_thickness[(start_node, end_node)] = 0.1 
nx.set_edge_attributes(G, edge_attrs_thickness, "edge_thickness")


# Set Edge style
edge_attrs_line_style = {}
for start_node, end_node, edge_value in G.edges(data=True):
    edge_style = "solid"
    if str(edge_value['key']) == 'Concept':
        edge_style = 'dashed'
    edge_attrs_line_style[(start_node, end_node)] = edge_style 
nx.set_edge_attributes(G, edge_attrs_line_style, "edge_style")



# Using iGraph libeary for Network layout , NX library layout not 
print("Initiating Bokeh visualisation ...")
h = ig.Graph.from_networkx(G)
layout_igraph = h.layout_kamada_kawai(seed = [i[1] for i in nx.spring_layout(G,k=0.2, iterations=500,seed =99).items()])
#Choose colors for node and edge highlighting
node_highlight_color = 'white'
# assign Node index 
node_indices = [n for n in G.nodes]

#Choose a title!
title = 'Epiverse Connect'

TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="42" alt="" width="42"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@index</span>
        </div>
        <div>
            <span style="font-size: 15px;">@Description</span>
        </div>
        <div>
            <span style="font-size: 15px;"><b>Scope</b>: @Scope</span>
        </div>
        <div>
            <span style="font-size: 13px;"><br><b>Github</b>: @{Link to Github}</span>
        </div>
        <div>
            <span style="font-size: 13px;"><b>License</b>: @{License}</span>
        </div>
        <div>
            <span style="font-size: 13px;"><b>Website</b>: @Website</span>
        </div>  
        <div>
            <span style="font-size: 13px;"><b>Maintainer</b>: @{Maintainer Email}</span>
        </div>          
    </div>
"""


#Create a plot — set dimensions, toolbar, and title
plot = figure(tooltips = TOOLTIPS,
              tools="pan,wheel_zoom,reset", active_scroll='wheel_zoom',
            x_range=Range1d(-60, 60), y_range=Range1d(-100, 100), 
              title=title, plot_width=1400, plot_height=900
             )
plot.title.text_font_size = "25px"
plot.title.align = "center"
#Create a network graph object


node_x = []
node_y = []
for c in layout_igraph.coords:
    node_x.append(c[0]*20)
    node_y.append(c[1]*20)
graph_layout = dict(zip(node_indices, zip(node_x, node_y)))
# layout_igraph
network_graph = from_networkx(G, graph_layout, scale =10, as_directed = True)
#assign Node shape | change data_goods_list to data_soruce after finalizing 
network_graph.node_renderer.data_source.data['name'] = node_indices
pixel_url = "https://c.tadst.com/gfx/tiles/eclipse/20200705/2/0/2.png"
network_graph.node_renderer.data_source.data['imgs'] = [j if j!='none' else pixel_url for j in [i[1]['image_url'] for i in G.nodes(data=True)]]
network_graph.node_renderer.data_source.data['markers'] = ['triangle' if i in data_goods_list else 'circle' for i in node_indices]

# Plotting nodes
network_graph.node_renderer.glyph = Scatter(size=40, fill_color='colors', marker= 'markers',line_width=2)
#Set node highlight colors
network_graph.node_renderer.hover_glyph = Scatter(size =40, fill_color=node_highlight_color, line_width=2)
network_graph.node_renderer.selection_glyph = Scatter(size =40, fill_color=node_highlight_color, line_width=2)

#Set edge opacity and width
network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=6, line_color = 'edge_color',
                                             line_dash = 'edge_style' )
#Set edge highlight colors
network_graph.edge_renderer.selection_glyph = MultiLine(line_color='edge_color', line_width=2,line_join ='bevel')
network_graph.edge_renderer.hover_glyph = MultiLine(line_color='edge_color', line_width=2)
network_graph.edge_renderer.data_source.data['name'] = [n[2]['key'] for n in G.edges(data=True)]


#Highlight nodes and edges
network_graph.selection_policy = NodesAndLinkedEdges() 
network_graph.inspection_policy =  NodesAndLinkedEdges()
plot.renderers.append(network_graph)

#Add Labels
x, y = zip(*network_graph.layout_provider.graph_layout.values())
node_labels = list(G.nodes())
source = ColumnDataSource({'x': x, 'y': y, 'name': [node_labels[i] for i in range(len(x))]})
labels = LabelSet(x='x', y='y', text='name', source=source, 
                  background_fill_color= 'grey', text_font_size='17px',
                  background_fill_alpha=0.05,text_font ='helvetica',
                 text_alpha =1, text_align ="right", x_offset=50, y_offset=-40)
plot.renderers.append(labels)

# add the labels to the edge renderer data source
source = network_graph.edge_renderer.data_source
source.data['names'] = [n[2]['key'] for n in G.edges(data=True)] #["%d-%d" % (x, y) for (x,y) in zip(source.data['start'], source.data['end'])]

# create a transform that can extract and average the actual x,y positions
code = """
    const result = new Float64Array(xs.length)
    const coords = provider.get_edge_coordinates(source)[%s]
    for (let i = 0; i < xs.length; i++) {
        result[i] = (coords[i][0] + coords[i][1])/2
    }
    return result
"""
xcoord = CustomJSTransform(v_func=code % "0", args=dict(provider=network_graph.layout_provider, source=source))
ycoord = CustomJSTransform(v_func=code % "1", args=dict(provider=network_graph.layout_provider, source=source))

# Use the transforms to supply coords to a LabelSet
labels_edges = LabelSet(x=transform('start', xcoord),
                  y=transform('start', ycoord),
                  text='names', text_font_size="9px",
                  x_offset=2, y_offset=5,
                  source=source, render_mode='canvas',text_font ='arial')
plot.add_layout(labels_edges)
plot.xaxis.visible = False
plot.yaxis.visible = False
plot.grid.grid_line_alpha = 0.3

items = []
for label,col in color_dict.items():
    items += [(label,[plot.circle(color=col,size=4)])]
items += [("To retire",[plot.circle(color="#F04A4C",size=4)])]
plot.add_layout(Legend(items=items),'right')
plot.legend.title = "Legend"

items_nodes = []
items_nodes += [("Data Sources",[plot.triangle(color="white",size=20,line_color='black',line_width=2)])]
items_nodes += [("R Packages",[plot.circle(color="white",size=20,line_color='black',line_width=2)])]
plot.add_layout(Legend(items=items_nodes))

# # change appearance of legend text
plot.legend.label_text_font = "arial"
plot.legend.label_text_font_size = "20px"
plot.legend.label_text_color = "navy"

# # change border and background of legend
plot.legend.border_line_width = 3
plot.legend.border_line_color = "grey"
plot.legend.border_line_alpha = 0.8
plot.legend.background_fill_color = "white"
plot.legend.background_fill_alpha = 0.2
plot.legend.padding = 10
plot.legend.title_text_font_size = "15px"
plot.legend.glyph_height =40
plot.legend.glyph_width =30
plot.legend.border_line_alpha = 0.2

# Changing outline 
plot.outline_line_width = 7
plot.outline_line_alpha = 0.3
plot.outline_line_color = "navy"

div = Div(text='', width=500)

plot.add_tools(TapTool(callback=CustomJS(args={'div': div}, code="""
    String.prototype.format = function() {
        var a = this;
        for (var k in arguments) {
            a = a.replace(new RegExp("\\{" + k + "\\}", 'g'), arguments[k]);
            }
            return a
        }
        var ind = cb_data.source.selected['1d'].indices;
        var color = cb_data.source.data['color'][ind];
        var name = cb_data.source.data['index'][ind];
        div.text = 'temp test';
        """)))


x_graph, y_graph = [v[0] for v in graph_layout.values()], [v[1] for v in graph_layout.values()]
node_ds = ColumnDataSource(data=dict(index=list(G.nodes()),
                                     x = x_graph,
                                     y = y_graph,
                                     color=[c for c in colors.values()]),
                           name="Node Renderer")

plot.js_on_event('tap', CustomJS(args={'src': node_ds, 'div': div}, code="""
    if (src.selected.indices.length == 0){
        div.text = '';
    }
"""))

#plot.renderers.append(network_graph)
layout=row(plot,div)

# Adding arrow heads to the edges 
edge_coordinates = tuple(zip(airtable_dataframe_dataflow['from_package'].to_list(),
                       airtable_dataframe_dataflow['to_package'].to_list()))

x_edges_start = [graph_layout[xs][0] for xs in [i for i,j in edge_coordinates]]
x_edges_end = [graph_layout[xe][0] for xe in [j for i,j in edge_coordinates]]

y_edges_start = [graph_layout[ys][1] for ys in [i for i,j in edge_coordinates]]
y_edges_end = [graph_layout[ye][1] for ye in [j for i,j in edge_coordinates]]

for xs,xe,ys,ye in zip(x_edges_start,x_edges_end,y_edges_start,y_edges_end):
    plot.add_layout(Arrow(end=VeeHead(size=10,fill_color="#888888"), line_color="#888888",
                   x_start=xs, y_start=ys, x_end=xe, y_end=ye,line_alpha=0.7))

print("Exporting HTML file.")
save(plot, filename="index.html")

logger.info("Exported. That's all folks!")





