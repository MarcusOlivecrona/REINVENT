from bokeh.plotting import figure, ColumnDataSource, curdoc
from bokeh.models import CustomJS, Range1d
from bokeh.models.glyphs import Text
from bokeh.layouts import row, column, widgetbox, layout
from bokeh.models.widgets import Div
import bokeh.palettes
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import rdBase

import sys
import os.path
import numpy as np
import math

"""Bokeh app that visualizes training progress for the De Novo design reinforcement learning.
   The app is updated dynamically using information that the train_agent.py script writes to a
   logging directory."""

rdBase.DisableLog('rdApp.error')

error_msg = """Need to provide valid log directory as first argument.
                     'bokeh serve . --args [log_dir]'"""
try:
    path = sys.argv[1]
except IndexError:
    raise IndexError(error_msg)
if not os.path.isdir(path):
    raise ValueError(error_msg)

score_source = ColumnDataSource(data=dict(x=[], y=[], y_mean=[]))
score_fig = figure(title="Scores", plot_width=600, plot_height=600)
score_fig.line('x', 'y', legend='Average score', source=score_source)
score_fig.line('x', 'y_mean', legend='Running average of average score', line_width=2, 
               color="firebrick", source=score_source)

score_fig.xaxis.axis_label = "Step"
score_fig.yaxis.axis_label = "Average Score"
score_fig.title.text_font_size = "20pt"
score_fig.legend.location = "bottom_right"
score_fig.css_classes = ["score_fig"]

img_fig = Div(text="", width=850, height=590)
img_fig.css_classes = ["img_outside"]

def downsample(data, max_len):
    np.random.seed(0)
    if len(data)>max_len:
        data = np.random.choice(data, size=max_len, replace=False)
    return data

def running_average(data, length):
    early_cumsum = np.cumsum(data[:length]) / np.arange(1, min(len(data), length) + 1)
    if len(data)>length:
        cumsum = np.cumsum(data) 
        cumsum =  (cumsum[length:] - cumsum[:-length]) / length
        cumsum = np.concatenate((early_cumsum, cumsum))
        return cumsum
    return early_cumsum

def create_bar_plot(init_data, title):
    init_data = downsample(init_data, 50)
    x = range(len(init_data))
    source = ColumnDataSource(data=dict(x= [], y=[]))
    fig = figure(title=title, plot_width=300, plot_height=300)
    fig.vbar(x=x, width=1, top=init_data, fill_alpha=0.05)
    fig.vbar('x', width=1, top='y', fill_alpha=0.3, source=source)
    fig.y_range = Range1d(min(0, 1.2 * min(init_data)), 1.2 * max(init_data))
    return fig, source

def create_hist_plot(init_data, title):
    source = ColumnDataSource(data=dict(hist=[], left_edge=[], right_edge=[]))
    init_hist, init_edge = np.histogram(init_data, density=True, bins=50)
    fig = figure(title=title, plot_width=300, plot_height=300)
    fig.quad(top=init_hist, bottom=0, left=init_edge[:-1], right=init_edge[1:],
            fill_alpha=0.05)
    fig.quad(top='hist', bottom=0, left='left_edge', right='right_edge',
            fill_alpha=0.3, source=source)
    return fig, source


weights = [f for f in os.listdir(path) if f.startswith("weight")]
weights = {w:{'init_weight': np.load(os.path.join(path, "init_" + w)).reshape(-1)} for w in weights}

for name, w in weights.items():
    w['bar_fig'], w['bar_source'] = create_bar_plot(w['init_weight'], name)
    w['hist_fig'], w['hist_source'] = create_hist_plot(w['init_weight'], name + "_histogram")

bar_plots = [w['bar_fig'] for name, w in weights.items()]
hist_plots = [w['hist_fig'] for name, w in weights.items()]

layout = layout([[img_fig, score_fig], bar_plots, hist_plots], sizing_mode="fixed")
curdoc().add_root(layout)

def update():
    score = np.load(os.path.join(path, "Scores.npy"))
    with open(os.path.join(path, "SMILES"), "r") as f:
        mols = []
        scores = []
        for line in f:
                line = line.split()
                mol = Chem.MolFromSmiles(line[0])
                if mol and len(mols)<6:
                    mols.append(mol)
                    scores.append(line[1])
    img = Draw.MolsToGridImage(mols, molsPerRow=3, legends=scores, subImgSize=(250,250), useSVG=True)
    img = img.replace("FFFFFF", "EDEDED")
    img_fig.text = '<h2>Generated Molecules</h2>' + '<div class="img_inside">' + img + '</div>'
    score_source.data = dict(x=score[0], y=score[1], y_mean=running_average(score[1], 50))

    for name, w in weights.items():
        current_weights = np.load(os.path.join(path, name)).reshape(-1)
        hist, edge = np.histogram(current_weights, density=True, bins=50)
        w['hist_source'].data = dict(hist=hist, left_edge=edge[:-1], right_edge=edge[1:])
        current_weights = downsample(current_weights, 50)
        w['bar_source'].data = dict(x=range(len(current_weights)), y=current_weights)

update()
curdoc().add_periodic_callback(update, 1000)

