import ipywidgets as widgets
from ipywidgets import fixed
from qsmp.viz import viz

style = {'description_width': 'initial'}

quantileRangeSlider = widgets.FloatRangeSlider(
    value=[0.5, 0.99],
    min=0.5, max=1, step=0.001,
    description='Distance quantiles',
    readout_format='.3f',
    continous_update=False,
    layout=widgets.Layout(width='70%')
)

nDistSlider = widgets.IntSlider(
    value=5,
    min=1, max=10, step=1,
    description='# of distances',
    continous_update=False,
    style=style
)

maxModesSlider = widgets.IntSlider(
    value=3,
    min=1, max=20, step=1,
    description="Max. # of modes",
    continous_update=False,
    style=style
)

nNeighSlider = widgets.IntSlider(
    value=3,
    min=1, max=10, step=1,
    description="Max. # of nearest neighbors",
    continous_update=False,
    style=style
)

showNeighCheckBox = widgets.Checkbox(
    value=False,
    description="Show neighbors",
    indent=False,
    disabled=False,
)

def modes_across_maxdist(
    T, wave_len, density, NNindex, NNdist, sigmas):

    sigmaDropDown = widgets.Dropdown(
        options=sigmas,
        value=0,
        description='sigma',
        layout=widgets.Layout(width='20%')
    )

    out = widgets.interactive_output(
        viz.show_modes_across_maxdist, dict(
            T=fixed(T), wave_len=fixed(wave_len), density=fixed(density),
            NNindex=fixed(NNindex), NNdist=fixed(NNdist),
            q=quantileRangeSlider, n_dist=nDistSlider, sigma=sigmaDropDown, max_modes=maxModesSlider, n_neighbors=nNeighSlider,
            show_neighbors=showNeighCheckBox)
    )

    ui = widgets.VBox([
        quantileRangeSlider,
        widgets.HBox([nDistSlider, sigmaDropDown]),
        widgets.HBox([maxModesSlider, nNeighSlider, showNeighCheckBox]),
        out
    ]
    )
    return ui
