import matplotlib.pyplot as plt


def line_graph(
    values,
    labels,
    x_guides=None,
    x_name=None,
    y_name=None,
    x_min_max=None,
    y_min_max=None,
    legend_loc=None,
    subplot=None,
    plot_size=(5, 5),
):
    """Plot line graph(s).
    TODO add test
    Args:
        values (list(list(float or tuple)) or list(float or tuple): List of graphs or a graph to plot
            E.g. a graph = list(y) or list((y,x))
        labels (list(str) or str): List of labels or a label for graph.
            If labels is a string, this function assumes the values is a single graph.
        x_guides (list(int)): List of guidelines (a vertical dotted line)
        x_name (str): x axis label
        y_name (str): y axis label
        x_min_max (list or tuple): Min and max value of the x axis
        y_min_max (list or tuple): Min and max value of the y axis
        legend_loc (str): legend location
        subplot (list or tuple): `matplotlib.pyplot.subplot` format. E.g. to draw 1 x 2 subplot,
            pass `(1,2,1)` for the first subplot and `(1,2,2)` for the second subplot.
        plot_size (list or tuple): Plot size (width, height)
    """
    if subplot:
        # Setup figure only once
        if subplot[2] == 1:
            if plot_size:
                plt.figure(figsize=(
                    plot_size[0]*subplot[1],  # fig width = plot width * num columns
                    plot_size[1]*subplot[0]   # fig height = plot height * num rows
                ))
            plt.subplots_adjust(wspace=0.5)
        plt.subplot(*subplot)
    else:
        if plot_size:
            plt.figure(figsize=plot_size)

    if isinstance(labels, str):
        if isinstance(values[0], (int, float)):
            y, x = values, range(len(values))
        else:
            y, x = zip(*values)
        plt.plot(x, y, label=labels, lw=1)
    else:
        assert len(values) == len(labels)
        for i, v in enumerate(values):
            if isinstance(v[0], (int, float)):
                y, x = v, range(len(v))
            else:
                y, x = zip(*v)
            plt.plot(x, y, label=labels[i], lw=1)

    if x_guides:
        for x in x_guides:
            plt.axvline(x=x, color="gray", lw=1, linestyle="--")

    if x_name:
        plt.xlabel(x_name)
    if y_name:
        plt.ylabel(y_name)
    if x_min_max:
        plt.xlim(*x_min_max)
    if y_min_max:
        plt.ylim(*y_min_max)
    if legend_loc:
        plt.legend(loc=legend_loc)
