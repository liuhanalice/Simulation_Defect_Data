import numpy as np
import matplotlib.pyplot as plt


def plotsample(data, Y=None, plot_scale=None, view=[30,-120], title=None, 
               show=True, save=False, filepath=None, label_mapping=None):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d', proj_type = 'ortho')

    if plot_scale is not None:
        ax.set_box_aspect(plot_scale)
    else:
        ax.set_box_aspect([np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])])
    
    if Y is not None:
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=Y, s=5)
        legends = np.unique(Y)
    else:
        scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(*view)
    if title is not None:
        plt.title(title)
    
    if label_mapping is not None:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label_mapping[label], 
                    markerfacecolor=plt.cm.viridis(label / max(Y)), markersize=10) 
                    for label in np.unique(Y)]
        ax.legend(handles=handles)
    
    if save:
        if filepath is None:
            filepath = 'sample.png'
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()

def plotsample_cmap(data, Y, plot_scale=None, view=[30,-120], title=None, clim=None,
                    show=True, save=False, filepath=None, legend=False):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(projection='3d')

    if plot_scale is not None:
        ax.set_box_aspect(plot_scale)
    else:
        ax.set_box_aspect([np.ptp(data[:, 0]), np.ptp(data[:, 1]), np.ptp(data[:, 2])])

    if clim is not None:
        vmin, vmax = clim[0], clim[1]
    else:
        vmin, vmax = min(Y), max(Y)
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=Y, s=5, cmap='viridis', vmin=vmin, vmax=vmax)
    # set vmin and vmax to fix the colorbar
    
    # colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Class')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(*view)
    if title is not None:
        plt.title(title)
    
    if legend:
        # legend
        # customize the legend bar size
        plt.legend(shrink=0.5)

    if save:
        if filepath is None:
            filepath = 'sample.png'
        plt.savefig(filepath)

    if show:
        plt.show()
    else:
        plt.close()
