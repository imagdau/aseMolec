import matplotlib.pyplot as plt
import numpy as np

def plot_prop(prop1, prop2, **kwargs):
    lmin = min(min(prop1), min(prop2))
    lmax = max(max(prop1), max(prop2))
    RMSE = np.sqrt(np.mean((prop1-prop2)**2))
    if 'cols' in kwargs.keys():
        cols = kwargs['cols']
    else:
        cols = None
    plt.scatter(prop1, prop2, s=3.0, c=cols, label="RMSE = %.4f" % RMSE)
    plt.xlim([lmin, lmax])
    plt.ylim([lmin, lmax])
    plt.plot([lmin, lmax], [lmin, lmax], '--', linewidth=1, color='gray')
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().ticklabel_format(useOffset=False)
