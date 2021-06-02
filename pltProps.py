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

# col, Navg, legend, labs, title
def plot_traj(fnames, **kwargs):
    i = 0
    if 'col' in kwargs.keys():
        col = kwargs['col']
    else:
        col = 1
    if 'Navg' in kwargs.keys():
        N = kwargs['Navg']
    else:
        N = 1
    for f in fnames:
        thermo = np.loadtxt(f)
        y = np.convolve(thermo[:,col], np.ones(N)/N, mode='valid')
        ymin = round(np.ceil((N-1)/2))
        ymax = -round(np.floor((N-1)/2))
        if ymax == 0:
            ymax = thermo.shape[0]
        if 'legend' in kwargs.keys():
            lb = kwargs['legend'][i]
        else:
            lb = None
        plt.plot(thermo[ymin:ymax,0]/1000, y, label=lb)
        i += 1
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    plt.legend()

# col, start, bins, legend, labs, title
def plot_hist(fnames, **kwargs):
    i = 0
    if 'col' in kwargs.keys():
        col = kwargs['col']
    else:
        col = 1
    if 'bins' in kwargs.keys():
        b = kwargs['bins']
    else:
        b = 100
    if 'start' in kwargs.keys():
        start = kwargs['start']
    else:
        start = 0
    for f in fnames:
        thermo = np.loadtxt(f)
        if 'legend' in kwargs.keys():
            lb = kwargs['legend'][i]
        else:
            lb = None
        plt.hist(thermo[start:,col], bins=b, histtype='step', label=lb)
        i += 1
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    plt.legend()
