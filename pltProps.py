import matplotlib.pyplot as plt
import numpy as np

# ### Standard Error of the Mean
# a = np.random.normal(size=100000)
# N = 10
# win = int(100000/N)
# m = np.mean(a.reshape([N,win]), axis=1)
# print(m.size)
# print(np.std(m))
# print(np.std(m)*np.sqrt(win))
# print(np.std(a))
def stats(v, win=1):
    N = np.floor(v.size/win).astype(int)
    v_win = v[:N*win].reshape(N,win)
    means = np.mean(v_win, axis=1)
    return np.mean(means), np.std(means), np.around((N*win*100)/v.size, 2)

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
    return RMSE

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
    if 'Nsamp' in kwargs.keys():
        Nsamp = kwargs['Nsamp']
    else:
        Nsamp = 1000
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
        plt.plot(thermo[ymin:ymax,0]/Nsamp, y, label=lb)
        if 'sel' in kwargs.keys():
            sel = kwargs['sel']
            if (i+1) in sel.keys():
                plt.scatter(thermo[sel[i+1],0]/Nsamp, y[np.array(sel[i+1])-ymin], marker='o', color='C{}'.format(i), s=50)
        i += 1
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    if lb:
        plt.legend()

# col, start, bins, legend, labs, title
def plot_hist(fnames, **kwargs):
    avgs = []
    stds = []
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
    if 'orient' in kwargs.keys():
        orientation=kwargs['orient']
    else:
        orientation='vertical'
    if 'htype' in kwargs.keys():
        htype = kwargs['htype']
    else:
        htype = 'step'
    if 'Navg' in kwargs.keys():
        Navg = kwargs['Navg']
    else:
        Navg = 1
    if 'density' in kwargs.keys():
        density = kwargs['density']
    else:
        density = False
    for f in fnames:
        thermo = np.loadtxt(f)
        if 'legend' in kwargs.keys():
            lb = kwargs['legend'][i]
        else:
            lb = None
        result = plt.hist(thermo[start:,col], bins=b, histtype=htype, label=lb, orientation=orientation, alpha=0.7, density=density)
        centers = result[1][:-1]+np.diff(result[1])/2
        counts = result[0]
        # avg = np.sum(centers*counts)/np.sum(counts)
        # std = np.sqrt(np.sum(((centers-avg)**2)*counts)/np.sum(counts))
        # avg = np.mean(thermo[start:,col])
        # std = np.std(thermo[start:,col])
        avg, std, _ = stats(thermo[start:,col], Navg)
        avgs += [avg]
        stds += [std]
        if 'sel' in kwargs.keys():
            sel = kwargs['sel']
            if (i+1) in sel.keys():
                counts = np.histogram(thermo[start:,col], bins=b)
                ids = np.argmin(np.abs(counts[1][:-1].reshape(-1,1)-thermo[sel[i+1],col]), axis=0)
                plt.scatter(counts[1][ids], counts[0][ids], marker='o', color='C{}'.format(i), s=50)
        i += 1
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    if lb:
        plt.legend()
    return np.array(avgs), np.array(stds)

def plot_menvs(menvs, lb, **kwargs):
    if 'nbins' in kwargs.keys():
        nbins = kwargs['nbins']
    else:
        nbins = np.max(menvs[lb])
    bins = np.vstack([np.array(range(nbins+1))]*menvs[lb].shape[1])
    counts, coords = np.histogramdd(menvs[lb], bins=bins)
    #later could expend to more dimensions, for now just implement for 2
    #for more than 2, need to make a choice on how to project in lower dimension
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap = 'viridis'
    plt.pcolormesh(coords[0]-0.5, coords[1]-0.5, counts.T, cmap=cmap, edgecolors='grey')
    plt.xticks(coords[0][:-1])
    plt.yticks(coords[1][:-1])
    if 'style' in kwargs.keys():
        if kwargs['style']=='cbar':
            plt.colorbar()
        if kwargs['style']=='nums':
            for i in range(nbins):
                for j in range(nbins):
                    plt.text(i,j,'{0:d}'.format(int(counts[i, j])), ha='center', va='center')
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
