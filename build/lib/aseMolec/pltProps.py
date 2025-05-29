import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from matplotlib.ticker import StrMethodFormatter
import numpy as np
from aseMolec import extAtoms as ea
import re

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
    RRMSE = RMSE/np.sqrt(np.mean((prop1-np.mean(prop1))**2))
    if 'cols' in kwargs.keys():
        cols = kwargs['cols']
    else:
        cols = None
    if 'cmap' in kwargs.keys():
        cmap = kwargs['cmap']
    else:
        cmap = None
    if 'rel' in kwargs.keys():
        rel = kwargs['rel']
    else:
        rel = False
    if 'return_rel' in kwargs.keys():
        return_rel = kwargs['return_rel']
    else:
        return_rel = False
    if 'rast' in kwargs.keys():
        rast = kwargs['rast']
    else:
        rast = False
    if 'xy' in kwargs.keys():
        xy = kwargs['xy']
    else:
        xy = [0.35,0.04]
    plt.scatter(prop1, prop2, s=3.0, c=cols, cmap=cmap, rasterized=rast)
    if rel:
        plt.text(xy[0], xy[1], "  RMSE = {0:.4f}\nRRMSE = {1:.4f}".format(RMSE, RRMSE), transform=plt.gca().transAxes)
    else:
        plt.text(xy[0], xy[1], "  RMSE = {0:.4f}".format(RMSE), transform=plt.gca().transAxes)
    plt.xlim([lmin, lmax])
    plt.ylim([lmin, lmax])
    plt.plot([lmin, lmax], [lmin, lmax], '--', linewidth=1, color='gray')
    if 'title' in kwargs.keys():
        plt.title(kwargs['title'])
    if 'labs' in kwargs.keys():
        plt.xlabel(kwargs['labs'][0])
        plt.ylabel(kwargs['labs'][1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.gca().ticklabel_format(useOffset=False)
    if return_rel:
        return RMSE, RRMSE
    else:
        return RMSE

# col, Navg, legend, labs, title
def plot_traj(fnames, **kwargs):
    i = 0
    if 'col' in kwargs.keys():
        col = kwargs['col']
    else:
        col = 1
    if 'colors' in kwargs.keys():
        colors = kwargs['colors']
    else:
        colors = np.array(cm.get_cmap('tab10').colors)
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 1.0
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
        plt.plot(thermo[ymin:ymax,0]/Nsamp, y, label=lb, color=colors[i,:], alpha=alpha)
        if 'sel' in kwargs.keys():
            sel = kwargs['sel']
            if (i+1) in sel.keys():
                plt.scatter(thermo[sel[i+1],0]/Nsamp, y[np.array(sel[i+1])-ymin], marker='o', color=colors[i,:], s=50)
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
    if 'colors' in kwargs.keys():
        colors = kwargs['colors']
    else:
        colors = np.array(cm.get_cmap('tab10').colors)
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 0.7
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
    if 'scale' in kwargs.keys():
        scale = kwargs['scale']
    else:
        scale = 1
    for f in fnames:
        thermo = np.loadtxt(f)
        if 'legend' in kwargs.keys():
            lb = kwargs['legend'][i]
        else:
            lb = None
        result = plt.hist(thermo[start:,col]/scale, bins=b, histtype=htype, label=lb, orientation=orientation, alpha=alpha, density=density, color=colors[i,:])
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

def plot_traj_hist(fnames, col=2, Navg=1, start=0, bins=50, legend=None, labs=None, ylims=None, colors=np.array([]), title=None, fs=10, Nsamp=4000):
    plt.rcParams.update({'font.size': fs})
    if not colors.size:
        colors = np.array(cm.get_cmap('tab10').colors)
    gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1], wspace=0)
    plt.subplot(gs[0])
    if legend:
        plot_traj(fnames, col=col, Navg=Navg, legend=legend, labs=labs[0:2], title=title, colors=colors, alpha=0.8)
    else:
        plot_traj(fnames, col=col, Navg=Navg, labs=labs[0:2], title=title, colors=colors, alpha=0.8)
    if ylims:
        plt.ylim(ylims)
    ylim = plt.gca().get_ylim()
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
    plt.subplot(gs[1])
    avgs, errs = plot_hist(fnames, col=col, start=start, bins=bins, labs=labs[1:3], title=None, orient='horizontal', htype='stepfilled', colors=colors, alpha=0.8, Navg=Nsamp)
    plt.ylim(ylim)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.tight_layout()
    return avgs, errs

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

def plot_hist_thermo(thermos, **kwargs):
    avgs = []
    stds = []
    i = 0
    if 'col' in kwargs.keys():
        col = kwargs['col']
    else:
        col = 1
    if 'colors' in kwargs.keys():
        colors = kwargs['colors']
    else:
        colors = np.array(cm.get_cmap('tab10').colors)
    if 'alpha' in kwargs.keys():
        alpha = kwargs['alpha']
    else:
        alpha = 0.7
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
    if 'scale' in kwargs.keys():
        scale = kwargs['scale']
    else:
        scale = 1
    for thermo in thermos:
        if 'legend' in kwargs.keys():
            lb = kwargs['legend'][i]
        else:
            lb = None
        result = plt.hist(thermo[start:,col]/scale, bins=b, histtype=htype, label=lb, orientation=orientation, alpha=alpha, density=density, color=colors[i,:])
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

def plot_intra_inter(db1, db2, labs):
    plt.figure(figsize=(15,12), dpi=200)
    plt.subplot(3,3,1)
    plot_prop(ea.get_prop(db1, 'info', 'energy_intram', True).flatten(), \
              ea.get_prop(db2, 'info', 'energy_intram', True).flatten(), \
              title=r'Intra Energy $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.subplot(3,3,2)
    plot_prop(ea.get_prop(db1, 'info', 'energy_interm', True).flatten(), \
              ea.get_prop(db2, 'info', 'energy_interm', True).flatten(), \
              title=r'Inter Energy $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.subplot(3,3,3)
    plot_prop(ea.get_prop(db1, 'info', 'energy', True).flatten(), \
              ea.get_prop(db2, 'info', 'energy', True).flatten(), \
              title=r'Total Energy $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.subplot(3,3,4)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces_intram')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces_intram')).flatten(), \
              title=r'Intra Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(3,3,5)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces_interm')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces_interm')).flatten(), \
              title=r'Inter Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(3,3,6)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces')).flatten(), \
              title=r'Total Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(3,3,7)
    plot_prop(ea.get_prop(db1, 'info', 'virial_intram', True).flatten(), \
              ea.get_prop(db2, 'info', 'virial_intram', True).flatten(), \
              title=r'Intra Virial $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.subplot(3,3,8)
    plot_prop(ea.get_prop(db1, 'info', 'virial_interm', True).flatten(), \
              ea.get_prop(db2, 'info', 'virial_interm', True).flatten(), \
              title=r'Inter Virial $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.subplot(3,3,9)
    plot_prop(ea.get_prop(db1, 'info', 'virial', True).flatten(), \
              ea.get_prop(db2, 'info', 'virial', True).flatten(), \
              title=r'Total Virial $(\rm eV/atom)$ ', labs=labs, rel=True)
    plt.tight_layout()
    
def plot_trans_rot_vib(db1, db2, labs):
    plt.figure(figsize=(16,5), dpi=200)
    plt.subplot(1,4,1)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces_trans')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces_trans')).flatten(), \
              title=r'Translational Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(1,4,2)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces_rot')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces_rot')).flatten(), \
              title=r'Rotational Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(1,4,3)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces_vib')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces_vib')).flatten(), \
              title=r'Vibrational Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.subplot(1,4,4)
    plot_prop(np.concatenate(ea.get_prop(db1, 'arrays', 'forces')).flatten(), \
              np.concatenate(ea.get_prop(db2, 'arrays', 'forces')).flatten(), \
              title=r'Total Forces $\rm (eV/\AA)$ ', labs=labs, rel=True)
    plt.tight_layout()

def plot_intra_inter_energy(db_test, db_pred):
    E0_test = ea.get_E0(db_test)
    E0_pred = ea.get_E0(db_pred)
    db_test = ea.sel_by_conf_type(db_test, 'LiquidConfigs')
    db_pred = ea.sel_by_conf_type(db_pred, 'LiquidConfigs')

    RMSE = {}
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(8,8), dpi=200)
    plt.subplot(2,2,1)
    RMSE['IntraEnergy'] = plot_prop(ea.get_prop(db_test, 'bind', '_intram', True, E0_test).flatten(), \
                                    ea.get_prop(db_pred, 'bind', '_intram', True, E0_pred).flatten(), \
                                    title='Intra Energy (ev/atom) ', labs=['DFT', 'GAP'])
    plt.subplot(2,2,2)
    RMSE['InterEnergy'] = plot_prop(ea.get_prop(db_test, 'info', 'energy_interm', True).flatten(), \
                                    ea.get_prop(db_pred, 'info', 'energy_interm', True).flatten(), \
                                    title='Inter Energy (ev/atom) ', labs=['DFT', 'GAP'])
    plt.subplot(2,2,3)
    # RMSE['AtomEnergy'] = plot_prop(np.array([E0_test[k] for k in E0_test]), \
    #                                np.array([E0_pred[k] for k in E0_pred]), \
    #                                title='Atomic Energy (ev/atom) ', labs=['DFT', 'GAP'])
    RMSE['AtomEnergy'] = plot_prop(ea.get_prop(db_test, 'atom', peratom=True, E0=E0_test).flatten(), \
                                   ea.get_prop(db_pred, 'atom', peratom=True, E0=E0_pred).flatten(), \
                                   title='Atom Energy (ev/atom) ', labs=['DFT', 'GAP'])
    plt.subplot(2,2,4)
    RMSE['TotalEnergy'] = plot_prop(ea.get_prop(db_test, 'info', 'energy', True).flatten(), \
                                    ea.get_prop(db_pred, 'info', 'energy', True).flatten(), \
                                    title='Total Energy (ev/atom) ', labs=['DFT', 'GAP'])
    plt.tight_layout(pad=0.5)
    plt.savefig('energy.png')
    plt.close()
    return RMSE

def plot_intra_inter_forces(db_test, db_pred):
    db_test = ea.sel_by_conf_type(db_test, 'LiquidConfigs')
    db_pred = ea.sel_by_conf_type(db_pred, 'LiquidConfigs')
    elms = np.array([el for at in db_test for el in at.get_chemical_symbols()])
    Nelms = np.unique(elms).size

    RMSE = {}
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=((Nelms+1)*4,3*4), dpi=200)
    for i,el in enumerate(np.unique(elms)):
        plt.subplot(3,Nelms+1,i+1)
        RMSE['IntraForces'+el] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces_intram'))[elms==el,:].flatten(), \
                                           np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces_intram'))[elms==el,:].flatten(), \
                                           title=el+'\nIntra Forces (ev/A) ', labs=['DFT', 'GAP'])
        plt.subplot(3,Nelms+1,(Nelms+1)+i+1)
        RMSE['InterForces'+el] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces_interm'))[elms==el,:].flatten(), \
                                           np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces_interm'))[elms==el,:].flatten(), \
                                           title='Inter Forces (ev/A) ', labs=['DFT', 'GAP'])
        plt.subplot(3,Nelms+1,2*(Nelms+1)+i+1)
        RMSE['TotalForces'+el] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces'))[elms==el,:].flatten(), \
                                           np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces'))[elms==el,:].flatten(), \
                                           title='Total Forces (ev/A) ', labs=['DFT', 'GAP'])
    plt.subplot(3,Nelms+1,(Nelms+1))
    RMSE['IntraForces'] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces_intram')).flatten(), \
                                    np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces_intram')).flatten(), \
                                    title='Intra Forces (ev/A) ', labs=['DFT', 'GAP'])
    plt.subplot(3,Nelms+1,2*(Nelms+1))
    RMSE['InterForces'] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces_interm')).flatten(), \
                                    np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces_interm')).flatten(), \
                                    title='Inter Forces (ev/A) ', labs=['DFT', 'GAP'])
    plt.subplot(3,Nelms+1,3*(Nelms+1))
    RMSE['TotalForces'] = plot_prop(np.concatenate(ea.get_prop(db_test, 'arrays', 'forces')).flatten(), \
                                    np.concatenate(ea.get_prop(db_pred, 'arrays', 'forces')).flatten(), \
                                    title='Total Forces (ev/A) ', labs=['DFT', 'GAP'])
    plt.tight_layout(pad=0.5)
    plt.savefig('forces.png')
    plt.close()
    return RMSE

def plot_intra_inter_virial(db_test, db_pred):
    db_test = ea.sel_by_conf_type(db_test, 'LiquidConfigs')
    db_pred = ea.sel_by_conf_type(db_pred, 'LiquidConfigs')

    RMSE = {}
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(12,4), dpi=200)
    plt.subplot(1,3,1)
    RMSE['IntraVirial'] = plot_prop(ea.get_prop(db_test, 'info', 'virial_intram', True).flatten(), \
                                    ea.get_prop(db_pred, 'info', 'virial_intram', True).flatten(), \
                                    title='Intra Virial (ev/atom) ', labs=['DFT', 'GAP'])
    plt.subplot(1,3,2)
    RMSE['InterVirial'] = plot_prop(ea.get_prop(db_test, 'info', 'virial_interm', True).flatten(), \
                                    ea.get_prop(db_pred, 'info', 'virial_interm', True).flatten(), \
                                    title='Inter Virial (ev/atom) ', labs=['DFT', 'GAP'])
    plt.subplot(1,3,3)
    RMSE['TotalVirial'] = plot_prop(ea.get_prop(db_test, 'info', 'virial', True).flatten(), \
                                    ea.get_prop(db_pred, 'info', 'virial', True).flatten(), \
                                    title='Total Virial (ev/atom) ', labs=['DFT', 'GAP'])
    plt.tight_layout(pad=0.5)
    plt.savefig('virial.png')
    plt.close()
    return RMSE

def loadtxttag(fname):
    with open(fname, 'r') as file:
        comment = file.readline()
        header = file.readline().split()
        assert header[0] == '#'            
        fields = header[1:]
    dat = np.loadtxt(fname)
    db = dict()
    for i, aux in enumerate(fields):
        buf = re.findall('\((.*?)\)', aux)
        if buf:
            fld = aux[:-len(buf[0])-2]
            db[fld] = {'units':buf[0], 'data':dat[:,i]}
        else:
            fld = aux
            db[fld] = {'data':dat[:,i]}
    return db

def convert_units(dat, key, units, fact):
    for k in dat:
        dat[k][key]['units'] = units
        dat[k][key]['data'] *= fact

def rename_key(dat, key_old, key_new):
    for k in dat:
        dat[k][key_new] = dat[k].pop(key_old)

def simpleplot(db, i, j, byKey=False, **kwargs):
    if byKey:
        k1 = i
        k2 = j
    else:
        keys = list(db)
        k1 = keys[i]
        k2 = keys[j]
    if 'units' in db[k1]:
        u1 = ' ('+db[k1]['units']+')'
    else:
        u1 = ''
    if 'units' in db[k2]:
        u2 = ' ('+db[k2]['units']+')'
    else:
        u2 = ''
    if 'skip' in kwargs:
        skip = kwargs.pop('skip')
    else:
        skip = 0
    plt.plot(db[k1]['data'][skip:], db[k2]['data'][skip:], **kwargs)
    plt.xlabel(k1+u1)
    plt.ylabel(k2+u2)
    
def listdict_to_dictlist(dct):
    Ns = np.array([len(v) for v in dct.values() if type(v)==list])
    if Ns.size==0:
        Ns = np.array([1]) #edge case: none of the dct elements is a list
    assert np.all(Ns==Ns[0]) #check that all lists are the same size
    dlist = []
    for i in range(Ns[0]):
        d = {}
        for k in list(dct):
            if type(dct[k])==list:
                d[k] = dct[k][i]
            else: #if element is not a list, replicate values
                d[k] = dct[k]
        dlist += [d]
    return dlist

def multiplot(db, i, jcol, **kwargs):
    keys = list(db)
    unts = db[keys[jcol[0]]]['units']
    for k,j in enumerate(jcol):
        if kwargs:
            kwarg_list = listdict_to_dictlist(kwargs)
            assert len(kwarg_list)==len(jcol)
            simpleplot(db, i, j, label=keys[j], **kwarg_list[k])
        else:
            simpleplot(db, i, j, label=keys[j])
        assert unts==db[keys[j]]['units']
    plt.ylabel('Series ('+unts+')')
    plt.legend()
