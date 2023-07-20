import numpy as np
import hashlib
import ase.units

#creates unique hash for a matrix of numbers
def hash_array(v):
    return hashlib.md5(np.array2string(v, precision=8, sign='+', floatmode='fixed').encode()).hexdigest()

#creates unique hash for Atoms from atomic numbers and positions
def hash_atoms(db):
    for at in db:
        v = np.concatenate((at.numbers.reshape(-1,1), at.positions),axis=1)
        at.info['uid'] = hash_array(v)

# __eq__(self, other) implemented in atoms.py envokes really high precision
def equal(self, other, prec=1e-12):
    a = self.arrays
    b = other.arrays
    return (len(self) == len(other) and
            (np.abs(a['positions']-b['positions'])<prec).all() and
            (a['numbers'] == b['numbers']).all() and
            (np.abs((self.cell-other.cell))<prec).all() and
            (self.pbc == other.pbc).all())

#prints all available properties in list of Atoms
def check_keys(db):
    for at in db:
        if 'config_type' in at.info:
            print([at.info['config_type']]+list(at.info.keys())+list(at.arrays.keys()))
        else:
            print(list(at.info.keys())+list(at.arrays.keys()))

#selects configurations which have property
def sel_by_prop(db, prop):
    reflist = []
    for at in db:
        props = list(at.info.keys())+list(at.arrays.keys())
        if prop in props:
            reflist.append(at)
    return reflist

#select configurations which have an info field with certain value
def sel_by_info_val(db, info_key, info_val):
    sel = []
    for at in db:
        if info_key in at.info.keys():
            if at.info[info_key] == info_val:
                sel += [at]
    return sel

#selects configurations by uid
def sel_by_uid(db, uid):
    reflist = []
    for at in db:
        if uid == at.info['uid']:
            reflist.append(at)
    return reflist

#selects configurations of a certain config_type
def sel_by_conf_type(db, config_type):
    reflist = []
    for at in db:
        if (at.info['config_type'] == config_type):
            reflist.append(at)
    return reflist

#deletes all properties except for coordinates and identification
def del_prop(db):
    for at in db:
        if at.calc:
            del at.calc
        keys = list(at.info.keys())
        for k in keys:
            if 'energy' in k:
                del at.info[k]
            if 'stress' in k:
                del at.info[k]
            if 'virial' in k:
                del at.info[k]
        keys = list(at.arrays.keys())
        for k in keys:
            if 'momenta' in k:
                del at.arrays[k]
            if 'force' in k:
                del at.arrays[k]

#deletes all properties which contain tag
def del_prop_by_tag(db, tag):
    for at in db:
        if at.calc:
            del at.calc
        keys = list(at.info.keys())
        for k in keys:
            if tag in k:
                del at.info[k]
        keys = list(at.arrays.keys())
        for k in keys:
            if tag in k:
                del at.arrays[k]

#renames prop tag, eg: energy_PBE -> energy_dft
def rename_prop_tag(db, oldtag, newtag):
    for at in db:
        keys = list(at.info.keys())
        for k in keys:
            if oldtag in k:
                at.info[k.replace(oldtag, newtag)] = at.info.pop(k)
        keys = list(at.arrays.keys())
        for k in keys:
            if oldtag in k:
                at.arrays[k.replace(oldtag, newtag)] = at.arrays.pop(k)

def get_E0(db, tag=''):
    E0 = {}
    for at in db:
        if len(at)==1:
            E0[at.get_chemical_symbols()[0]]=at.info['energy'+tag]
    return E0

def get_density_gcm3(at):
    densfact = (ase.units.m/1.0e2)**3/ase.units.mol
    return np.sum(at.get_masses())/at.get_volume()*densfact

#returns desired property for list of Atoms
def get_prop(db, type, prop='', peratom=False, E0={}):
    if peratom:
        N = lambda a : a.get_global_number_of_atoms()
    else:
        N = lambda a : 1
    if type == 'info':
        return np.array(list(map(lambda a : a.info[prop]/N(a), db)))
    if type == 'arrays':
        return np.array(list(map(lambda a : a.arrays[prop]/N(a), db)), dtype=object)
    if type == 'cell':
        return np.array(list(map(lambda a : a.cell/N(a), db)))
    if type == 'meth':
        return np.array(list(map(lambda a : getattr(a, prop)()/N(a), db)))
    if type == 'atom':
        if not E0:
            E0 = get_E0(db, prop)
        return np.array(list(map(lambda a : (np.sum([E0[s] for s in a.get_chemical_symbols()]))/N(a), db)))
    if type == 'bind':
        if not E0:
            E0 = get_E0(db, prop)
        return np.array(list(map(lambda a : (a.info['energy'+prop]-np.sum([E0[s] for s in a.get_chemical_symbols()]))/N(a), db)))

def set_prop(db, type, prop, tag):
    for i,at in enumerate(db):
        if type == 'info':
            at.info[tag] = prop[i]
        if type == 'arrays':
            at.arrays[tag] = prop[i]

def calc_virial(db, tag='', keep=False, convstr=False):
    for at in db:
        if keep:
            at.info['virial'+tag] = -at.info['stress'+tag]*at.get_volume()
        else:
            at.info['virial'+tag] = -at.info.pop('stress'+tag)*at.get_volume()
        if convstr:
            at.info['virial'+tag] = ' '.join(map(str, at.info.pop('virial'+tag).reshape(-1,order='F')))

def split_db(db, N=150, seed=12345):
    db_ia = sel_by_conf_type(db, 'IsolatedAtoms')
    db_im = sel_by_conf_type(db, 'IsolatedMolecules')
    db_lc = sel_by_conf_type(db, 'LiquidConfigs')
    per = np.random.RandomState(seed=seed).permutation(len(db_lc))
    db1 = db_ia.copy()
    db2 = db_ia.copy()
    for i in range(len(db_lc)):
        frame_lc = db_lc[i]
        frame_im = sel_by_uid(db_im, frame_lc.info['uid'])
        if i in per[:N]:
            db1 += [frame_lc]
            db1 += frame_im
        else:
            db2 += [frame_lc]
            db2 += frame_im
    return db1, db2
