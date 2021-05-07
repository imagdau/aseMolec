import numpy as np
import hashlib

#creates unique hash for a matrix of numbers
def hash_array(v):
    return hashlib.md5(np.array2string(v, precision=8, sign='+', floatmode='fixed').encode()).hexdigest()

#creates unique hash for Atoms from atomic numbers and positions
def hash_atoms(db):
    for at in db:
        v = np.concatenate((at.numbers.reshape(-1,1), at.positions),axis=1)
        at.info['uid'] = hash_array(v)

#prints all available properties in list of Atoms
def check_keys(db):
    for at in db:
        print([at.info['config_type']]+list(at.info.keys())+list(at.arrays.keys()))

#selects configurations which have property
def sel_by_prop(db, prop):
    reflist = []
    for at in db:
        props = list(at.info.keys())+list(at.arrays.keys())
        if prop in props:
            reflist.append(at)
    return reflist

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
        keys = list(at.info.keys())
        for k in keys:
            if 'energy' in k:
                del at.info[k]
            if 'virial' in k:
                del at.info[k]
        keys = list(at.arrays.keys())
        for k in keys:
            if 'force' in k:
                del at.arrays[k]

#deletes all properties which contain tag
def del_prop_by_tag(db, tag):
    for at in db:
        keys = list(at.info.keys())
        for k in keys:
            if tag in k:
                del at.info[k]
        keys = list(at.arrays.keys())
        for k in keys:
            if tag in k:
                del at.arrays[k]

#returns desired property for list of Atoms
def get_prop(db, type, prop=''):
    if type == 'info':
        return np.array(list(map(lambda a : a.info[prop], db)))
    if type == 'arrays':
        return np.array(list(map(lambda a : a.arrays[prop], db)), dtype=object)
    if type == 'cell':
        return np.array(list(map(lambda a : a.cell, db)))
    if type == 'meth':
        return np.array(list(map(lambda a : getattr(a, prop)(), db)))
