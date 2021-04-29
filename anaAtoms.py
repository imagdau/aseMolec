from ase import neighborlist
from ase import Atoms
import ase.geometry
from scipy import sparse
import numpy as np
import extAtoms as ea

#computes molIDs
def find_molecs(db, fct=1.0):
    for at in db:
        #from https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
        cutOff = neighborlist.natural_cutoffs(at, mult=fct)
        nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
        nbLst.update(at)
        conMat = nbLst.get_connectivity_matrix(sparse=True)
        Nmol, molID = sparse.csgraph.connected_components(conMat)
        at.arrays['molID'] = molID

#wraps single molecule: completes molecule over pbc and sfits COM back to unit cell
def wrap_molec(mol, fct=1.0):
    cutOff = neighborlist.natural_cutoffs(mol, mult=fct)
    nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    visited = []
    tovisit = [0]
    while tovisit:
        i = tovisit.pop(0)
        nbLst.update(mol)
        nbs, vecs = nbLst.get_neighbors(i)
        for j, v in zip(nbs, vecs):
            if (j not in visited) and (j not in tovisit):
                mol.positions[j,:] += np.dot(v, mol.cell)
                tovisit.append(j)
        visited.append(i)
    m = mol.get_masses()
    cm = np.sum(mol.positions*m.reshape(-1,1), axis=0)/np.sum(m)
    wrap_cm = ase.geometry.wrap_positions(positions=[cm], cell=mol.cell, pbc=mol.pbc)
    mol.positions += (wrap_cm-cm)

#wraps all molecules over a list of configurations
def wrap_molecs(db, fct=1.0):
    for at in db:
        molID = at.arrays['molID']
        for m in np.unique(molID):
            mol = at[molID==m] #copy by value
            wrap_molec(mol, fct)
            #at[molID==m].positions = mol.positions #does not work at[molID==m] is not a ref
            at.positions[molID==m,:] = mol.positions

#splits condensed phase into separate molecules
def split_molecs(db):
    smdb = []
    for at in db:
        molID = at.arrays['molID']
        for m in np.unique(molID):
            smdb += [at[molID==m]] #copy by value
    return smdb

#collects intra- and inter- molecular contributions
def collect_molec_results(db, smdb, fext):
    for at in db:
        sel = ea.sel_by_uid(smdb, at.info['uid']) #assumes molecules are in the original condensed phase order
        print(np.sum(np.abs(at.positions - np.concatenate(ea.get_prop(sel,'arrays','positions'))))) #check if that was true
        at.info['energy'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'energy'+fext))
        at.info['virial'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'virial'+fext))
        at.arrays['forces'+fext+'_intram'] = np.concatenate(ea.get_prop(sel, 'arrays', 'forces'+fext))
        at.info['energy'+fext+'_interm'] = at.info['energy'+fext]-at.info['energy'+fext+'_intram']
        at.info['virial'+fext+'_interm'] = at.info['virial'+fext]-at.info['virial'+fext+'_intram']
        at.arrays['forces'+fext+'_interm'] = at.arrays['forces'+fext]-at.arrays['forces'+fext+'_intram']
