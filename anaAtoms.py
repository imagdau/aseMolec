from ase import neighborlist
from ase import Atoms
import ase.geometry
from scipy import sparse
import numpy as np
import extAtoms as ea
import scipy.spatial

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
def wrap_molec(mol, fct=1.0, full=False):
    if not full:
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
def wrap_molecs(db, fct=1.0, full=False, prog=False):
    iter = 0
    for at in db:
        if prog:
            iter += 1
            print(iter)
        if 'molID' not in at.arrays.keys():
            find_molecs([at], fct)
        molID = at.arrays['molID']
        for m in np.unique(molID):
            mol = at[molID==m] #copy by value
            wrap_molec(mol, fct, full)
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

#find voids: copied from https://github.com/gabor1/workflow/blob/main/wfl/utils/find_voids.py
def find_voids(at):
    transl_symprec = 1.0e-1
    # save original cell
    cell_orig = at.get_cell()
    reciprocal_cell_orig = at.get_reciprocal_cell()
    # create supercell
    at_sc = at * [3, 3, 3]
    at_sc.set_positions(at_sc.get_positions() - np.sum(cell_orig, axis=0))
    # calculate Voronoi tesselation
    vor = scipy.spatial.Voronoi(at_sc.get_positions())
    # list possible centers from Voronoi vertices that are close to original cell
    possible_centers_lat = np.matmul(vor.vertices, reciprocal_cell_orig.T)
    possible_indices = np.where(np.all(np.abs(possible_centers_lat - 0.5) <= 0.6, axis=1))[0]
    # create atoms object with supercell of all possible interstitial positions
    vertices = vor.vertices[possible_indices]
    at_w_interst = at.copy()
    at_w_interst.extend(Atoms('X{}'.format(len(possible_indices)), positions=vertices))
    # eliminate duplicates that are equivalent by translation
    dists = at_w_interst.get_all_distances(mic=True)
    del_list = set()
    for i in range(len(at_w_interst) - 1):
        dups = i + 1 + np.where(dists[i][i + 1:] < transl_symprec)[0]
        del_list = del_list.union(set(dups))
    del at_w_interst[list(del_list)]
    return at_w_interst

def find_voids_grid(db, dx=1.0, xmin=1.0, prog=False):
    db_grid = []
    iter = 0
    for at in db:
        if prog:
            iter += 1
            print(iter)
        mat = at.cell
        N = [int(x) for x in (np.diag(at.cell)/dx)]
        Na = len(at)
        x = np.arange(0, 1, 1/N[0])
        y = np.arange(0, 1, 1/N[1])
        z = np.arange(0, 1, 1/N[2])
        X, Y, Z = np.meshgrid(x, y, z)
        grid = np.dot(np.array([X,Y,Z]).reshape(3,-1).T, at.cell)
        at_wgrid = at.copy()
        at_wgrid.extend(Atoms('X{}'.format(len(grid)), positions=grid))
        dst = at_wgrid.get_all_distances(mic=True)
        dst = dst[Na:,:Na]
        ids = np.where(np.any(dst<=xmin, axis=1))[0]
        ids += Na
        del at_wgrid[list(ids)]
        db_grid.append(at_wgrid)
    return db_grid
