from ase import neighborlist
from ase import Atoms
import ase.geometry
from scipy import sparse
import numpy as np
from aseMolec import extAtoms as ea
import scipy.spatial
from ase.ga.utilities import get_rdf
import ase.data
import warnings
from collections import Counter
import hashlib
chem_syms = ase.data.chemical_symbols

#extends fct to dictionary if needed
def modif_natural_cutoffs(at, fct):
    if type(fct) is int or type(fct) is float:
        return neighborlist.natural_cutoffs(at, mult=fct)
    elif type(fct) is dict:
        cutOff = neighborlist.natural_cutoffs(at, mult=1)
        newCutOff = []
        for ctf, el in zip(cutOff, at.get_chemical_symbols()):
            if el in fct:
                newCutOff += [ctf*fct[el]]
            else:
                newCutOff += [ctf]
        return newCutOff
    else:
        raise NameError('Unknown fct type '+str(type(fct)))

#returns molecular name based on formula
def mol_chem_name(formula):
    if formula=='C3H4O3':
        return 'EC'
    elif formula=='C4H6O3':
        return 'PC'
    elif formula=='C3H2O3':
        return 'VC'
    elif formula=='C4H8O3':
        return 'EMC'
    elif formula=='C3H6O3':
        return 'DMC'
    elif formula=='C5H10O3':
        return 'DEC'
    elif formula=='Li':
        return 'Li'
    elif formula=='F6P':
        return 'PF6'
    elif len(formula)==1:
        return formula
    else:
        return 'UNK_'+formula

def mol_config(molSym):
    d = dict(Counter(molSym))
    name = ''
    name_UNK = ''
    for k in sorted(list(d.keys())):
        if 'UNK_' in k:
            name_UNK += (k + '(%d)' % d[k] + ':')
        else:
            name += (k + '(%d)' % d[k] + ':')
    return (name[:-1]+'---'*(not len(name_UNK)==0)+name_UNK[:-1])

#computes molID for single config, not adding molID to atoms.arrays
def find_molec(at, fct=1.0):
    #from https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
    cutOff = modif_natural_cutoffs(at, fct)
    nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
    nbLst.update(at)
    conMat = nbLst.get_connectivity_matrix(sparse=True)
    Nmol, molID = sparse.csgraph.connected_components(conMat)
    Natoms, Nmols = np.unique(np.unique(molID, return_counts=True)[1], return_counts=True)
    return list(zip(Nmols,Natoms))

#computes molIDs
def find_molecs(db, fct=1.0, return_mask=False):
    masks = []
    for at in db:
        #from https://wiki.fysik.dtu.dk/ase/ase/neighborlist.html
        cutOff = modif_natural_cutoffs(at, fct)
        nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=True)
        nbLst.update(at)
        conMat = nbLst.get_connectivity_matrix(sparse=True)
        Nmol, molID = sparse.csgraph.connected_components(conMat)
        at.arrays['molID'] = molID
        if return_mask:
            mask = np.zeros([len(molID)]*2)
            for mID in np.unique(molID):
                mask += (((molID==mID).reshape(1,-1))*((molID==mID).reshape(-1,1))).astype(int)
            masks += [mask]
    if return_mask:
        return masks

#computes number of neighbours
def find_num_nb(db, Rcut=6.0):
    NumNbs = []
    for at in db:
        nbLst = neighborlist.NeighborList([Rcut/2.0]*len(at), self_interaction=False, bothways=True)
        nbLst.update(at)
        conMat = nbLst.get_connectivity_matrix(sparse=False)
        NumNbs += list(np.sum(conMat, axis=0))
    return np.array(NumNbs)

#extracts molecules CM into a new trajectory without changing any coordinates
#designed mainly for extracting diffusion coefficients, assumes no wrapping
#assumes molIDs exist and molecules are full !!!IMPORTANT, calculation of CM, intertia, torque ... !!!
#this is a bit redundant with wrap_molecs, maybe could be combined in the future
def extract_molecs(db, fct=1, intra_inter=False, prog=False):
    moldb = []
    for i, at in enumerate(db):
        if prog:
            print(i)
        if 'molID' not in at.arrays.keys():
            find_molecs([at], fct=fct)
        molID = at.arrays['molID']
        molCM = []
        molM = []
        molI = []
        molSym = []
        molQ = []
        molD = []
        molF = []
        molT = []
        atftrn = []
        atfrot = []
        for m in np.unique(molID):
            mol = at[molID==m] #copy by value
            mass = mol.get_masses()
            cm = np.sum(mol.positions*mass.reshape(-1,1), axis=0)/np.sum(mass)
            M = np.sum(mass)
            I = mom_inertia(mol)
            molCM.append(cm)
            molM.append(M)
            molI.append(I.flatten())
            molSym.append(mol_chem_name(mol.symbols.get_chemical_formula()))
            if 'initial_charges' in at.arrays:
                charge = mol.arrays['initial_charges']
                D = np.sum((mol.positions-cm)*charge.reshape(-1,1), axis=0) #subtract cm, so dipole is also correct for charges molecules, e.g. PF6
                molQ.append(np.sum(charge))
                molD.append(D)
            if 'forces' in at.arrays:
                Fcm = np.sum(mol.arrays['forces'], axis=0)
                Tcm = np.sum(np.cross(mol.positions-cm, mol.arrays['forces'], axis=1), axis=0)
                ftrn = mass.reshape(-1,1)/M*Fcm #redistributed to atoms
                if np.allclose(I, 0, atol=1e-6): #this is the case for molecules made of single atoms: Li-ion
                    frot = np.zeros([1,3])
                #WARNING: not implemented yet: handles linear molecules (singular moment of inertia) by setting frot to zero
                elif np.allclose(np.linalg.det(I), 0, atol=1e-6):
                    frot = np.zeros([len(mol),3])
                else:
                    frot = mass.reshape(-1,1)*np.cross(np.linalg.solve(I, Tcm),mol.positions-cm) #reditributed to atoms
                molF.append(Fcm)
                molT.append(Tcm)
                atftrn.append(ftrn)
                atfrot.append(frot)
        newmol = Atoms(positions=np.array(molCM), pbc=True, cell=at.cell)
        newmol.set_masses(molM)
        newmol.arrays['molSym'] = np.array(molSym)
        newmol.arrays['momInertia'] = np.array(molI)
        if molQ:
            newmol.arrays['initial_charges'] = np.array(molQ)
            newmol.arrays['dipoles'] = np.array(molD)
            newmol.arrays['dipoles_abs'] = np.sqrt(np.sum(np.array(molD)**2, axis=1))/0.2081943 #from e*A to Debye
        if molF:
            newmol.arrays['forces'] = np.array(molF)
            newmol.arrays['torques'] = np.array(molT)
        moldb.append(newmol)
        if intra_inter:
            at.arrays['forces_trans'] = np.concatenate(atftrn)
            at.arrays['forces_rot'] = np.concatenate(atfrot)
            at.arrays['forces_vib'] = at.arrays['forces']-at.arrays['forces_trans']-at.arrays['forces_rot']
    return moldb

def extract_molecs_tag(db, fct=1, intra_inter=False, prog=False, tag=''):
    moldb = []
    for i, at in enumerate(db):
        if prog:
            print(i)
        if 'molID' not in at.arrays.keys():
            find_molecs([at], fct=fct)
        molID = at.arrays['molID']
        molCM = []
        molM = []
        molI = []
        molSym = []
        molQ = []
        molD = []
        molF = []
        molT = []
        atftrn = []
        atfrot = []
        for m in np.unique(molID):
            mol = at[molID==m] #copy by value
            mass = mol.get_masses()
            cm = np.sum(mol.positions*mass.reshape(-1,1), axis=0)/np.sum(mass)
            M = np.sum(mass)
            I = mom_inertia(mol)
            molCM.append(cm)
            molM.append(M)
            molI.append(I.flatten())
            molSym.append(mol_chem_name(mol.symbols.get_chemical_formula()))
            if tag+'initial_charges' in at.arrays:
                charge = mol.arrays[tag+'initial_charges']
                D = np.sum((mol.positions-cm)*charge.reshape(-1,1), axis=0) #subtract cm, so dipole is also correct for charges molecules, e.g. PF6
                molQ.append(np.sum(charge))
                molD.append(D)
            if tag+'forces' in at.arrays:
                Fcm = np.sum(mol.arrays[tag+'forces'], axis=0)
                Tcm = np.sum(np.cross(mol.positions-cm, mol.arrays[tag+'forces'], axis=1), axis=0)
                ftrn = mass.reshape(-1,1)/M*Fcm #redistributed to atoms
                if np.allclose(I, 0, atol=1e-6): #this is the case for molecules made of single atoms: Li-ion
                    frot = np.zeros([1,3])
                #WARNING: not implemented yet: handles linear molecules (singular moment of inertia) by setting frot to zero
                elif np.allclose(np.linalg.det(I), 0, atol=1e-6):
                    frot = np.zeros([len(mol),3])
                else:
                    frot = mass.reshape(-1,1)*np.cross(np.linalg.solve(I, Tcm),mol.positions-cm) #reditributed to atoms
                molF.append(Fcm)
                molT.append(Tcm)
                atftrn.append(ftrn)
                atfrot.append(frot)
        newmol = Atoms(positions=np.array(molCM), pbc=True, cell=at.cell)
        newmol.set_masses(molM)
        newmol.arrays['molSym'] = np.array(molSym)
        newmol.arrays['momInertia'] = np.array(molI)
        if molQ:
            newmol.arrays[tag+'initial_charges'] = np.array(molQ)
            newmol.arrays[tag+'dipoles'] = np.array(molD)
            newmol.arrays[tag+'dipoles_abs'] = np.sqrt(np.sum(np.array(molD)**2, axis=1))/0.2081943 #from e*A to Debye
        if molF:
            newmol.arrays[tag+'forces'] = np.array(molF)
            newmol.arrays[tag+'torques'] = np.array(molT)
        moldb.append(newmol)
        if intra_inter:
            at.arrays[tag+'forces_trans'] = np.concatenate(atftrn)
            at.arrays[tag+'forces_rot'] = np.concatenate(atfrot)
            at.arrays[tag+'forces_vib'] = at.arrays[tag+'forces']-at.arrays[tag+'forces_trans']-at.arrays[tag+'forces_rot']
    return moldb

#assumes coordinates are unwrapped
def mom_inertia(mol):
    m = mol.get_masses()
    rcm = np.sum(m.reshape(-1,1)*mol.positions,axis=0)/np.sum(m)
    r = mol.positions-rcm
    I = np.zeros([3,3])
    I[0,0] = np.sum(m*(r[:,1]**2+r[:,2]**2))
    I[1,1] = np.sum(m*(r[:,0]**2+r[:,2]**2))
    I[2,2] = np.sum(m*(r[:,0]**2+r[:,1]**2))
    I[0,1] = -np.sum(m*r[:,0]*r[:,1])
    I[0,2] = -np.sum(m*r[:,0]*r[:,2])
    I[1,2] = -np.sum(m*r[:,1]*r[:,2])
    I[1,0] = I[0,1]
    I[2,0] = I[0,2]
    I[2,1] = I[1,2]
    return I

#wraps single molecule: completes molecule over pbc and sfits COM back to unit cell
def wrap_molec(mol, fct=1.0, full=False):
    if not full:
        cutOff = modif_natural_cutoffs(mol, fct)
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
    wrap_cm = ase.geometry.wrap_positions(positions=[cm], cell=mol.cell, pbc=mol.pbc)[0]
    mol.positions += (wrap_cm-cm)
    return wrap_cm

#wraps all molecules over a list of configurations
def wrap_molecs(db, fct=1.0, full=False, prog=False, returnMols=False):
    moldb = []
    iter = 0
    for at in db:
        if prog:
            iter += 1
            print(iter)
        if 'molID' not in at.arrays.keys():
            find_molecs([at], fct)
        molID = at.arrays['molID']
        molCM = []
        molSym = []
        for m in np.unique(molID):
            mol = at[molID==m] #copy by value
            cm = wrap_molec(mol, fct, full)
            #at[molID==m].positions = mol.positions #does not work at[molID==m] is not a ref
            at.positions[molID==m,:] = mol.positions
            molCM.append(cm)
            molSym.append(mol_chem_name(mol.symbols.get_chemical_formula()))
        newmol = Atoms(positions=np.array(molCM), pbc=True, cell=at.cell)
        newmol.arrays['molSym'] = np.array(molSym)
        at.info['Nmols'] = len(newmol)
        at.info['Comp'] = mol_config(newmol.arrays['molSym'])
        moldb.append(newmol)
    if returnMols:
        return moldb

#wraps all molecules over a list of configurations, given a mask
def wrap_molecs_partial(db, fct=1.0, full=False, prog=False, mask=None, wrap=True):
    iter = 0
    for at in db:
        if prog:
            iter += 1
            print(iter)
        if 'molID' not in at.arrays.keys():
            masked_at = at[mask]
            find_molecs([masked_at], fct)
            at.arrays['molID'] = np.array([-1]*len(at))
            at.arrays['molID'][mask] = masked_at.arrays['molID']
        molID = at.arrays['molID']
        at.arrays['molSym'] = np.array(['None']*len(at))
        at.arrays['molHash'] = np.array(['0000']*len(at))
        for m in np.unique(molID):
            if m>=0:
                mol = at[molID==m] #copy by value
                wrap_molec(mol, fct, full)
                if wrap: #this is counterintuitive given the name of the function, but for now it's the easiet way to achieve this
                    at.positions[molID==m,:] = mol.positions
                molSym = mol.symbols.get_chemical_formula()
                at.arrays['molSym'][molID==m] = molSym
                at.arrays['molHash'][molID==m] = hashlib.sha256(molSym.encode()).hexdigest()[:4]

#splits condensed phase into separate molecules
def split_molecs(db, scale=1.0):
    wrap_molecs(db, fct=1.0, full=False, prog=False)
    ea.hash_atoms(db)
    smdb = []
    if isinstance(scale, float):
        scale = np.ones(len(db))*scale
    for i, at in enumerate(db):
        molID = at.arrays['molID']
        for m in np.unique(molID):
            buf = at[molID==m] #copy by value
            buf.cell *= scale[i]
            smdb += [buf]
    return smdb

#splits condensed phase into dictionary of molecules by type
def split_molecs_dict(db, L=20.0):
    wrap_molecs(db, fct=1.0, full=False, prog=False)
    ea.hash_atoms(db)
    smdb = {}
    for i, at in enumerate(db):
        molID = at.arrays['molID']
        for m in np.unique(molID):
            buf = at[molID==m] #copy by value
            buf.cell = [L,L,L]
            buf.center()
            buf.info['mID'] = m
            molSym = mol_chem_name(buf.get_chemical_formula())
            if molSym in smdb:
                smdb[molSym] += [buf]
            else:
                smdb[molSym] = [buf]
    return smdb

#collects intra- and inter- molecular contributions
#checked in 02-DFTcalcs-Castep/10-EC4-EMC8-singleMolec-PBEG06/FiniteSizeCheck
#intra-virial is volume independent, while intra-stress is not
#therefore decomposition only works for the virial
def collect_molec_results(db, smdb, fext='', dryrun=False, molecEng=True):
    for at in db:
        sel = ea.sel_by_uid(smdb, at.info['uid']) #assumes molecules are in the original condensed phase order
        if dryrun:
            print(np.sum(np.abs(at.positions - np.concatenate(ea.get_prop(sel,'arrays','positions'))))) #check if that was true
        else:
            if molecEng:
                at.info['energy'+fext+'_intram_mol'] = ea.get_prop(sel, 'info', 'energy'+fext)
            at.info['energy'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'energy'+fext))
            if ('virial'+fext) in at.info.keys():
                at.info['virial'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'virial'+fext))
            at.arrays['forces'+fext+'_intram'] = np.concatenate(ea.get_prop(sel, 'arrays', 'forces'+fext)).astype(float)
            at.info['energy'+fext+'_interm'] = at.info['energy'+fext]-at.info['energy'+fext+'_intram']
            if ('virial'+fext) in at.info.keys():
                at.info['virial'+fext+'_interm'] = at.info['virial'+fext]-at.info['virial'+fext+'_intram']
            at.arrays['forces'+fext+'_interm'] = at.arrays['forces'+fext]-at.arrays['forces'+fext+'_intram']

#collects molecules without assuming any order, but expects mID in info
def collect_molec_results_dict(db, smdb, fext='', dryrun=False):
    for at in db:
        del at.calc
        uid = at.info['uid']
        sm = ea.sel_by_uid(smdb, uid)
        idx = np.argsort(ea.get_prop(sm, 'info', 'mID'))
        sel = [sm[i] for i in idx]
        #check positions differ only up to a translation
        if dryrun:
            molID = at.arrays['molID']
            mdiffs = []
            for m in np.unique(at.arrays['molID']):
                diffs = at.positions[molID==m,:]-sel[m].positions
                mdiffs += [np.max(np.abs(diffs-diffs[0,:]))]
            print(np.max(mdiffs))
        else:
            # at.info['energy'+fext+'_intram_mol'] = ea.get_prop(sel, 'info', 'energy'+fext)
            at.info['energy'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'energy'+fext))
            at.info['virial'+fext+'_intram'] = sum(ea.get_prop(sel, 'info', 'virial'+fext))
            at.arrays['forces'+fext+'_intram'] = np.concatenate(ea.get_prop(sel, 'arrays', 'forces'+fext)).astype(float)
            at.info['energy'+fext+'_interm'] = at.info['energy'+fext]-at.info['energy'+fext+'_intram']
            at.info['virial'+fext+'_interm'] = at.info['virial'+fext]-at.info['virial'+fext+'_intram']
            at.arrays['forces'+fext+'_interm'] = at.arrays['forces'+fext]-at.arrays['forces'+fext+'_intram']
            if ('initial_charges'+fext) in at.info.keys():
                at.arrays['initial_charges'+fext+'_intram'] = np.concatenate(ea.get_prop(sel, 'arrays', 'initial_charges'+fext)).astype(float)
                at.arrays['initial_charges'+fext+'_interm'] = at.arrays['initial_charges'+fext]-at.arrays['initial_charges'+fext+'_intram']

#starting from one configuration, adjusts the volume according to vol_fracs
def scan_vol(at, vol_fracs, frozen=True):
    db = []
    lat_fracs = vol_fracs**(1.0/3.0)
    mol = wrap_molecs([at], fct=1, full=False, prog=False, returnMols=True)[0]
    molID = at.arrays['molID']
    for f in lat_fracs:
        mol_disps = mol.positions*(f-1)
        nat = at.copy()
        nat.cell *= f
        id = 0
        if frozen:
            for m in np.unique(molID):
                nat.positions[molID==m,:] += mol_disps[id,:]
                id += 1
        else:
            nat.positions *= f
        db.append(nat)
    return db

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

def find_voids_grid(db, dx=2.0, xminfct=2.0, prog=False):
    db_grid = []
    iter = 0
    pts = []
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
        xmin = (3*at.get_volume()/at.get_global_number_of_atoms()/4/np.pi)**(1/3)
        ids = np.where(np.any(dst<=xmin*xminfct, axis=1))[0]
        ids += Na
        del at_wgrid[list(ids)]
        db_grid.append(at_wgrid)
        pts.append(len(grid)-len(ids))
    return db_grid, pts

def track_initial_bonds(db, fct=1, prog=False):
    cutOff = modif_natural_cutoffs(db[0], fct)
    nbLst = neighborlist.NeighborList(cutOff, self_interaction=False, bothways=False)
    nbLst.update(db[0])
    conMat = nbLst.get_connectivity_matrix(sparse=False)
    dists = db[0].get_all_distances(mic=True)[conMat==True].reshape(-1,1)
    iter = 0
    for at in db[1:]:
        if prog:
            iter += 1
            print(iter)
        dists = np.hstack([dists, at.get_all_distances(mic=True)[conMat==True].reshape(-1,1)])
    return dists

def track_distrib_grid(db, N=2, prog=False):
    masses = []
    numbers = []
    densities = []
    iter = 0
    for at in db:
        if prog:
            iter += 1
            print(iter)
        frac = at.get_scaled_positions(wrap=True)
        m = at.get_masses()
        idx = np.sum(np.floor(frac*N)*np.array([N**2,N**1,N**0]),axis=1).astype(int)
        masses.append([np.sum(m[idx==i]) for i in range(N**3)])
        numbers.append([sum(idx==i) for i in range(N**3)])
        densities.append([np.sum(m[idx==i])*(N**3)*10/6.022/at.get_volume() for i in range(N**3)])
    masses = np.array(masses)
    numbers = np.array(numbers)
    return np.array(densities), masses/np.mean(masses, axis=1).reshape(-1,1), numbers/np.mean(numbers, axis=1).reshape(-1,1)

def mol_env(at, Rcut=6.0, returnEnvs=False):
    Nmol = len(at)
    molSym = at.arrays['molSym']
    molEnv = dict()
    molEnvArr = []
    lbs = list(np.unique(molSym))
    for lb in lbs:
        molEnv[lb] = []
    nbLst = neighborlist.NeighborList([Rcut/2]*len(at), self_interaction=False, bothways=True)
    nbLst.update(at)
    S = nbLst.get_connectivity_matrix(sparse=False)
    for i in range(Nmol):
        counts = np.unique(molSym[S[i,:]==1], return_counts=True)
        buf = []
        for lb in lbs:
            if lb in counts[0]:
                buf.append(counts[1][list(counts[0]).index(lb)])
            else:
                buf.append(0)
        molEnv[molSym[i]].append(np.array(buf))
        molEnvArr.append(np.array(buf))
    for lb in lbs:
        #molEnv[lb] = np.concatenate(molEnv[lb]) - will not work, creates a 1D array
        molEnv[lb] = np.array(molEnv[lb])
    if returnEnvs:
        at.arrays['molEnv'] = np.array(molEnvArr)
        at.info['molEnvLb'] = lbs
    return molEnv

def sublist(subls, totls):
    idx = []
    for l in subls:
        if l in totls:
            idx.append(totls.index(l))
        else:
            return False, None
    return True, idx

def mol_envs(moldb, lbs, Rcut=6.0, returnEnvs=False):
    menvs = dict()
    for lb in lbs:
        menvs[lb] = np.empty(shape=[0,len(lbs)]).astype(int)
    for at in moldb:
        menv = mol_env(at, Rcut, returnEnvs)
        nlbs = list(menv.keys())
        is_sublist, mask = sublist(nlbs, lbs)
        if is_sublist:
            for lb in nlbs:
                buf = np.zeros([menv[lb].shape[0], len(lbs)]).astype(int)
                buf[:,mask] = menv[lb] #if less molecules, e.g. only EMC, fill out rest with zero
                menvs[lb] = np.vstack([menvs[lb], buf])
            if returnEnvs:
                buf = np.zeros([len(at), len(lbs)]).astype(int)
                buf[:,mask] = at.arrays['molEnv']
                at.info['molEnvLb'] = lbs
                at.arrays['molEnv'] = buf
    return menvs

def compute_rdfs(at, rmax, nbins):
    rdfs = {}
    N = len(at)
    z_counts = dict([(x,y) for x,y in zip(*np.unique(at.numbers, return_counts=True))])
    dm = at.get_all_distances(mic=True)
    intra_mask = find_molecs([at], return_mask=True)[0]
    for z1 in z_counts:
        for z2 in z_counts:
            if z2<z1:
                continue
            rdf, r = get_rdf(atoms=at, rmax=rmax, nbins=nbins, distance_matrix=dm*intra_mask, elements=[z1,z2], no_dists=False)
            if z2>z1:
                rdf *= 2.0
            rdfs[chem_syms[z1]+chem_syms[z2]+'_intra'] = rdf*z_counts[z1]/N
    inter_mask = 1-intra_mask
    for z1 in z_counts:
        for z2 in z_counts:
            if z2<z1:
                continue
            rdf, r = get_rdf(atoms=at, rmax=rmax, nbins=nbins, distance_matrix=dm*inter_mask, elements=[z1,z2], no_dists=False)
            if z2>z1:
                rdf *= 2.0
            rdfs[chem_syms[z1]+chem_syms[z2]+'_inter'] = rdf*z_counts[z1]/N
    return rdfs, r

def compute_rdfs_traj_avg(traj, rmax, nbins):
    N = len(traj)
    rdfs, r = compute_rdfs(traj[0], rmax, nbins)
    for at in traj[1:]:
        tmp_rdfs, _ = compute_rdfs(at, rmax, nbins)
        for d in tmp_rdfs:
            rdfs[d] += tmp_rdfs[d]
    for d in rdfs:
        rdfs[d] /= N
    return rdfs, r

def compute_rdfs_traj_stats(traj, rmax, nbins, win=1):
    N = np.floor(len(traj)/win).astype(int)
    rdfs, r = compute_rdfs_traj_avg(traj[slice(0, win)], rmax, nbins)
    for d in rdfs:
        rdfs[d] = [rdfs[d]]
    for i in range(1,N):
        tmp_rdfs, _ = compute_rdfs_traj_avg(traj[slice(win*i, win*(i+1))], rmax, nbins)
        for d in tmp_rdfs:
            rdfs[d] += [tmp_rdfs[d]]
    for d in rdfs:
        rdfs[d] = np.array(rdfs[d])
    for d in rdfs:
        rdfs[d] = {'avg': list(np.mean(rdfs[d], axis=0)), 'std': list(np.std(rdfs[d], axis=0))}
    return {'rdfs': rdfs, 'r': list(r)}
