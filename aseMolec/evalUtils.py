import ase.io
from ase import Atoms
import os
import numpy as np

def eval_gap_quippy(db, gap, prog=True):
    for i, at in enumerate(db):
        if prog:
            print('Evaluating config:', i)
        at.calc = gap
        at.info['energy'] = at.get_potential_energy()
        at.info['virial'] = -at.get_stress(voigt=False)*at.get_volume()
        at.arrays['forces'] = at.get_forces()
        del at.calc

def eval_gap_quip(inxyz_file, outxyz_file, gap_file, init_args=None):
    quipcmd = "atoms_filename="+inxyz_file
    temp_file = os.path.splitext(outxyz_file)[0]+'_temp.xyz'
    if init_args is not None:
        quipcmd += " init_args=\""+init_args+"\""
    quipcmd += " param_filename="+gap_file+" E=True F=True V=True"
    os.system("quip "+quipcmd+" | grep AT | sed 's/AT//' > "+temp_file)
    db = ase.io.read(temp_file,':')
    for at in db:
        at.calc.reset()
        at.arrays['forces'] = at.arrays.pop('force')
        del at.arrays['map_shift']
        del at.arrays['n_neighb']
        at.info['stress'] = -at.info['virial']/at.get_volume()
        del at.info['nneightol']
        del at.info['cutoff']
    ase.io.write(outxyz_file, db)
    os.system('rm -rfv '+temp_file)

try:
    from mace.calculators import MACECalculator
except ImportError:
    pass
else:
    def eval_mace(inxyz_file, outxyz_file, mace_file, init_args=None):
        db = ase.io.read(inxyz_file, ':')
        atomic_numbers = list(np.unique([n for at in db for n in at.numbers]))
        calc = MACECalculator(model_path=mace_file, r_max=6.0, device='cpu', atomic_numbers=atomic_numbers, default_dtype="float64")
        for at in db:
            at.calc = calc
            at.info['energy'] = at.get_potential_energy()
            at.arrays['forces'] = at.get_forces()
            del at.calc
        ase.io.write(outxyz_file, db)

