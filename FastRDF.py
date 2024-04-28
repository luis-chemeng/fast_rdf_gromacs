#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Useful links:
> https://freud.readthedocs.io/en/latest/gettingstarted/examples/examples/GROMACS-MDTRAJ-WATER-RDF/Compute_RDF.html
> https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_rdf.html?highlight=compute_rdf#mdtraj.compute_rdf
> https://en.wikibooks.org/wiki/Molecular_Simulation/Radial_Distribution_Functions
@author: luis-chemeng
"""

import mdtraj
import numpy as np
import pandas as pd
from scipy import integrate
from scipy import constants as cons


def parse_gro(gro_file, verbose=True):
    """
    This function returns a dictionary of residues.
    The keys are the residue names, and the values are matrices (pd.DataFrames) containing the atom indexes.
    Each column corresponds to an atom type in the residue, and each row corresponds to a residue.

    Parameters
    ----------
    gro_file : string
        Structure file from gromacs (*.gro).
    verbose : bool, optional
        If you would like a resume of the residues and atom type found. The default is True.

    Returns
    -------
    Dictionary of residues

    """
    
    resume = ''
    
    with open(gro_file, 'r') as file:
        # Read the lines
        lines = file.readlines()
    
    natoms = int(lines[1])
    
    resume += 'The system contains {} atoms.\n'.format(natoms)
    
    data = []
    for line in lines[2:-1]:
        resnum = line[0:5].strip()
        resname = line[5:10].strip()
        atname = line[10:15].strip()
        atnum = line[15:20].strip()
        data.append([resnum, resname, atname, atnum])
        
    data = pd.DataFrame(data, columns=['resnum', 'resname', 'atname', 'atnum'])
    
    data['resnum'] = data['resnum'].astype(int)
    data['atnum'] = data['atnum'].astype(int)
    
    resnames_type = data['resname'].unique()
    
    resume += 'Residues found:\n'
    
    d_residues = {}
    for resname_type in resnames_type:
        
        resume += '\t- {} : ['.format(resname_type)
        
        resnum_i = data[data['resname']==resname_type].iloc[0]['resnum']
        resnum_f = data[data['resname']==resname_type].iloc[-1]['resnum']
        first_res = data[data['resnum']==resnum_i]
        atnum_i = first_res['atnum'].iloc[0]
        natoms_res = first_res.shape[0]
        res_atoms = first_res['atname'].to_list()
        
        for at in res_atoms:
            resume += at + ', '
        resume = resume[:-2] + ']\n'
        
        vec = np.array(data[data['resname']==resname_type]['atnum'])
        mat = vec.reshape(-1, natoms_res)
        dat = pd.DataFrame(mat, columns=res_atoms)
        # d_residues.update({resname_type : [resnum_i, resnum_f, atnum_i, natoms_res, res_atoms, vec, mat]})
        d_residues.update({resname_type : dat})
     
    if verbose:
        print(resume)
    
    return d_residues

def gen_pairs(code_at1, code_at2, d_residues):
    """
    This fucntion generate the pair of atom indexes required to compute the RDF.

    Parameters
    ----------
    code_at1 : string
        It is a code composed of the residue type and the atom type, separeted by a hyphen ('-'). Example: RES-AT1
    code_at2 : string
        Similar to the previous one, but for a second atom.
    d_residues : dictionary
        Dictionary of residues obtained from the parse_gro fucntion.

    Returns
    -------
    result : 2d numpy array
        All possible pairs of indexes from the two encoded atom type. 

    """
    
    resname1, atname1 = tuple(code_at1.split('-'))
    resname2, atname2 = tuple(code_at2.split('-'))
    
    d_residues[resname1]
    
    idxs1 = d_residues[resname1][atname1].values.reshape(-1, 1)
    idxs2 = d_residues[resname2][atname2].values.reshape(-1, 1)
    
    result = {}
    
    if resname1 == resname2:
        if atname1 == atname2:
            
            mat1 = np.tile(idxs1, (1, idxs1.shape[0]))
            mat2 = mat1.T
            mat3 = np.dstack((mat1, mat2))
            mat3 = np.transpose(mat3, (2, 0, 1))
            
            indices = np.tril_indices(mat3.shape[1], k=-1)
            pairs = mat3[:, indices[0], indices[1]].T          
            result.update({'inter' : pairs-1})
            
        else:
            
            mat1 = np.tile(idxs1, (1, idxs1.shape[0]))
            mat2 = np.tile(idxs2, (1, idxs2.shape[0])).T
            mat3 = np.dstack((mat1, mat2))
            mat3 = np.transpose(mat3, (2, 0, 1))
            
            indicesl = np.tril_indices(mat3.shape[1], k=-1)
            indicesu = np.triu_indices(mat3.shape[1], k=1)
            pairsl = mat3[:, indicesl[0], indicesl[1]].T
            pairsu = mat3[:, indicesu[0], indicesu[1]].T
            pairs = np.concatenate((pairsl, pairsu), axis=0)
            result.update({'inter' : pairs-1})
            
            pairs = np.hstack((mat1.reshape(-1, 1), mat2.reshape(-1, 1)))
            result.update({'both' : pairs-1})
            
            pairs = np.hstack((np.diag(mat1).reshape(-1, 1), np.diag(mat2.T).reshape(-1, 1)))
            result.update({'intra' : pairs-1})
    else:
        mat1 = np.tile(idxs1, (1, idxs2.shape[0]))
        mat2 = np.tile(idxs2, (1, idxs1.shape[0])).T
        pairs = np.hstack((mat1.reshape(-1, 1), mat2.reshape(-1, 1)))
        result.update({'inter' : pairs-1})
    
    return result
    

def compute_rdf(coordinate_file, topology_file, pairs_atoms, r_min=0.01, r_max=1.0, bins=300, i_frac=0.2, f_frac=0.8, save_csv=False, modifier=""):
    """
    
    Parameters
    ----------
    coordinate_file : string
        Archivo de coordenadas (*.trr, *.xtc).
    topology_file : string
        Archivo de topólgía (*.gro).
    pairs_atoms : string
        Parejas de atomos para calcular la rdf.
    r_min : float, optional
        Distancia minima para rdf. The default is 0.01. 
    r_max : float, optional
        Distancia maxima para rdf. The default is 1.0.
    bins : string, optional
        Número de puntos de la rdf.The default is 300.
    save_csv : bool, optional
        Guardar como un archivo csv. The default is False.
    modifier : string, optional
        Cadena para modificar el nombre del archivo. The default is "".
        
    Returns
    -------
    rdf_tuple : ((numpy array, numpy array))
        Tupla con los valores de r y RDF en vectores de Numpy.

    """
    # Diferenciar entre si es un archivo *.xtc o *.trr
    file_type = coordinate_file[-3:]
    # El archivo de coordenadas se carga al formato de mdtraj
    if file_type == "xtc":
        traj = mdtraj.load_xtc(coordinate_file, top=topology_file)
    elif file_type == "trr":
        traj = mdtraj.load_trr(coordinate_file, top=topology_file)
    else:
        print("El archivo de coordenadas suministrado no es ni *.xtc ni *.trr")
    
        
    # Número total de frames
    n_frames = traj.n_frames
    
    # Frame inicial y Frame final
    i_frame = round(i_frac*n_frames)
    f_frame = round(f_frac*n_frames)
    
    traj = traj.slice(range(i_frame, f_frame))
    
    rdf_tuple = mdtraj.compute_rdf(traj, pairs_atoms, (r_min, r_max), n_bins=bins)
    
    x = rdf_tuple[0].reshape(-1, 1)
    y = rdf_tuple[1].reshape(-1, 1)
    
    rdf_array = np.concatenate((x, y), 1)
    
    if save_csv:
        output_name = "rdf_" + modifier + ".csv"
        np.savetxt(output_name, rdf_array, delimiter=',')
    
    return rdf_array


def compute_ncor(rdf_array, MW, rho_bulk):
    rho_n = 1000*(rho_bulk*cons.N_A/MW)*(10**-9)**3
    ncor = 4*cons.pi*rho_n*integrate.cumtrapz(rdf_array[:,1]*rdf_array[:,0]*rdf_array[:,0], rdf_array[:,0], initial=0)
    return ncor

