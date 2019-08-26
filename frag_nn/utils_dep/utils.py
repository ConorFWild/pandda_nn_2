import numpy as np

from frag_nn import constants as c

from biopandas.pdb import PandasPdb as ppdb

import clipper_python as clipper

from transforms3d.euler import euler2mat



def load_model(pdb_path):
    pdb_model = ppdb().read_pdb(pdb_path)
    return pdb_model


def get_ligand_model(pdb_model):
    hetatms = pdb_model.df["HETATM"]
    ligand_model = hetatms[hetatms["residue_name"] == "LIG"]

    return ligand_model


def get_ligand_centroid(ligand_model):
    xyz = ligand_model[["x_coord", "y_coord", "z_coord"]].to_numpy()
    centroid = np.mean(xyz, axis=0)

    return centroid

def load_xmap(mtz_path):
    hkl_info, hkl_data = load_mtz(mtz_path)
    xmap, grid = transform_fft(hkl_info, hkl_data)

    return xmap, grid


def cart_to_grid(centroid, xmap):
    centroid_orth = clipper.Coord_orth(centroid)
    centroid_map = xmap.coord_map(centroid_orth)

    return centroid_map


def load_xmap_from_ccp4(event_map_path):
    xmap = clipper.Xmap_float()
    ccp4_file = clipper.CCP4MAPfile()
    ccp4_file.open_read(event_map_path)
    ccp4_file.import_xmap_float(xmap)
    ccp4_file.close_read()

    return xmap


def subsample_xmap(xmap=None, ligand_centroid_orth=np.array([0, 0, 0]), grid_size=10, grid_step=0.5):
    # Get grid dimensions
    grid_dimensions = [grid_size, grid_size, grid_size]

    # Generate a random grid rotation
    grid_rotation = np.random.rand(3)*2*np.pi

    # Convert grid rotation to euler angle, and that to a rotation matrix
    rot_mat_np = euler2mat(grid_rotation[0], grid_rotation[1], grid_rotation[2],
                           "sxyz")

    # Cast to clipper matrix
    rot_mat = clipper.Mat33_double(rot_mat_np)

    # Generate a scale matrix to get the right grid size
    scale_mat = clipper.Mat33_double(grid_step, 0, 0,
                                     0, grid_step, 0,
                                     0, 0, grid_step)

    # Get grid origin from ligand centroid and rotation
    # algorithm: rotate grid_dimensions*grid_step vector with random rotation, step to ligand origin,
    # then step back halfway along it
    vec_orth = np.array(grid_dimensions)*grid_step
    grid_diagonal_reshaped_orth = vec_orth.reshape(3,1)
    rotated_grid_diagonal_vector_orth = np.matmul(rot_mat_np, grid_diagonal_reshaped_orth).reshape(-1)
    ligand_centroid_orth_reshaped = ligand_centroid_orth.reshape(-1)
    origin_orth = ligand_centroid_orth_reshaped - (rotated_grid_diagonal_vector_orth/2)

    # Generate the Translation vector as a clipper vector
    trans = clipper.Vec3_double(origin_orth[0],
                                origin_orth[1],
                                origin_orth[2])

    # Generate the clipper rotation-translation operator
    rtop = clipper.RTop_double(rot_mat * scale_mat,
                               trans)

    # Generate the clipper grid
    grid = clipper.Grid(grid_dimensions[0],
                        grid_dimensions[1],
                        grid_dimensions[2])

    # Define nxmap from the clipper grid and rotation-translation operator
    nxmap = clipper.NXmap_float(grid, rtop)

    # Interpolate the Xmap onto the clipper nxmap
    clipper.interpolate(nxmap, xmap)

    # Convert the nxmap to a numpy array
    nxmap_np = nxmap.export_numpy()

    return nxmap_np


def load_mmol(pdb_path):
    f = clipper.MMDBfile()
    f.read_file(pdb_path)
    mmol = clipper.MiniMol()
    f.import_minimol(mmol)
    f.close_read()

    return mmol


def load_data(record):
    mtz = record["mtz"].numpy()
    hkl_info, hkl_data = load_mtz(mtz)
    xmap = transform_fft(hkl_info, hkl_data)

    return xmap


def load_mtz(mtz_path):
    mtz = clipper.CCP4MTZfile()
    mtz.open_read(mtz_path)
    hkl_info = clipper.HKL_info()
    hkl_data = clipper.data32.HKL_data_F_phi_float(hkl_info)
    mtz.import_hkl_info(hkl_info)
    mtz.import_hkl_data(hkl_data, "*/*/[FWT,PHWT]")
    mtz.close_read()

    return hkl_info, hkl_data


def transform_fft(hkl_info, hkl_data):

    grid = clipper.Grid_sampling(hkl_info.spacegroup,
                                 hkl_info.cell,
                                 hkl_info.resolution)

    xmap = clipper.Xmap_float(hkl_info.spacegroup,
                        hkl_info.cell,
                        grid)
    xmap.fft_from(hkl_data)

    return xmap, grid


def mask_protien(model, xmap):
    protien_atom_grid_coords = None
    protien_layer = np.zeros(xmap.shape)
    protien_layer[protien_atom_grid_coords] = 1


def mask_lig():
    pass


def get_label_is_hit(record):
    if record["ligand_confidence_inspect"] in c.hit_classes:
        label = 1.0
    else:
        label = 0.0

    return label

