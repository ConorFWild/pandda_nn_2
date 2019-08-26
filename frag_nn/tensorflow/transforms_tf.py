import numpy as np
# import scipy
# import skimage

from frag_nn import constants as c

from biopandas.pdb import PandasPdb as ppdb

import clipper_python as clipper

from transforms3d.euler import euler2mat



def load_model(pdb_path):
    pdb_model = ppdb().read_pdb(pdb_path)
    return pdb_model

def get_ligand_model(pdb_model):
    hetatms = pdb_model.df["HETATM"]
    # print(hetatms)
    ligand_model = hetatms[hetatms["residue_name"] == "LIG"]

    return ligand_model

def get_ligand_centroid(ligand_model):

    # TODO: probably shouldn't be mean
    # print(ligand_model)

    xyz = ligand_model[["x_coord", "y_coord", "z_coord"]].to_numpy()
    centroid = np.mean(xyz, axis=0)

    return centroid

def load_xmap(mtz_path):
    hkl_info, hkl_data = load_mtz(mtz_path)
    xmap, grid = transform_fft(hkl_info, hkl_data)

    return xmap, grid

def cart_to_grid(centroid, xmap):

    # print("Cart to grid")
    # print("Cetnroid is: {}".format(centroid))

    centroid_orth = clipper.Coord_orth(centroid)
    # print("Centroid orth is: {}".format(centroid_orth))

    centroid_map = xmap.coord_map(centroid_orth)
    # print("Centroid map is: {}".format(centroid_map))


    return centroid_map

# def subsample_map(xmap, ligand_centroid_grid, cut_size=7, rotate=False):
#
#     if rotate is True:
#         pass
#
#     else:
#         map_numpy = np.zeros((xmap.grid_asu.nu(), xmap.grid_asu.nv(), xmap.grid_asu.nw()), dtype='double')
#
#         xmap.export_numpy(map_numpy)
#
#         cut_map = map_numpy[int(ligand_centroid_grid[0])-cut_size:int(ligand_centroid_grid[0])+cut_size,
#                   int(ligand_centroid_grid[1]) - cut_size:int(ligand_centroid_grid[1]) + cut_size,
#                   int(ligand_centroid_grid[2]) - cut_size:int(ligand_centroid_grid[2]) + cut_size]
#
#     return cut_map


def load_xmap_from_ccp4(event_map_path):

    xmap = clipper.Xmap_float()
    # print(dir(clipper))
    ccp4_file = clipper.CCP4MAPfile()
    ccp4_file.open_read(event_map_path)
    ccp4_file.import_xmap_float(xmap)
    ccp4_file.close_read()

    return xmap

def subsample_xmap(xmap=None, ligand_centroid_orth=np.array([0, 0, 0]), grid_size=10, grid_step=0.5):

    # print("Subsampling")

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
    # algorithm: rotate grid_dimensions*grid_step vector with random rotation, step to ligand origin, then step back halfway along it
    vec_orth = np.array(grid_dimensions)*grid_step
    grid_diagonal_reshaped_orth = vec_orth.reshape(3,1)
    # print(vec_reshaped)
    # print(vec_reshaped.shape)
    # print(rot_mat_np)
    # print(rot_mat_np.shape)
    rotated_grid_diagonal_vector_orth = np.matmul(rot_mat_np, grid_diagonal_reshaped_orth).reshape(-1)
    # print("ligand centroid grid is: {}".format(ligand_centroid_grid))
    # print(ligand_centroid_grid[0])
    # print(float(ligand_centroid_grid[0]))
    ligand_centroid_orth_reshaped = ligand_centroid_orth.reshape(-1)
    origin_orth = ligand_centroid_orth_reshaped - (rotated_grid_diagonal_vector_orth/2)
    # print("Prinitng ligand grid")
    # print(ligand_centroid_grid_np.shape)
    # print(rotated_grid_diagonal_vector.shape)
    # print("printing origin")
    # print(origin)

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
    # print("interpolating!")
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

    # mmol = load_mmol(record["pdb"])
    hkl_info, hkl_data = load_mtz(mtz)
    xmap = transform_fft(hkl_info, hkl_data)

    return xmap

def load_mtz(mtz_path):
    # print(mtz_path)
    mtz = clipper.CCP4MTZfile()
    mtz.open_read(mtz_path)
    hkl_info = clipper.HKL_info()
    hkl_data = clipper.data32.HKL_data_F_phi_float(hkl_info)
    mtz.import_hkl_info(hkl_info)
    mtz.import_hkl_data(hkl_data, "*/*/[FWT,PHWT]")
    mtz.close_read()

    # print("read mtz")

    return hkl_info, hkl_data



#
# def transform_rotate(image):
#     # Define random angles
#
#     # Rotate
#     image = scipy.ndimage.rotate(image,
#                                  random[0],
#                                  (0, 1),
#                                  reshape=False)
#     image = scipy.ndimage.rotate(image,
#                                  random[1],
#                                  (1, 2),
#                                  reshape=False)
#     image = scipy.ndimage.rotate(image,
#                                  random[2],
#                                  (0, 2),
#                                  reshape=False)
#
#     return image
#
#
# def transform_crop(image):
#     # Define crop parameters
#
#     image = skimage.util.crop(image,
#                               (40, 40, 40))
#
#     return image


def transform_fft(hkl_info, hkl_data):

    grid = clipper.Grid_sampling(hkl_info.spacegroup,
                                 hkl_info.cell,
                                 hkl_info.resolution)

    # print("Making sampling grid")

    xmap = clipper.Xmap_float(hkl_info.spacegroup,
                        hkl_info.cell,
                        grid)
    xmap.fft_from(hkl_data)

    # print("fft'd")

    # map_numpy = np.zeros((xmap.grid_asu.nu(), xmap.grid_asu.nv(), xmap.grid_asu.nw()), dtype='double')
    #
    # xmap.export_numpy(map_numpy)

    # return map_numpy

    return xmap, grid

def mask_protien(model, xmap):
    protien_atom_grid_coords = None
    protien_layer = np.zeros(xmap.shape)
    protien_layer[protien_atom_grid_coords] = 1

def mask_lig():
    pass

def get_label_is_hit(record):
    # print("Getting label")
    if record["ligand_confidence_inspect"] in c.hit_classes:
        label = 1.0
    else:
        label = 0.0

    return label

