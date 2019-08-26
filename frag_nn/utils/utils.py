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


def load_xmap_from_ccp4(event_map_path):

    print(event_map_path)

    xmap = clipper.Xmap_float()
    # print(dir(clipper))
    ccp4_file = clipper.CCP4MAPfile()
    ccp4_file.open_read(event_map_path)
    ccp4_file.import_xmap_float(xmap)
    ccp4_file.close_read()

    return xmap


def subsample_xmap(xmap=None, grid_rotation=None, ligand_centroid_orth=np.array([0, 0, 0]), grid_size=10, grid_step=0.5):

    # print("Subsampling")


    # Get grid dimensions
    grid_dimensions = [grid_size, grid_size, grid_size]

    # Generate a random grid rotation
    # grid_rotation = np.random.rand(3)*2*np.pi*np.array([0, 0, 1])
    # grid_rotation = np.array([0, 0, np.pi/2 + np.pi*np.random.randint(2)])



    # Convert grid rotation to euler angle, and that to a rotation matrix
    rot_mat_np = euler2mat(grid_rotation[0],
                           grid_rotation[1],
                           grid_rotation[2],
                           "sxyz")


    # Cast to clipper matrix
    # rot_mat = clipper.Mat33_double(rot_mat_np)

    # Generate a scale matrix to get the right grid size
    scale_mat_np = np.eye(3)*(1/grid_step)
    # scale_map_np = np.eye(3)*(1/grid_step)

    # scale_mat = clipper.Mat33_double(grid_step, 0, 0,
    #                                  0, grid_step, 0,
    #                                  0, 0, grid_step)
    # scale_mat = clipper.Mat33_double(scale_mat_np)

    # Get grid origin from ligand centroid and rotation
    # algorithm: rotate grid_dimensions*grid_step vector with random rotation, step to ligand origin, then step back halfway along it
    vec_orth = np.array(grid_dimensions)*grid_step
    grid_diagonal_reshaped_orth = vec_orth.reshape(3,1)

    half_grid_diagonal_orth = grid_diagonal_reshaped_orth/2

    rotated_grid_diagonal_vector_orth = np.matmul(rot_mat_np, half_grid_diagonal_orth).reshape(-1)

    ligand_centroid_orth_reshaped = ligand_centroid_orth.reshape(-1)
    # ligand_centroid_orth_reshaped = half_grid_diagonal_orth.reshape(-1)

    origin_orth = ligand_centroid_orth_reshaped + rotated_grid_diagonal_vector_orth

    # origin_orth = -origin_orth

    # origin_orth = origin_orth/grid_step

    # rot_mat_inv_np = euler2mat(-grid_rotation[0],
    #                        -grid_rotation[1],
    #                        -grid_rotation[2],
    #                        "sxyz")

    # rot_mat_inv_np = np.linalg.inv(rot_mat_np)
    #
    # origin_orth = np.matmul(rot_mat_np, origin_orth)


    # Generate the Translation vector as a clipper vector
    # trans = clipper.Vec3_double(origin_orth[0],
    #                             origin_orth[1],
    #                             origin_orth[2])
    # trans = clipper.Vec3_double(0, 0, 0)

    # Generate the clipper rotation-translation operator
    # rtop = clipper.RTop_double(rot_mat,
    #                            trans)
    # rtop = clipper.RTop_orth(rot_mat*scale_mat,
    #                            trans)

    rotate_inv_np = np.linalg.inv(rot_mat_np)
    # rotate_scale_np = np.matmul(scale_mat_np, rot_mat_np)
    # rotate_scale_inv_np = np.linalg.inv(rotate_scale_np)
    rotate_scale_inv_np = np.matmul(scale_mat_np, rotate_inv_np)
    scale_mat_inv_np = np.linalg.inv(scale_mat_np)

    rotate_inv_scale_inv_np = np.matmul(scale_mat_inv_np, rotate_inv_np)
    # rotate_inv_scale_inv = clipper.Mat33_double(rotate_inv_scale_inv_np)


    # rotate_scale = clipper.Mat33_double(rot_mat_np)
    rotate_scale_inv = clipper.Mat33_double(rotate_scale_inv_np)

    # rotate_inv_scale_np = np.matmul(rotate_inv_np, scale_mat_np)
    # rotate_inv_scale = clipper.Mat33_double(rot_mat_np)

    trans_inv = clipper.Vec3_double(-origin_orth[0],
                                -origin_orth[1],
                                -origin_orth[2])

    rtop = clipper.RTop_orth(-rotate_scale_inv,
                             -rotate_scale_inv*trans_inv)


    # Generate the clipper grid
    grid = clipper.Grid(grid_dimensions[0],
                        grid_dimensions[1],
                        grid_dimensions[2])

    # Define nxmap from the clipper grid and rotation-translation operator
    nxmap = clipper.NXmap_float(grid,
                                rtop)


    # Interpolate the Xmap onto the clipper nxmap
    # print("interpolating!")
    clipper.interpolate(nxmap, xmap)

    # Convert the nxmap to a numpy array
    nxmap_np = nxmap.export_numpy()


    # print("ligand centre is: {}".format(ligand_centroid_orth))
    # print("Rotation is: {}".format(grid_rotation))
    # print("Rotation matrix is:")
    # print(rot_mat_np)
    # print("Grid diagonal vector is: {}".format(grid_diagonal_reshaped_orth))
    # print("Rotated grid diagonal vector is: {}".format(rotated_grid_diagonal_vector_orth))
    # print("centre of rotation: {}".format(ligand_centroid_orth_reshaped))
    # print("Translation is : {}".format(origin_orth))
    # print("New origin is: {}".format(origin_orth))
    # map_origin = nxmap.coord_orth(clipper.Coord_map(0, 0, 0))
    #
    # map_centre = nxmap.coord_orth(clipper.Coord_map(int(grid_size / 2), int(grid_size / 2), int(grid_size / 2)))
    #
    # map_edge = nxmap.coord_orth(clipper.Coord_map(grid_size - 1, grid_size - 1, grid_size - 1))
    # print("map origin is")
    # print(map_origin)
    #
    # print("map centre is ")
    # print(map_centre)
    # print("map edge is ")
    # print(map_edge)

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

def get_label_is_hit_v2(record):
    # print("Getting label")
    if record["Ligand Confidence"] in ["High", "Medium"]:
        label = 1.0
    else:
        label = 0.0

    return label