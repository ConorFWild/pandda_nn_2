import numpy as np

import tensorflow as tf

import pathlib as p

import frag_nn.tensorflow.transforms_tf as trans_tf
import frag_nn.constants as c


# def wrap_iterator(records, iterator_type, grid_size=10, grid_step=0.5):
#
#     if iterator_type == "default":
#
#         return de



def xmap_iterator(records, grid_size=10, grid_step=0.5):

    def iterator():
        for record in records:
            try:

                pdb_path = record[c.model_pdb_record_name]
                mtz_path = record[c.input_mtz_record_name]

                # Check that the path actually is there before tying to load
                if (p.Path(pdb_path).exists()) & (p.Path(mtz_path).exists()):

                    # Load the model
                    model = trans_tf.load_model(pdb_path)

                    # Find the ligand
                    ligand_model = trans_tf.get_ligand_model(model)

                    # Find the ligand centroid
                    ligand_centroid = trans_tf.get_ligand_centroid(ligand_model)

                    # Load the xmap
                    xmap, grid_model = trans_tf.load_xmap(mtz_path)

                    # Translate the ligand centroid to grid coords
                    ligand_centroid_grid = trans_tf.cart_to_grid(ligand_centroid,
                                                                 xmap)

                    # Cut around the ligand
                    xmap_trans = trans_tf.subsample_xmap(xmap=xmap,
                                                         ligand_centroid_grid=ligand_centroid_grid,
                                                        grid_size=grid_size,
                                                        grid_step=grid_step)

                    # # Find the sample label from the database
                    label = trans_tf.get_label_is_hit(record)

                    # Convert hit value to tensor
                    t_label = tf.cast(label, tf.float32)

                    # Convert xmap to tensor
                    t_xmap = tf.cast(xmap_trans, tf.float32)
                    t_xmap = tf.expand_dims(
                        t_xmap,
                        axis=0)
                    t_xmap = tf.expand_dims(
                        t_xmap,
                        axis=-1)

                    # Redefine for convenience
                    x = t_xmap
                    y = t_label

                    # print(y)

                    yield x, y

            # Handle failure to be able to access data
            # TODO: remove the exception printing? Don't print for performance reasons

            except Exception as e:
                # print(e)
                # TODO: should be happy to error, remove this as it is just for debug
                # exit()
                pass

    return iterator

def create_xmap_procedural_iterator(records, grid_size=10, grid_step=0.5):

    def procedural_iterator():

        for record in records:
            x, y = get_xmap_label(record, grid_size, grid_step)

            if x is None:
                continue

            yield x, y

    return procedural_iterator


def create_xmap_stochastic_iterator(records, grid_size=10, grid_step=0.5):
    def stochastic_iterator():

        while True:

            record = records[np.random.randint(0, len(records))]

            x, y = get_xmap_label(record, grid_size, grid_step)

            if x is None:
                continue

            yield x, y

    return stochastic_iterator


def create_ccp4_stochastic_iterator(records, grid_size=10, grid_step=0.5):
    def stochastic_iterator():

        while True:

            record = records[np.random.randint(0, len(records))]

            x, y = get_ccp4_label(record, grid_size, grid_step)

            if x is None:
                continue

            yield x, y

    return stochastic_iterator

def create_noise_stochastic_iterator(records, grid_size=10, grid_step=0.5):
    def stochastic_iterator():

        while True:

            record = records[np.random.randint(0, len(records))]

            x, y = get_xmap_label(record, grid_size, grid_step)

            if x is None:
                continue

            x_tmp = np.random.rand(grid_size, grid_size, grid_size)

            x = tf.cast(x_tmp, tf.float32)
            x = tf.expand_dims(
                x,
                axis=0)
            x = tf.expand_dims(
                x,
                axis=-1)

            yield x, y

    return stochastic_iterator


def get_ccp4_label(record, grid_size=10, grid_step=0.5):
    try:

        pdb_path = record[c.model_pdb_record_name]
        event_map_path = record[c.event_map_record_name]

        # Check that the path actually is there before tying to load
        if (p.Path(pdb_path).exists()) & (p.Path(event_map_path).exists()):
            # Load the model
            model = trans_tf.load_model(pdb_path)

            # Find the ligand
            ligand_model = trans_tf.get_ligand_model(model)

            # Find the ligand centroid
            ligand_centroid_orth = trans_tf.get_ligand_centroid(ligand_model)

            # Load the xmap
            # print("Loading ccp4 map")
            xmap = trans_tf.load_xmap_from_ccp4(event_map_path)

            # Translate the ligand centroid to grid coords
            # print("Transforming ligand coords")
            # ligand_centroid_grid = trans_tf.cart_to_grid(ligand_centroid,
            #                                              xmap)

            # Cut around the ligand
            xmap_trans = trans_tf.subsample_xmap(xmap=xmap,
                                                 ligand_centroid_orth=ligand_centroid_orth,
                                                 grid_size=grid_size,
                                                 grid_step=grid_step)

            # # Find the sample label from the database
            label = trans_tf.get_label_is_hit(record)

            # Convert hit value to tensor
            t_label = tf.cast(label, tf.float32)

            # Convert xmap to tensor
            t_xmap = tf.cast(xmap_trans, tf.float32)
            # t_xmap = tf.expand_dims(
            #     t_xmap,
            #     axis=0)
            t_xmap = tf.expand_dims(
                t_xmap,
                axis=-1)

            # Redefine for convenience
            x = t_xmap
            y = t_label

            # print(y)

            return x, y

        # Handle failure to be able to access data
        # TODO: remove the exception printing? Don't print for performance reasons

    except Exception as e:
        print(e)
        # TODO: should be happy to error, remove this as it is just for debug
        # exit()
        return None, None


def get_xmap_label(record, grid_size=10, grid_step=0.5):
    try:

        pdb_path = record[c.model_pdb_record_name]
        mtz_path = record[c.input_mtz_record_name]

        # Check that the path actually is there before tying to load
        if (p.Path(pdb_path).exists()) & (p.Path(mtz_path).exists()):
            # Load the model
            model = trans_tf.load_model(pdb_path)

            # Find the ligand
            ligand_model = trans_tf.get_ligand_model(model)

            # Find the ligand centroid
            ligand_centroid = trans_tf.get_ligand_centroid(ligand_model)

            # Load the xmap
            xmap, grid_model = trans_tf.load_xmap(mtz_path)

            # Translate the ligand centroid to grid coords
            ligand_centroid_grid = trans_tf.cart_to_grid(ligand_centroid,
                                                         xmap)

            # Cut around the ligand
            xmap_trans = trans_tf.subsample_xmap(xmap=xmap,
                                                 ligand_centroid_grid=ligand_centroid_grid,
                                                 grid_size=grid_size,
                                                 grid_step=grid_step)

            # # Find the sample label from the database
            label = trans_tf.get_label_is_hit(record)

            # Convert hit value to tensor
            t_label = tf.cast(label, tf.float32)

            # Convert xmap to tensor
            t_xmap = tf.cast(xmap_trans, tf.float32)
            t_xmap = tf.expand_dims(
                t_xmap,
                axis=0)
            t_xmap = tf.expand_dims(
                t_xmap,
                axis=-1)

            # Redefine for convenience
            x = t_xmap
            y = t_label

            # print(y)

            return x, y

        # Handle failure to be able to access data
        # TODO: remove the exception printing? Don't print for performance reasons

    except Exception as e:
        # print(e)
        # TODO: should be happy to error, remove this as it is just for debug
        # exit()
        return None, None