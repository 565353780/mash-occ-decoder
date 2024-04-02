import time
import math
import torch
import trimesh
import numpy as np
import torch.optim as optim

from torch import autograd
from tqdm import trange, tqdm

from mash_occ_decoder.Lib import libmcubes
from mash_occ_decoder.Lib.libmise import MISE
from mash_occ_decoder.Lib.common import make_3d_grid
from mash_occ_decoder.Lib.libsimplify import simplify_mesh

from mash_occ_decoder.Config.config import MASH_DECODER_CONFIG


class Generator3D(object):
    """Generator class for Occupancy Networks.
    It provides functions to generate the final mesh as well refining options.
    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    """

    def __init__(
        self,
        model,
        points_batch_size=100000,
        refinement_step=0,
        with_normals=False,
        padding=0.1,
        sample=False,
        device=MASH_DECODER_CONFIG.device,
        threshold=MASH_DECODER_CONFIG.mc_threshold,
        resolution0=MASH_DECODER_CONFIG.mc_res0,
        upsampling_steps=MASH_DECODER_CONFIG.mc_up_steps,
        chunk_size=MASH_DECODER_CONFIG.mc_chunk_size,
        input_type=None,
        vol_info=None,
        vol_bound=None,
        simplify_nfaces=None,
    ):
        # self.model = model.to(device)
        self.model = model
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        self.chunk_size = chunk_size
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def eval_points(self, mash_params_file_path: str, data: dict):
        qry_pts = data["qry"]

        n_qry = qry_pts.shape[1]

        mash_params = np.load(mash_params_file_path, allow_pickle=True).item()

        mask_params = mash_params["mask_params"]
        sh_params = mash_params["sh_params"]
        rotate_vectors = mash_params["rotate_vectors"]
        positions = mash_params["positions"]

        params = np.hstack([mask_params, sh_params, rotate_vectors, positions])

        q_ftrs = torch.from_numpy(params).unsqueeze(0).to(self.device).to(torch.float32)

        chunk_size = self.chunk_size
        if self.device == "cpu":
            chunk_size = int(chunk_size * 0.1)
        n_chunk = math.ceil(n_qry / chunk_size)

        ret = []

        for_data = range(n_chunk)
        if self.device == "cpu":
            print("[INFO][Generator3D::eval_points]")
            print("\t start detect occ...")
            for_data = tqdm(for_data)
        for idx in for_data:
            data_chunk = {}
            for key in data:
                if key == "ftrs":
                    continue
                if key == "qry":
                    if idx < n_chunk - 1:
                        data_chunk[key] = data[key][
                            :, chunk_size * idx : chunk_size * (idx + 1), ...
                        ]
                        data_chunk["mash_params"] = q_ftrs[
                            :, chunk_size * idx : chunk_size * (idx + 1), ...
                        ]
                    else:
                        data_chunk[key] = data[key][:, chunk_size * idx : n_qry, ...]
                        data_chunk["mash_params"] = q_ftrs[
                            :, chunk_size * idx : n_qry, ...
                        ]
                else:
                    data_chunk[key] = data[key]

            occ = self.model(data_chunk)
            ret.append(occ)

        ret = torch.cat(ret, -1)
        ret = ret.squeeze(0)
        return ret

    def generate_mesh(self, mash_params_file_path: str, return_stats=True):
        """Generates the output mesh.
        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        """
        # self.model.eval()
        stats_dict = {}

        mesh = self.generate_from_latent(mash_params_file_path, stats_dict=stats_dict)

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh

    def generate_from_latent(self, mash_params_file_path: str, c=None, stats_dict={}):
        """Generates mesh from latent.
            Works for shapes normalized to a unit cube
        Args:
            data (tensor): data tensor
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        """
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        data = {}

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid((-0.5,) * 3, (0.5,) * 3, (nx,) * 3)
            data["qry"] = pointsf.unsqueeze(0).to(self.device)
            values = self.eval_points(mash_params_file_path, data).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            mesh_extractor = MISE(self.resolution0, self.upsampling_steps, threshold)

            points = mesh_extractor.query()
            while points.shape[0] != 0:
                # Query points
                pointsf = points / mesh_extractor.resolution
                # Normalize to bounding box
                pointsf = box_size * (pointsf - 0.5)
                pointsf = torch.FloatTensor(pointsf).to(self.device)
                data["qry"] = pointsf.unsqueeze(0).to(self.device)
                # Evaluate model and update
                values = self.eval_points(mash_params_file_path, data).cpu().numpy()
                values = values.astype(np.float64)
                mesh_extractor.update(points, values)
                points = mesh_extractor.query()

            value_grid = mesh_extractor.to_dense()

        # Extract mesh
        stats_dict["time (eval points)"] = time.time() - t0

        mesh = self.extract_mesh(value_grid, c, stats_dict=stats_dict)
        return mesh

    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        """Extracts the mesh from the predicted occupancy grid.
        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        """
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1.0 - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(occ_hat, 1, "constant", constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(occ_hat_padded, threshold)
        stats_dict["time (marching cubes)"] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1

        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound["query_vol"][:, 0].min(axis=0)
            bb_max = self.vol_bound["query_vol"][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (
                self.vol_bound["axis_n_crop"].max()
                * self.resolution0
                * 2**self.upsampling_steps
            )
            vertices = vertices * mc_unit + bb_min
        else:
            # Normalize to bounding box
            vertices /= np.array([n_x - 1, n_y - 1, n_z - 1])
            vertices = box_size * (vertices - 0.5)

        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict["time (normals)"] = time.time() - t0

        else:
            normals = None

        # Create mesh
        mesh = trimesh.Trimesh(
            vertices, triangles, vertex_normals=normals, process=False
        )

        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.0)
            stats_dict["time (simplify)"] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict["time (refine)"] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c):
        """Estimates the normals by computing the gradient of the objective.
        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        """
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(self.device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        """Refines the predicted mesh.
        Args:
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        """

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert n_x == n_y == n_z
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for _ in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True
            )[0]

            normal_target = normal_target / (
                normal_target.norm(dim=1, keepdim=True) + 1e-10
            )
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
