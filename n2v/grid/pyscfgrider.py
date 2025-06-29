"""
grider_pyscf.py
Grider for PyScf
"""

from gbasis.evals.density import (evaluate_density, 
#                                  evaluate_posdef_kinetic_energy_density,
                                  evaluate_density_laplacian,
                                  evaluate_density_gradient,
 #                                 evaluate_general_kinetic_energy_density,
                                )
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.evals.electrostatic_potential import point_charge_integral
from gbasis.wrappers import from_pyscf

# from grid.molgrid import MolGrid
# from grid.onedgrid import GaussLaguerre,  GaussLegendre, HortonLinear
# from grid.becke import BeckeWeights

import numpy as np
from opt_einsum import contract

try:
    from pyscf import dft
    has_pyscf = True
except ImportError:
    has_pyscf = False

if has_pyscf:
    class PySCFGrider:
        """
        PySCF Grider Class
        Provides methods to obtain components on the grid using the package gbasis
        """
        def __init__(self, mol, pbs_mol):

            self.mol   = mol
            
            # these two are gbasis's basis, not pyscf.
            self.basis = from_pyscf(mol)
            self.pbs   = from_pyscf(pbs_mol) if pbs_mol is not None else None


        
        def generate_rectangular_grid(self, x, y, z):
            """
            Genrates Mesh from 3 separate linear spaces and flatten,
            needed for cubic grid.
            Parameters
            ----------
            grid: tuple of three np.ndarray
                (x, y, z)
            Returns
            -------
            grid: np.ndarray
                shape (nx, ny, nz, 3).
                nx=len(x)
            shape: tuple
                (nx, ny, nz) shape of rectangular grid.
            """
            # x,y,z, = grid
            shape = (len(x), len(y), len(z))
            X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
            X = X.reshape((X.shape[0] * X.shape[1] * X.shape[2], 1))
            Y = Y.reshape((Y.shape[0] * Y.shape[1] * Y.shape[2], 1))
            Z = Z.reshape((Z.shape[0] * Z.shape[1] * Z.shape[2], 1))
            grid = np.concatenate((X,Y,Z), axis=1)
            return grid, shape
        
        def generate_spherical_grid(self, level=4):
            """
            Genrates a spherical grid with pyscf.
            Parameters
            ----------
            level: int
                gird dense level
            Returns
            -------
                grids: pyscf.dft.gen_grid.Grids
            """
            
            grids = dft.gen_grid.Grids(self.mol)
            grids.level = level
            grids = grids.build()
            return grids

        def density(self, D, grid, basis):
            """
            Computes density on grid. 

            Parameters
            ----------
            D: np.ndarray
                Density matrices.
                Following the pyscf interface.
                if ref==1:
                    D = D shape (nbf, nbf)
                elif ref==2:
                    D = [Da, Db] shape (2, nbf, nbf)
            basis:
                AO basis
            grid: np.ndarray, shape (N, 3)
                grid of length N.
                
            Returns
            -------
            n: np.ndarray
                Density on the grid.
                (N, ) for ref=1
                (2, N) for ref=2  
                  
            """
            if D.ndim == 2:
                n = self.to_grid(D, grid, basis)
                return n
            else:
                assert len(D) == 2
                na = self.to_grid(D[0], grid, basis)
                nb = self.to_grid(D[1], grid, basis)
                return np.array(na, nb)

        def hartree(self, D, grid):
            """
            Computes Hartree Potential on grid. 

            Parameters
            ----------

            D: np.ndarray
                Density matrices.
                Following the pyscf interface.
                if ref==1:
                    D = D shape (nbf, nbf)
                elif ref==2:
                    D = [Da, Db] shape (2, nbf, nbf)

            grid: np.ndarray, shape (N, 3)
                grid of length N.


            Returns
            -------

            hartree_potential: np.ndarray
                Hartree potential on the requested grid
            """        
            assert grid.shape[1] == 3
            
            hartree_potential = point_charge_integral(self.basis, 
                                                    points_coords=grid, 
                                                    points_charge=-np.ones(grid.shape[0]), 
                                                    transform=None)
            if D.ndim == 3:
                D = D[0] + D[1]
            hartree_potential *= D[:, :, None]
            hartree_potential = np.sum(hartree_potential, axis=(0, 1))
            return hartree_potential

        def external(self, grid: np.ndarray):
            """
            Computes External Potential on grid. 

            Parameters
            ----------
            grid: np.ndarray, shape (N, 3)
                grid of length N.

            Returns
            -------
            external_potential: np.ndarray
                External potential on the given grid. 
            """        
            with np.errstate(divide="ignore"):  # silence warning for dividing by zero
                external_potential = self.mol.atomic_charges()[None, :] \
                / (np.sum((grid[:, None, :] - self.mol.atom_coords().T[None, :, :]) ** 2, axis=1) ** 0.5)

            if external_potential.ndim > 1:
                external_potential = np.sum(external_potential, axis=1)

            return -external_potential

        def to_grid(self, f_nm, grid: np.ndarray, basis=None):
            """
            Expresses a matrix quantity on the grid

            Parameters
            ----------
            coeff: np.ndarray
                Vector/Matrix on ao basis. 
                Shape: {(num_ao_basis, ), (num_ao_basis, num_ao_basis)}
            grid: np.ndarray, shape (N, 3)
                grid of length N.
            basis:
                pyscf basis. If None, use sekf.basis
            Returns
            -------
            f_g: np.ndarray
                Vector/Matrix expressed on the requested grid
            """
            
            if basis is None:
                basis = self.basis

            phis = evaluate_basis(basis, grid) # (num_ao_basis, N)
            f_g =   f_nm @ phis
            
            if f_nm.ndim == 2:
                f_g = np.sum(f_g * phis.conj(), axis=0)
            return f_g

        def to_ao(self, f_g, grid: np.ndarray, weights: np.ndarray, basis=None):
            """
            Expresses grid quantity on the AO basis

            Parameters
            ----------
            f_g: np.ndarray
                Function expressed in g points 
            grid: np.ndarray, shape (N, 3)
                grid of length N.
            weights: np.ndarray, shape (N, )
                weights for integral
            basis:
                pyscf basis. If None, use sekf.basis
            Returns
            -------
            f_nm: np.ndarray
                f_g in ao basis
            """
            if basis is None:
                basis = self.basis
                
            phis = evaluate_basis(basis, grid)
            f_nm = contract( 'ap,p,p,bp->ab', phis.conj(), f_g, weights, phis)
            f_nm = 0.5 * (f_nm + f_nm.T)
            return f_nm
        
        def orbitals(self, C, grid):
            """
            Obtains orbitals on grid

            Parameters
            ----------
            C: np.ndarray
                Molecular Orbitals on Atomic Orbital basis set. 
            grid: np.ndarray, shape (N, 3)
                grid of length N.

            Returns
            -------
            mat_g: np.ndarray
                Orbitals in g points in space
            """

            phis = evaluate_basis(self.basis, grid)
            mat_g = C.T.dot(phis)
            return mat_g

        def laplacian_density(self, density, grid):
            """
            Calculates the laplacian of the density
            
            Parameters
            ----------
            density: np.ndarray
                Density in the ao basis 
            grid: np.ndarray, shape (N, 3)
                grid of length N.

            Returns
            -------
            lap_density: np.ndarray
                Laplacian of density given on g points in space
            """

            lap_density = evaluate_density_laplacian( density, self.basis, grid)
            return lap_density
    
        def gradient_density(self, density, grid):        
            """
            Evaluatges gradient of density on requested grid

            Parameters
            ----------
            density: np.ndarray
                Density in the ao basis 
            grid: np.ndarray, shape (N, 3)
                grid of length N.

            Returns
            -------
            grad_density: np.ndarray
                Gradient of density given on g points in space
            """

            grad_density = evaluate_density_gradient(density, self.basis, grid)
            return grad_density
            
        def ao_deriv(self, grid, derivs=[0,0,0], transform=None):
            """ 
            Calculates AO on the grid (and its derivatives). 
            If Transformation is given, e.g. ao2mo, MO will be given

            Parameters
            ----------
            derivs: List of 3 integers
                Array that corresponds to the derivative of each spatial coordinate (x,y,z)
                0 -> no derivative
                1 -> first derivative ...
                [1,0,3] -> first derivative on x. 
                           no derivative on y.
                           third derivative on z

            transform: np.ndarray
                Matrix to transform within basis. E.g. ao2mo -> MO will be given.
            grid: np.ndarray, shape (N, 3)
                grid of length N.

            Returns
            -------
            orbs_deriv: np.ndarray 
                Array of atomic orbitals and/or their derivatives. 
            """

            orbs_deriv = evaluate_deriv_basis(self.basis, grid, np.array(derivs), 
                                              transform=transform )

            return orbs_deriv

        # Specialized for methods. 
        # def posdef_kinetic_energy_density(self, density, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the positive-definite kinetic energy density on grid
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') 
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_posdef_kinetic_energy_density(density, self.basis, points)

        #     return t

        # def kinetic_energy_density(self, density, alpha=-1/4, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Evaluates the general form of the kinetic energy density
        #     t = 1/2 \nabla \cdot \nabla \gamma(r,r') + alpha \nabla^2 n(r)
        #     """

        #     points = self.assert_grid(grid)
        #     t = evaluate_general_kinetic_energy_density(density, self.basis, points, alpha=alpha)

        #     return t

        # def kinetic_energy_density_pauli(self, C, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Obtains kinetic energy density in terms of the Pauli kinetic energy density
            
        #     Parameters
        #     ----------
        #     C: np.ndarray
        #         Occupied Molecular Orbitals
        #     """

        #     points = self.assert_grid(grid)

        #     density = C @ C.T
        #     density_g = self.density(density, grid=grid)

        #     basis_dx = self.ao_deriv(derivs=[1,0,0], transform=None, grid=grid)
        #     basis_dy = self.ao_deriv(derivs=[0,1,0], transform=None, grid=grid)
        #     basis_dz = self.ao_deriv(derivs=[0,0,1], transform=None, grid=grid)

        #     if method == 'grid':
        #         orbs = self.orbitals(C, grid=grid)
        #         d_orbs = ((basis_dx + basis_dy + basis_dz).T @ C).T
        #         tau_p = np.zeros_like( density_g )
        #         for i in range(C.shape[1]):
        #             for j in range(C.shape[1]):
        #                 if i == j:
        #                     pass
        #                 else:
        #                     tau_p += np.abs( orbs[i,:] * (d_orbs[j,:]) - orbs[j,:] * (d_orbs[i,:]) )**2

        #     elif method == 'basis':
        #         basis = self.ao_deriv(grid=grid)
        #         dx = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dx.T)
        #         dy = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dy.T)
        #         dz = contract('pm,mi,nj,pn->ijp', basis.T, C, C, basis_dz.T)
            
        #         dx = (dx - np.transpose(dx, (1, 0, 2))) ** 2
        #         dy = (dy - np.transpose(dy, (1, 0, 2))) ** 2
        #         dz = (dz - np.transpose(dz, (1, 0, 2))) ** 2

        #         occ = np.ones(C.shape[1])
        #         occ_matrix = np.expand_dims(occ, axis=0) @ np.expand_dims(occ, axis=1)

        #         tau_p = np.sum((dx + dy + dz).T * occ_matrix, axis=(1,2)) 


        #     tau_p /= (2*density_g)

        #     return tau_p

        # def avg_local_orb_energy(self, density, orbitals, eigvals, grid='spherical'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates average local orbital energy. Described by Staroverov
        #     J. Chem. Phys. 146, 084103. [Equations 4 and/or 6]
        #     $$
        #     e_tilde = 1/n(r) * [ \sum_i \varepsilon_i * | \phi_i(r) |^2 ]
        #     $$
        #     """

        #     points = self.assert_grid(grid)

        #     phis      = evaluate_basis(self.basis, points)
        #     density_g = self.density(density=density, grid=grid)
        #     e_tilde   = contract('xp, xo, xo, x, xp-> p', phis, 
        #                                                 orbitals, orbitals, eigvals, 
        #                                                 phis) / density_g

        #     return e_tilde

        # def external_tilde(self, grid='spherical', method='grid'):
        #     """
        #     Please look at OuCarter or mRKS method
        #     Generates effective external potential from LDA exchange. Described by Ou + Carter. 
        #     J. Chem. Theory Comput. 2018, 14, 11, 5680–5689

        #     $$
        #     v^{~}{ext}(r) = \epsilon^{-LDA}(r) - \frac{\tau^{LDA}{L}}{n^{LDA}(r)}
        #     - v_{H}^{LDA}(r) - v_{xc}^{LDA}(r)
        #     $$
        #     (22) in [1].
        #     """

        #     points = self.assert_grid(grid)

        #     # LDA results
        #     Da0, Db0 = self.mf.make_rdm1()
        #     Ca0, Cb0 = self.mf.mo_coeff
        #     ea0, eb0 = self.mf.mo_energy

        #     da0_g = self.density(Da0, grid)
        #     db0_g = self.density(Db0, grid)

        #     # LDA exchange
        #     cx = -(3/np.pi)**(1/3)
        #     vxca = cx * da0_g ** (1/3)
        #     vxcb = cx * db0_g ** (1/3)

        #     # External tilde
        #     e_tilde = self.avg_local_orb_energy(Da0, Ca0, ea0, grid=grid)
        #     lap     = self.laplacian_density(Da0, grid=grid)
        #     grad    = self.gradient_density(Da0, grid=grid)
        #     grad    = grad[:,0] + grad[:,1] + grad[:,2]
        #     hartree = self.hartree(Da0, grid=grid)
            
        #     tau_l  = self.kinetic_energy_density_pauli(Ca0, grid=grid, method=method)
        #     tau_l += - 0.25 * lap + np.abs( grad )**2 / (8*da0_g) 
        
        #     external_tilde = e_tilde - tau_l/da0_g - hartree - vxca

        #     return external_tilde

