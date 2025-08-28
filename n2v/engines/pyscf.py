"""
Provides interface n2v interface to PySCF
"""
import warnings, time
import numpy as np
from scipy import linalg as scilag
from opt_einsum import contract

from .engine import Engine

try:
    from pyscf import gto, dft, df, scf
    from pyscf.scf.hf import get_jk
    has_pyscf = True
except ImportError:
    has_pyscf = False


def _randomized_svd_hermitian(A, m, *, factor=2, n_iter=2, random_state=None):
    """
    Approximate the leading m-dimensional singular (eigen) subspace of a
    low-rank Hermitian matrix A ∈ ℂ^{n×n} using randomized range finding.

    Parameters
    ----------
    A : (n, n) array_like
        Hermitian (symmetric) input matrix.  Only matrix–vector products are used,
        so A can be a NumPy/SciPy array or any LinearOperator implementing @.
    m : int
        Target subspace dimension (rank you care about).
    factor : int, default 2
        Oversampling multiplier.  The algorithm samples k = m * factor random
        vectors and finally returns the best m directions.
    n_iter : int, default 2
        Number of power-iteration steps to polish the subspace.  n_iter = 0 is
        fastest but least accurate; 2–4 is typical.
    random_state : int or np.random.Generator, optional
        For reproducibility.

    Returns
    -------
    U : (n, m) ndarray
        Orthonormal basis for the estimated leading m-dimensional eigen-subspace.
    S : (m,) ndarray
        Approximate leading singular (eigen) values, sorted descending.

    Notes
    -----
    * For a Hermitian A the left and right singular spaces coincide, so only U
      is returned.
    * The cost is O(k · n²) for dense A, but only O(k · nnz(A)) for sparse A if
      you supply a sparse/LinearOperator handle.
    """
    n = A.shape[0]
    k = int(np.ceil(m * factor))                      # total random directions
    rng = np.random.default_rng(random_state)

    # 1. Random sampling
    Omega = rng.standard_normal((n, k)) # real # + 1j * rng.standard_normal((n, k))
    Omega = Omega / np.linalg.norm(Omega, axis=0)
    Y = A @ Omega                       # (n × k)

    # 2. Power iterations (for ill-conditioned spectra)
    for _ in range(n_iter):
        Y = A @ (A @ Y)                 # (A·A)·Y keeps symmetry
        # Optional re-orthogonalize each step to stabilize
        Y, _ = np.linalg.qr(Y, mode='reduced')

    # 3. Orthonormalize to get range(Q)
    Q, _ = np.linalg.qr(Y, mode='reduced')        # (n × k)

    # 4. Project A into this subspace (small k×k matrix)
    B = Q.conj().T @ (A @ Q)                      # Hermitian k×k

    # 5. Exact eigen-decomposition of B (cheap)
    eigvals, eigvecs = np.linalg.eigh(B)          # ascending order
    idx = np.argsort(eigvals)[::-1]               # descending
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # 6. Lift back to full space and truncate to m
    U_hat = Q @ eigvecs[:, :m]                    # n × m
    S_hat = eigvals[:m].real                      # eigenvalues == singular vals
    return S_hat, U_hat

def _randomized_eig(A, smin, factor, n_iter, subspace_frac=0.1):
    m = int(len(A) * subspace_frac)
    s, V = _randomized_svd_hermitian(A, m, factor=factor, n_iter=n_iter)
    assert s[-1] > smin, "The subspace is too small. Need to increase subspace_frac"
    return s[s>smin], V[:,:s>smin]

if has_pyscf:
    from ..grid import PySCFGrider
    class PySCFEngine(Engine):
        """
        PySCF Engine
        """

        def set_system(self, molecule, pbs, ref=1, pbsrotation=True, product_basis=True, pbsrotcutoff=1e-1):
            """
            Stores basic information from a PySCF calculation

            Parameters
            ----------
            
            mol: pyscf.gto.mole.Mole
                Pyscf molecule object
            ref: {1,2}
                1 -> Restricted 
                2 -> Unrestricted
            pbs: str.
                Basis set for expressing inverted potential
            pbsrotation: bool.
                Whether to svd the pbs to get a orthonormal pbs.
            product_basis: bool.
                If we use pbs_new(r)=pbs(r) cross-plus ao(r)ao(r) also as the basis.
            pbsrotcutoff: float.
                If we rotate the basis, we get rid of the space with eig<pbsrotcutoff.
            """
            self.mol = molecule
            self.pbs_str = pbs
            self.ref = ref
            self.pbsrotation = pbsrotation
            self.product_basis = product_basis
            self.pbsrotcutoff = pbsrotcutoff
            if self.product_basis and not self.pbsrotation:
                self.pbsrotation = True
                print("Product basis sets best work with rotation. Set pbsrotation=True.")
            
            print(f"pbs={self.pbs_str} ref={self.ref} orth_pbs={self.pbsrotation} prod_pbs={self.product_basis}")
            
            self.initialize_pbs()
            
            self.nalpha = self.mol.nelec[0]
            self.nbeta = self.mol.nelec[1]
            
            return

        def initialize_pbs(self):
            """
            Initialize the pbs object.
            """
            self.pbs = gto.Mole()
            self.pbs.atom = self.mol.atom
            self.pbs.basis = self.pbs_str
            self.pbs.build()
            self.nbf = self.mol.nao
            self.npbs = self.pbs.nao
            
            if self.pbsrotation:
                # TODO: I do not know how to calculate the integral \\int dr ao(r) ao(r) pbs(r) pbs(r)
                # So I would limit the product basis to be ao(r) ao(r) instead of pbs(r) pbs(r)
                S2_pbs = self.get_S(self.pbs)
                if self.product_basis:
                    print("Constructing PBS={pbs(r), ao(r)ao(r)}")
                    S3time = -time.perf_counter()
                    S3_pbs = self.get_S3(self.mol, self.pbs).reshape((self.nbf**2, self.npbs))
                    S3time += time.perf_counter()
                    print(f"Get S3 (ijP)=\\int dr ao_i(r)ao_j(r)pbs_P(r) of shape {S3_pbs.shape} in {S3time:.2f}s")
                    
                    S4time = -time.perf_counter()
                    S4_pbs = self.get_S4(self.mol).reshape((self.nbf**2, self.nbf**2))
                    S4time += time.perf_counter()
                    print(f"Get S4 (ijkl)=\\int dr ao_i(r)ao_j(r)ao_k(r)ao_l(r) of shape {S4_pbs.shape} in {S4time:.2f}s")
                    
                    S_pbs = np.block([[S2_pbs, S3_pbs.T], [S3_pbs, S4_pbs]])
                    self.S3pbs = np.concatenate((S3_pbs, S4_pbs), axis=1)
                    self.S3pbs = self.S3pbs.reshape((self.nbf, self.nbf, -1))
                else:
                    S_pbs = S2_pbs
                    self.S3pbs = self.get_S3(self.mol, self.pbs)
                S_pbs = (S_pbs + S_pbs.T) / 2.
                time_diag = -time.perf_counter()
                
                # this is the largest bottleneck. Might need rSVD.
                e, v = _randomized_eig(S_pbs, self.pbsrotcutoff, 2, 2, subspace_frac=0.1)
                time_diag += time.perf_counter()
                print(f"Diag the overlap of PBS of shape {S_pbs.shape} in {time_diag:.2f}s.")
                
                self.pbs_rot = np.copy(v[:,e>self.pbsrotcutoff])
                self.npbs = self.pbs_rot.shape[1]
                self.S3pbs = self.S3pbs @ self.pbs_rot
            else:
                self.S3pbs = self.get_S3(self.mol, self.pbs)
            return
        
        def initialize_grid(self):
            """
            Initializes different grid object.
            """

            self.grid = PySCFGrider(self.mol, self.pbs)
            return
        
        def get_T(self):
            """
            Generates Kinetic Operator in AO basis.
            
            Returns
            -------
            T: np.ndarray. Shape: (nbf, nbf)
            """
            return self.mol.intor('int1e_kin')

        def get_Tpbas(self):
            """
            Generates Kinetic Operator in AO basis for potential basis. 

            Returns
            -------
            T_pbas: np.ndarray. Shape: (nbf, nbf)
            """
            return self.pbs.intor('int1e_kin')

        def get_V(self):
            """
            Generates External Potential in AO basis

            Returns
            -------
            V: np.ndarray. Shape: (nbf, nbf)
            """
            return self.mol.intor('int1e_nuc')

        def get_A(self):
            """
            Generates S^(-0.5)

            Returns
            -------
            A: np.ndarray. Shape: (nbf, nbf)
            """
            A = self.mol.intor('int1e_ovlp')
            A = scilag.fractional_matrix_power(A, -.5)
            return A

        def get_S(self, mol=None):
            """
            Builds Overlap matrix of AO basis
            
            Parameters
            ----------
            mol: pyscf's mol
                If None, use self.mol.
            
            Returns
            -------
            S: np.ndarray. Shape: (nbf, nbf)
            """
            if mol is None:
                mol = self.mol
            return mol.intor('int1e_ovlp')

        def get_S3(self, mol=None, pbs=None):
            """
            Builds 3 Overlap Matrix. 
            Manually built since Pyscf does not support it. 

            Returns
            -------
            S3: np.ndarray. Shape: (nbf, nbf, nbf or npbs)
                Third dimension depends on wether an additional basis is used. 
            """
            if mol is None:
                mol = self.mol
            if pbs is None:
                pbs = self.pbs
            # returns an array of shape (naux, nao, nao)
            S3 = df.incore.aux_e2(mol, pbs, intor='int3c1e', comp=1)
            return S3
        
        def get_S32e(self, mol=None, pbs=None):
            """
            Builds 3 Overlap Matrix. 
            Manually built since Pyscf does not support it. 

            Returns
            -------
            S3: np.ndarray. Shape: (nbf, nbf, nbf or npbs)
                Third dimension depends on wether an additional basis is used. 
            """
            if mol is None:
                mol = self.mol
            if pbs is None:
                pbs = self.pbs
            # returns an array of shape (naux, nao, nao)
            S32e = df.incore.aux_e2(mol, pbs, intor='int3c1e', comp=1)
            return S32e

        def get_S4_DF(self, mol=None):
            """
            Obtains a 4 AO Overlap Matrix using Density Fitting.
            """
            if mol is None:
                mol = self.mol
                
            mol_aux = df.make_auxmol(mol)
            S_mnP = df.incore.aux_e2(mol, mol_aux, intor='int3c1e', comp=1)
            S_PQ = mol_aux.intor('int1e_ovlp')
            S_PQinv = np.linalg.pinv(S_PQ, rcond=1e-9)
            S4 = contract('mnP,PQ,rsQ->mnrs', S_mnP, S_PQinv, S_mnP)
            return S4

        def get_S4(self, mol=None):
            """
            Obtains a 4 AO Overlap Matrix analytically.
            TODO: there is a way to evaluate for different bs: using mol with some stacked basis.
            
            """
            if mol is None:
                mol = self.mol
            try:
                ovlp4 = mol.intor('int4c1e', comp=1)
            except:
                warnings.warn("Direct 4 integral not found. Using DF instead.")
                ovlp4 = self.get_S4_DF(mol)
            return ovlp4

        def compute_hartree(self, D):
            """
            Computes Hartree Operator in AO basis from atomic orbitals.

            Parameters
            ----------
            D: np.ndarray
                Density matrices.
                Following the pyscf interface.
                if ref==1:
                    D = D shape (nbf, nbf)
                elif ref==2:
                    D = [Da, Db] shape (2, nbf, nbf)
            """
            if D.ndim == 2:
                if self.ref != 1:
                    raise ValueError("Unrestricted calculations needs Db. ref={self.ref}.")
            elif D.ndim == 3:
                assert len(D) == 2
                D = D[0] + D[1]
                
            J = get_jk(self.mol, dm=D)[0]
            return J


        
        
        # Post-SCF
        def diagonalize( self, matrix, ndocc ):
            """
            Diagonalizes Fock Matrix

            Parameters
            ----------
            marrix: np.ndarray
                Matrix to be diagonalized
            ndocc: int
                Number of occupied orbitals

            Returns
            -------
            C: np.ndarray
                Orbital Matrix
            Cocc: np.ndarray
                Occupied Orbital Matrix
            D: np.ndarray
                Density Matrix
            eigves: np.ndarray
                Eigenvalues
            """

            self.A = self.get_A()

            Fp = self.A.dot(matrix).dot(self.A)
            eigvecs, Cp = np.linalg.eigh(Fp)
            C = self.A.dot(Cp)
            Cocc = C[:, :ndocc]
            D = contract('pi,qi->pq', Cocc, Cocc)
            return C, Cocc, D, eigvecs
        