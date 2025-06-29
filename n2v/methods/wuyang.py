"""
wuyang.py

Functions associated with wuyang inversion
"""

import numpy as np
from opt_einsum import contract
from scipy.optimize import minimize

class WuYang():
    """
    Performs Optimization as in: 10.1063/1.1535422 - Qin Wu + Weitao Yang

    Attributes:
    -----------
    lambda_rgl: {None, float}. If float, lambda-regularization is added with lambda=lambda_rgl.
    """

    regul_norm = None  # Regularization norm: ||v||^2
    lambda_reg = None  # Regularization constant
    pinv_cutoff = 1e-2
    
    def wuyang(self, opt_max_iter, reg=None, tol=1e-7, gtol=1e-3,
               opt_method='trust-krylov', opt=None):
        """
        Calls scipy minimizer to minimize lagrangian. 
        """
        self.lambda_reg = reg
        if opt is None:
            opt = {"disp"    : False}
        opt['maxiter'] = opt_max_iter
        opt['gtol'] = gtol
        # Initialization for D and C
        self._diagonalize_with_potential_pbs(self.v_pbs)

        if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
            opt_results = minimize( fun = self.lagrangian_wy,
                                    x0  = self.v_pbs,
                                    jac = self.gradient_wy,
                                    method  = opt_method,
                                    tol     = tol,
                                    options = opt
                                    )

        else:
            opt_results = minimize( fun = self.lagrangian_wy,
                                    x0  = self.v_pbs,
                                    jac = self.gradient_wy,
                                    hess = self.hessian_wy,
                                    method = opt_method,
                                    tol    = tol,
                                    options = opt
                                    )

        if opt_results.success == False:
            self.v_pbs = opt_results.x
            self.opt_info = opt_results
            raise ValueError("Optimization was unsucessful (|grad|=%.2e) within %i iterations, "
                             "try a different initial guess. %s"% (np.linalg.norm(opt_results.jac), opt_results.nit, opt_results.message)
                             )
        else:
            print("Optimization Successful within %i iterations! "
                  "|grad|=%.2e" % (opt_results.nit, np.linalg.norm(opt_results.jac)))
            self.v_pbs = opt_results.x
            self.opt_info = opt_results

    def _diagonalize_with_potential_pbs(self, v):
        """
        Diagonalize Fock matrix with additional external potential
        """
        self.v_pbs = np.copy(v)
        vks_a = contract("ijk,k->ij", self.S3, v[:self.npbs]) + self.va
        fock_a = self.V + self.T + vks_a 
        self.Ca, self.Coca, self.Da, self.eigvecs_a = self.diagonalize( fock_a, self.nalpha )

        if self.ref == 1:
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.Ca, self.Coca, self.Da, self.eigvecs_a
            self.Fock =  fock_a
        else:
            vks_b = contract("ijk,k->ij", self.S3, v[self.npbs:]) + self.vb
            fock_b = self.V + self.T + vks_b        
            self.Cb, self.Cocb, self.Db, self.eigvecs_b = self.diagonalize( fock_b, self.nbeta )
            self.Fock =  (fock_a, fock_b)
        return 

    def _lagrangian_wy_R(self, v):
        """
        Lagrangian to be minimized wrt external potential
        Equation (5) of main reference
        """
        assert self.ref == 1
        assert np.allclose(self.Da, self.Db)
        self._diagonalize_with_potential_pbs(v)
        
        D = self.Da + self.Db
        grad_a = contract('ij,ijt->t', (D - self.Dt), self.S3)

        kinetic     =   np.sum(self.T * D)
        potential   =   np.sum((self.V + self.va) * (D - self.Dt))
        optimizing  =   np.sum(v * grad_a)

        L = (kinetic + potential + optimizing) 

        # Add lambda-regularization
        if self.lambda_reg is not None:
            T = self.T_pbs
            norm = 2 * (v.conj() @ T @ v)
            L -= norm * self.lambda_reg
            self.regul_norm = norm
        print(f"L={-L:.5f} |v|={np.linalg.norm(v):.5f}")
        return - L

    def _gradient_wy_R(self, v):
        """
        Calculates gradient wrt target density
        Equation (11) of main reference
        """
        assert self.ref == 1
        self._diagonalize_with_potential_pbs(v)
        
        D = self.Da + self.Db
        
        self.grad_a = contract('ij,ijt->t', (D - self.Dt), self.S3)
        self.grad   = self.grad_a

        if self.lambda_reg is not None:
            T = self.T_pbs
            rgl_vector = 2 * self.lambda_reg*np.dot(T, v)
            self.grad -= rgl_vector
        print(f"|grad|={np.linalg.norm(self.grad)}  |v|={np.linalg.norm(v):.5f}")
        return -self.grad

    def _hessian_wy_R(self, v):
        """
        Calculates gradient wrt target density
        Equation (13) of main reference
        """
        assert self.ref == 1
        self._diagonalize_with_potential_pbs(v)
        
        na, nb = self.nalpha, self.nbeta

        eigs_diff_a = self.eigvecs_a[:na, None] - self.eigvecs_a[None, na:]
        C3a = contract('mi,va,mvt->iat', self.Ca[:,:na].conj(), self.Ca[:,na:], self.S3)
        Ha = contract('iau,iat,ia->ut', C3a.conj(), C3a, eigs_diff_a**-1)

        if self.lambda_reg is not None:
            Ha -= 2 * self.T_pbs * self.lambda_reg
        Hs = Ha
        return -Hs.real


    def _lagrangian_wy_U(self, v):
        """
        Lagrangian to be minimized wrt external potential
        Equation (5) of main reference
        """
        assert self.ref == 2
        self._diagonalize_with_potential_pbs(v)
        
        grad_a = contract('ij,ijt->t', (self.Da - self.Dt[0]), self.S3)
        grad_b = contract('ij,ijt->t', (self.Db - self.Dt[1]), self.S3)

        D = self.Da + self.Db
        kinetic     =   np.sum(self.T * D)
        potential   =   np.sum((self.V + self.va) * (self.Da - self.Dt[0]))
        potential  +=   np.sum((self.V + self.vb) * (self.Db - self.Dt[1]))
        
        optimizing  =   np.sum(v[:self.npbs] * grad_a)
        optimizing +=   np.sum(v[self.npbs:] * grad_b)
        
        L = kinetic + potential + optimizing

        # Add lambda-regularization
        if self.lambda_reg is not None:
            T = self.T_pbs
            norm = (v[self.npbs:] @ T @ v[self.npbs:]) + (v[:self.npbs] @ T @ v[:self.npbs])
            L -= norm * self.lambda_reg
            self.regul_norm = norm
        return - L

    def _gradient_wy_U(self, v):
        """
        Calculates gradient wrt target density
        Equation (11) of main reference
        """
        assert self.ref == 2
        self._diagonalize_with_potential_pbs(v)
        
        
        self.grad_a = contract('ij,ijt->t', (self.Da - self.Dt[0]), self.S3)
        self.grad_b = contract('ij,ijt->t', (self.Db - self.Dt[1]), self.S3)

        self.grad   = np.concatenate(( self.grad_a, self.grad_b ))

        if self.lambda_reg is not None:
            T = self.T_pbs
            self.grad[:self.npbs] -= 2 * self.lambda_reg*np.dot(T, v[:self.npbs])
            self.grad[self.npbs:] -= 2 * self.lambda_reg*np.dot(T, v[self.npbs:])
        return -self.grad

    def _hessian_wy_U(self, v):
        """
        Calculates gradient wrt target density
        Equation (13) of main reference
        """
        assert self.ref == 2
        self._diagonalize_with_potential_pbs(v)

        na, nb = self.nalpha, self.nbeta

        eigs_diff_a = self.eigvecs_a[:na, None] - self.eigvecs_a[None, na:]
        C3a = contract('mi,va,mvt->iat', self.Ca[:,:na].conj(), self.Ca[:,na:], self.S3)
        Ha = 2 * contract('iau,iat,ia->ut', C3a.conj(), C3a, eigs_diff_a**-1)

        eigs_diff_b = self.eigvecs_b[:nb, None] - self.eigvecs_b[None, nb:]
        C3b = contract('mi,va,mvt->iat', self.Cb[:,:nb].conj(), self.Cb[:,nb:], self.S3)
        Hb = 2 * contract('iau,iat,ia->ut', C3b.conj(), C3b, eigs_diff_b**-1)
        if self.lambda_reg is not None:
            Ha -= 2 * self.T_pbs * self.lambda_reg
            Hb -= 2 * self.T_pbs * self.lambda_reg
        Hs = np.block(
                        [[Ha,                               np.zeros((self.npbs, self.npbs))],
                        [np.zeros((self.npbs, self.npbs)), Hb                              ]]
                    )

        return -Hs.real

    def lagrangian_wy(self, v):
        if self.ref == 1:
            return self._lagrangian_wy_R(v)
        else:
            return self._lagrangian_wy_U(v)
        
    def gradient_wy(self, v):
        if self.ref == 1:
            return self._gradient_wy_R(v)
        else:
            return self._gradient_wy_U(v)
        
    def hessian_wy(self, v):
        if self.ref == 1:
            return self._hessian_wy_R(v)
        else:
            return self._hessian_wy_U(v)
        
    def hessian_v_p_wy(self, v, p):
        return np.linalg.pinv(self.hessian_wy(v), rcond=self.pinv_cutoff) @ p
    
    def hessian_v_v_wy(self, v):
        return np.linalg.pinv(self.hessian_wy(v), rcond=self.pinv_cutoff) @ v
    
    #TODO: Implement Prof Tim Gould's method here
    
    def find_regularization_constant_wy(self, opt_max_iter, opt_method="trust-krylov", gtol=1e-3,
                                     tol=None, opt=None, lambda_list=None):
        """
        Finding regularization constant lambda.

        Note: it is recommend to set a specific convergence criteria by opt or tol,
                in order to control the same convergence
                for different lambda value.

        After the calculation is done, one can plot the returns to select a good lambda.

        Parameters:
        -----------
        opt_max_iter: int
                    maximum iteration

        opt_method: string default: "trust-krylov"
            opt_methods available in scipy.optimize.minimize

        tol: float
            Tolerance for termination. See scipy.optimize.minimize for details.
        gtol: float
             gtol for scipy.optimize.minimize: the gradient norm for
             convergence
        opt: dictionary, optional
            if given:
                scipy.optimize.minimize(method=opt_method, options=opt).
            Notice that opt has lower priorities than opt_max_iter and gtol.

        lambda_list: np.ndarray, optional
            A array of lambda to search; otherwise, it will be 10 ** np.linspace(-1, -7, 7).

        Returns:
        --------
        lambda_list: np.ndarray
            A array of lambda searched.

        P_list: np.ndarray
            The value defined by [Bulat, Heaton-Burgess, Cohen, Yang 2007] eqn (21).
            Corresponding to lambda in lambda_list.

        Ts_list: np.ndarray
            The Ts value for each lambda.


        """

        Ts_list = []
        L_list = []
        v_norm_list = []


        if lambda_list is None:
            lambda_list = 10 ** np.linspace(-3, -9, 7)

        if opt is None:
            opt = {"disp"    : False}
        opt['maxiter'] = opt_max_iter
        opt['gtol'] = gtol

        self.lambda_reg = None
        # Initial calculation with no regularization
        # Initialization for D and C
        self._diagonalize_with_potential_pbs(self.v_pbs)

        if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
                initial_result = minimize(fun=self.lagrangian_wy,
                                    x0=self.v_pbs,
                                    jac=self.gradient_wy,
                                    method=opt_method,
                                    tol=tol,
                                    options=opt
                                    )
        else:
            initial_result = minimize(fun=self.lagrangian_wy,
                                   x0=self.v_pbs,
                                   jac=self.gradient_wy,
                                   hessp=self.hessian_v_v_wy,
                                   method=opt_method,
                                   tol=tol,
                                   options=opt
                                   )
        if initial_result.success == False:
            raise ValueError("Optimization was unsucessful (|grad|=%.2e) within %i iterations, "
                             "try a different intitial guess"% (np.linalg.norm(initial_result.jac), initial_result.nit)
                             + initial_result.message)
        else:
            L0 = -initial_result.fun
            initial_v0 = initial_result.x  # This is used as the initial guess for with regularization calculation.

        for reg in lambda_list:
            self.lambda_reg = reg

            if opt_method.lower() == 'bfgs' or opt_method.lower() == 'l-bfgs-b':
                opt_results = minimize(fun=self.lagrangian_wy,
                                       x0=initial_v0,
                                       jac=self.gradient_wy,
                                       method=opt_method,
                                       tol=tol,
                                       options=opt
                                       )

            else:
                opt_results = minimize(fun=self.lagrangian_wy,
                                       x0=initial_v0,
                                       jac=self.gradient_wy,
                                       hess=self.hessian_wy,
                                       method=opt_method,
                                       tol=tol,
                                       options=opt
                                       )


            Ts_list.append(np.sum(self.T * (self.Da + self.Db)))
            v_norm_list.append(self.regul_norm)
            L_list.append(-opt_results.fun + self.lambda_reg * self.regul_norm)

        P_list = lambda_list * np.array(v_norm_list) / (L0 - np.array(L_list))

        return lambda_list, P_list, np.array(Ts_list)

