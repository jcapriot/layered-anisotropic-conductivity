from objfunc import ObjectiveFunction
from anisotropic_potential import LayeredHalfspace
import numpy as np

class AnisotropicPotentialObjectiveFunction(ObjectiveFunction):

    def __init__(self, z, forward_parameters, d_obs, Wd=None):
        self.z = z
        self.d_obs = d_obs
        self.Wd = Wd
        self.d_pre = None
        self.forward_parameters = forward_parameters
        self.n = len(z)
        self.last_x = np.empty([])
        self.last_dx = np.empty([])

    def f(self, x):
        if not np.allclose(x, self.last_x):
            self._update_pre_data(x)
            self.last_x = x
        Wd = self.Wd
        del_d = self.d_pre-self.d_obs
        if self.Wd is not None:
            del_d = Wd.dot(del_d)
        return del_d.dot(del_d)

    def d(self, x):
        if not np.allclose(x, self.last_dx):
            self._update_Jac(x)
            self.last_dx = x
            self.last_x = x
        n = self.n
        Wd = self.Wd
        del_d = self.d_pre-self.d_obs
        if self.Wd is not None:
            del_d = Wd.dot(Wd.dot(del_d))
        J = self._J
        d = J.T.dot(del_d)
        return d

    def H(self, x):
        J = self._J
        y = J.dot(x)
        if self.Wd is not None:
            y = self.Wd.dot(self.Wd.dot(y))
        return J.T.dot(y)

    def _update_pre_data(self, x):
        n = self.n
        y = x.reshape((6, n)).copy()
        np.exp(y[0], y[0])
        np.exp(y[1], y[1])
        np.exp(y[2], y[2])
        space = LayeredHalfspace(self.z, *y)
        self.d_pre = space.voltage(*self.forward_parameters)
        return space

    def _update_Jac(self, x):
        space = self._update_pre_data(x)
        n = self.n
        y = x.reshape((6, n)).copy()
        np.exp(y[0], y[0])
        np.exp(y[1], y[1])
        np.exp(y[2], y[2])
        Js = space.voltage_ds(*self.forward_parameters)
        Js = space.J_params(*Js)
        Js1, Js2, Js3, _, _, _ = Js
        Js1 *= y[0]
        Js2 *= y[1]
        Js3 *= y[2]
        self._J = np.hstack(Js)