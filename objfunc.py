import numpy as np
from scipy.spatial.distance import cdist

class ObjectiveFunction(object):
    """
    This class represents an objective function. While it can be used
    on its own, it is primarily meant to be extended for different
    objective functions.

    ObjectiveFunction(func,d_func,H_func,update=None)
    With a callable function "func" which returns the value, a callable
    function "d_func" which returns the derivative of "func", a callable
    function "H_func" which represents the Hessian operation times a
    vector. A callable update function "update" can be provided if any
    value needs to be updated between iterations.

    Parameters
    ----------
    func : callable function
    d_func : callable function
    H_func : callable function
    update : callable function

    Notes
    -----
    The objective function class is used to represent the common operations
    in a Newton-like minimization process: the value of the function, the
    derivative of the function, and the Hessian operation.

    The objective function class can be used on it's own, mostly for simple
    functions, but it is primarily meant to be extended to form a user
    defined objective function. These objective functions can be added
    together, and multiplied by scalars (common operations in geophysical
    inversions).

    The update function is optional, but if supplied it expects to take an
    argument of the current position, i.e. update(x). The update function is
    useful to do operations that will be needed in the other functions, i.e.
    building Jacobian matrices.
    """

    _funcs = []
    _scale = None
    _last_val = None

    def __init__(self, funcs, scale=None):
        self._funcs = funcs
        self._scale = scale

    def f(self,x):
        """
        Returns the value of the objective function evaluated at x
        """
        if(len(self._funcs)==0):
            raise NotImplementedError

        y = 0.0
        for f in self._funcs:
            val = f(x)
            f._last_val = val
            y += val
        if self._scale != 1 and self._scale is not None:
            y *= self._scale
        self._last_val = y
        return y

    def __call__(self,x):
        return self.f(x)

    def d(self,x):
        """
        Returns the derivative of the objective function evaluated at x
        """
        if(len(self._funcs)==0):
            raise NotImplementedError

        y = np.zeros_like(x)
        for f in self._funcs:
            y += f.d(x)
        if self._scale == 1.0 or self._scale is None:
            return y
        else:
            return self._scale*y

    def H(self,x):
        """
        Returns the Hessian times a vector x.

        Notes
        -----
        The Hessian should not be evaluated at "x", instead it should be
        set prior to entering this function.
        """

        if(len(self._funcs)==0):
            raise NotImplementedError

        y = np.zeros_like(x)
        for f in self._funcs:
            y += f.H(x)
        if self._scale == 1.0 or self._scale is None:
            return y
        else:
            return self._scale*y

    def update(self,x):
        """
        Updates the objective function using the model x.

        Notes
        -----
        This is a good place to update the Hessian and derivative operations
        before they are evaluated, i.e. if you need to evaluate and store a
        Jacobian matrix.
        """
        if len(self._funcs)>0:
            for f in self._funcs:
                try:
                    f.update(x)
                except AttributeError:
                    pass

    def __add__(self,other):
        if issubclass(type(other), ObjectiveFunction):
            if(len(self._funcs)>0):
                funcs = list(self._funcs)
            else:
                funcs = [self]
            funcs.append(other)
            return ObjectiveFunction(funcs)
        else:
            return NotImplemented

    def __mul__(self,other):
        try:
            other*1.0
        except TypeError:
            return NotImplemented

        if len(self._funcs)==0:
            return ObjectiveFunction([self],scale=other)
        else:
            return ObjectiveFunction(self._funcs, scale=other)

    def __rmul__(self,other):
        return self*other

    def __str__(self):
        if(len(self._funcs)==0):
            out = str(self._last_val)
            if self._scale is None:
                return out
            else:
                return str(self._scale)+'*('+out+')'
        else:
            out = ''
            for i in range(len(self._funcs)):
                f = self._funcs[i]
                if(f._scale is None):
                    out += str(f)+' + '
                else:
                    out += str(f._scale)+'*('+str(f)+')'+' + '
            return out[:-3]


class DataMisfitLinear(ObjectiveFunction):
    """
    A linear data misfit measure objective function

    DataMisfitLinear(G,d_obs,Wd=None):
    Constructs a linear data misfit object with a matrix like forward
    operator G, observed data d_obs, and an optional data weighting matrix
    Wd. If G has been prescaled by Wd before construction, Wd should be
    None. If Wd is None, it is assumed to be identity (essentially).
    G and Wd must have dot() operations.

    Parameters
    ----------
    G : matrix like
    d_obs : numpy array
    Wd : matrix like

    Notes
    -----
    The function is defined as

    .. math::

      ||W_d(G\\vec{m}-\\vec{d}_{obs})||^2

    """

    def __init__(self,G,d_obs,Wd=None):
        self.G = G
        self.d_obs = d_obs
        self.Wd = Wd

    def f(self,x):
        """
        The data misfit of model x

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        value : float

        Notes
        -----
        The function is defined as

        .. math::

            ||W_d(G \\vec{x}-\\vec{d}_{obs})||^2

        d_obs is stored internally during this operation to be used in the
        derivative operation.
        """
        G = self.G
        self.d_pre = G.dot(x)
        del_d = self.d_pre-self.d_obs
        Wd = self.Wd
        if Wd is not None:
          del_d = Wd.dot(del_d)
        return del_d.dot(del_d)

    def d(self,x):
        """
        The derivative of the misfit function

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        deriv : numpy array

        Notes
        -----
        The derivative is defined as:

        .. math::

            G^T W_d^T W_d (G \\vec{x}-\\vec{d}_{obs})
        """
        G = self.G
        del_d = self.d_pre-self.d_obs
        Wd = self.Wd
        if Wd is not None:
          del_d = Wd.T.dot(Wd.dot(del_d))
        return G.T.dot(del_d)

    def H(self,x):
        """
        The Hessian of the misfit function times a vector

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        Hx : numpy array

        Notes
        -----
        Defined as:

        .. math::

            G^T W_d^T W_d G \\vec{x}

        """
        G = self.G
        temp = G.dot(x)
        Wd = self.Wd
        if Wd is not None:
          temp = Wd.T.dot(Wd.dot(temp))
        return G.T.dot(temp)


class ModelObjectiveFunction(ObjectiveFunction):
    """
    A simple model objective function

    ModelObjectiveFunction(WmTWm,mref=0.0):
    Constructs a model objective function with a reference model, and a
    measuring matrix WmTWm

    Parameters
    ----------
    WmTWm : matrix like
    Matrix used to measure the model objective function
    mref : scalar or numpy array
    Reference model

    Notes
    -----
    The function is defined as

    .. math::

      ||W_m(\\vec{m}-\\vec{m}_{ref})||^2
    """

    def __init__(self,WmTWm,mref=0.0):
        self.W = WmTWm
        self.mref = mref

    def f(self,x):
        """
        The model objective function measure of model x

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        value : float

        Notes
        -----
        The function is defined as:

        .. math::

            ||W_m(\\vec{x}-\\vec{m}_{ref})||^2

        """
        dm = x-self.mref
        return dm.dot(self.W.dot(dm))

    def d(self,x):
        """
        The derivative of the model objective function at model x

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        deriv : numpy array

        Notes
        -----
        The derivative of the model objective function is defined as:

        .. math::

            W_m^T W_m(\\vec{x}-\\vec{m}_{ref})
        """
        dm = x-self.mref
        return self.W.dot(dm)

    def H(self,x):
        """
        The hessian of the model objective function times a vector x

        Parameters
        ----------
        x : numpy array

        Returns
        -------
        Hx : numpy array

        Notes
        -----
        The Hessian operation for the model objective function is:

        .. math::

            W_m^TW_m\\vec{x}
        """
        return self.W.dot(x)


class FCMObjectiveFunction(ObjectiveFunction):
    def __init__(self, t, eta, w=None, q=2, vi=None, ui=None):
        self.q = q
        self.t = np.array(t)
        self.n_params = t.shape[1]
        self.n_clusters = t.shape[0]
        # self.C = t.shape[-1]
        self.v = vi
        self.u = ui
        self.eta = eta
        self.u = ui
        self.v = vi
        if w is None:
            self.w = np.ones(self.n_params)
        else:
            self.w = w

    def updateU(self, m, v):
        n_p = self.n_params
        m = m.reshape(n_p, -1)
        d = cdist(m.T, v, 'mahalanobis', VI=np.diag(self.w)).T
        u = d**(-2/(self.q-1))
        u /= u.sum(axis=0)
        self.u = u
        return self.u

    def updateV(self, m, u, eta):
        n_p = self.n_params
        m = m.reshape(n_p, -1)

        uq = u**self.q
        vtop = eta*self.t+uq.dot(m.T)
        vbot = uq.sum(axis=1)+eta
        self.v = (vtop.T/vbot).T
        self.eta = eta
        return self.v

    def update_internals(self):
        self.uq = self.u**self.q
        self.usum = np.sum(self.uq, axis=0)
        self.vuq = self.v.T.dot(self.uq)
        self._H = np.outer(self.w, self.usum).reshape(-1)

    def f(self, m):
        n_p = self.n_params
        m = m.reshape(n_p, -1)
        d = cdist(m.T, self.v, 'mahalanobis', VI=np.diag(self.w)).T
        d *= d
        uq = self.uq
        FCM1 = np.sum(uq*d)
        FCM2 = np.sum(self.eta*(self.v-self.t)**2)
        FCM = FCM1+FCM2
        return FCM

    def d(self, m):
        n_p = self.n_params
        m = m.reshape(n_p, -1)
        vuq = self.vuq
        usum = self.usum
        dFCM = m*usum - vuq
        dFCM = (dFCM.T*self.w).T
        return dFCM.reshape(-1)

    def get_H(self):
        return np.diag(self._H)

    def H(self, x):
        return self._H*x

    def update(self, x):
        # update v then u
        self.updateV(x, self.u, self.eta)
        self.updateU(x, self.v)
        self.update_internals()