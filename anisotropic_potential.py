import numpy as np
from numpy import fft
from scipy.interpolate import RectBivariateSpline as Interpolator

class Halfspace:
    def __init__(self, sig1, sig2, sig3, alpha, beta, gamma):
        """
        sig1, sig2, sig3, alpha, beta, gamma
        """

        self._sig1 = sig1
        self._sig2 = sig2
        self._sig3 = sig3

        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        sinb = np.sin(beta)
        cosb = np.cos(beta)
        sing = np.sin(gamma)
        cosg = np.cos(gamma)

        sxx = (sig1*cosa**2*cosb**2
               + sig2*(sina*cosg-sinb*sing*cosa)**2
               + sig3*(sina*sing+sinb*cosa*cosg)**2)

        sxy = (sig1*sina*cosa*cosb**2
               - sig2*(sina*cosg-sinb*sing*cosa)*(sina*sinb*sing+cosa*cosg)
               + sig3*(sina*sing+sinb*cosa*cosg)*(sina*sinb*cosg-sing*cosa))

        sxz = (-sig1*sinb*cosa
               - sig2*(sina*cosg-sinb*sing*cosa)*sing
               + sig3*(sina*sing+sinb*cosa*cosg)*cosg)*cosb

        syy = (sig1*sina**2*cosb**2
               + sig2*(sina*sinb*sing+cosa*cosg)**2
               + sig3*(sina*sinb*cosg-sing*cosa)**2)

        syz = (-sig1*sina*sinb
               + sig2*(sina*sinb*sing+cosa*cosg)*sing
               + sig3*(sina*sinb*cosg-sing*cosa)*cosg)*cosb

        szz = (sig1*sinb**2
               + sig2*sing**2*cosb**2
               + sig3*cosb**2*cosg**2)

        self._sxx = sxx
        self._sxy = sxy
        self._sxz = sxz
        self._syy = syy
        self._syz = syz
        self._szz = szz

    def phi(self, x, y, z=None):
        """Potential at x, y, z due to unit current source at the origin."""
        x = np.array(x)
        shape = x.shape

        x = x.reshape(-1)
        y = np.array(y).reshape(-1)
        if z is None:
            z = np.zeros_like(x)
        else:
            z = np.array(z).reshape(-1)
        assert(len(x) == len(y) and len(y) == len(z))

        sig = np.array([[self._sxx, self._sxy, self._sxz],
                        [self._sxy, self._syy, self._syz],
                        [self._sxz, self._syz, self._szz]])
        rho = np.linalg.inv(sig)

        scale = 1/(2*np.pi*np.sqrt(self._sig1*self._sig2*self._sig3))
        xyzRxyz = np.sqrt(rho[0, 0]*x*x + rho[1, 1]*y*y + rho[2, 2]*z*z +
                          2*(rho[0, 1]*x*y + rho[0, 2]*x*z + rho[1, 2]*y*z))

        phi = scale/xyzRxyz
        phi = phi.reshape(shape)
        return phi

    def voltage(self, A, B, M, N, current=1):
        """
        Voltage difference at M & N, due to current sources at A and B.
        A : Tuple of numpy arrays for x and y position of current source
        B : Tuple of numpy arrays for x and y position of current sink
        M : Tuple of numpy arrays for x and y position of positive electrode
        N : Tuple of numpy arrays for x and y position of negative electrode
        """
        MAx = M[0]-A[0]
        MAy = M[1]-A[1]
        NAx = N[0]-A[0]
        NAy = N[1]-A[1]

        MBx = M[0]-B[0]
        MBy = M[1]-B[1]
        NBx = N[0]-B[0]
        NBy = N[1]-B[1]

        MA_potential = self.phi(MAy, MAx)
        MB_potential = self.phi(MBy, MBx)
        NA_potential = self.phi(NAy, NAx)
        NB_potential = self.phi(NBy, NBx)
        volts = MA_potential-NA_potential-MB_potential+NB_potential
        return volts*current

class LayeredHalfspace:
    def __init__(self, z, sig1, sig2, sig3, alpha, beta, gamma):
        """
        z, sig1, sig2, sig3, alpha, beta, gamma.
        Bam!
        """
        z = np.array(z)
        self._z = z
        self._h = z[1:]-z[:-1]
        assert np.all(self._h >= 0)
        n_layers = len(z)
        self.n_layers = n_layers

        sig1 = np.array(sig1)
        sig2 = np.array(sig2)
        sig3 = np.array(sig3)
        alpha = np.array(alpha)
        beta = np.array(beta)
        gamma = np.array(gamma)

        self._sig1 = sig1
        self._sig2 = sig2
        self._sig3 = sig3
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

        assert len(sig1) == n_layers
        assert len(sig2) == n_layers
        assert len(sig3) == n_layers
        assert len(alpha) == n_layers
        assert len(beta) == n_layers
        assert len(gamma) == n_layers

        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        sinb = np.sin(beta)
        cosb = np.cos(beta)
        sing = np.sin(gamma)
        cosg = np.cos(gamma)

        sxx = (sig1*cosa**2*cosb**2
               + sig2*(sina*cosg-sinb*sing*cosa)**2
               + sig3*(sina*sing+sinb*cosa*cosg)**2)

        sxy = (sig1*sina*cosa*cosb**2
               - sig2*(sina*cosg-sinb*sing*cosa)*(sina*sinb*sing+cosa*cosg)
               + sig3*(sina*sing+sinb*cosa*cosg)*(sina*sinb*cosg-sing*cosa))

        sxz = (-sig1*sinb*cosa
               - sig2*(sina*cosg-sinb*sing*cosa)*sing
               + sig3*(sina*sing+sinb*cosa*cosg)*cosg)*cosb

        syy = (sig1*sina**2*cosb**2
               + sig2*(sina*sinb*sing+cosa*cosg)**2
               + sig3*(sina*sinb*cosg-sing*cosa)**2)

        syz = (-sig1*sina*sinb
               + sig2*(sina*sinb*sing+cosa*cosg)*sing
               + sig3*(sina*sinb*cosg-sing*cosa)*cosg)*cosb

        szz = (sig1*sinb**2
               + sig2*sing**2*cosb**2
               + sig3*cosb**2*cosg**2)

        self._sxx = sxx
        self._sxy = sxy
        self._sxz = sxz
        self._syy = syy
        self._syz = syz
        self._szz = szz

    def phi_tilde(self, wx, wy):
        """
        Calculates phi_tilde for unitary source at the surface.
        wx, wy
        """
        ix0 = np.where(wx == 0)[0]
        iy0 = np.where(wy == 0)[0]
        wx, wy = np.meshgrid(wx, wy)
        wx[ix0, iy0] = 1.0
        wy[ix0, iy0] = 1.0

        sxx = self._sxx
        sxy = self._sxy
        sxz = self._sxz
        syy = self._syy
        syz = self._syz
        szz = self._szz
        h = self._h
        N = self.n_layers

        a = szz[-1]
        b = sxz[-1]*wx+syz[-1]*wy
        c = sxx[-1]*wx*wx+2*sxy[-1]*wx*wy+syy[-1]*wy*wy
        al = np.sqrt(a*c-b*b)
        Q = 1/(al)
        for i in range(N-2, -1, -1):
            a = szz[i]
            b = sxz[i]*wx+syz[i]*wy
            c = sxx[i]*wx*wx+2*sxy[i]*wx*wy+syy[i]*wy*wy
            al = np.sqrt(a*c-b*b)
            l = al/a
            tanh = np.tanh(h[i]*l)
            alQ = al*Q
            Q = (alQ+tanh)/(al*(alQ*tanh+1))

        Q[ix0, iy0] = 0
        return Q/(2*np.pi)

    def phi_tilde_J(self, wx, wy):
        """
        Calculates phi_tilde for unitary source at the surface.
        wx, wy
        """
        ix0 = np.where(wx == 0)[0]
        iy0 = np.where(wy == 0)[0]
        wx, wy = np.meshgrid(wx, wy)
        wx[ix0, iy0] = 1.0
        wy[ix0, iy0] = 1.0

        sxx = self._sxx
        sxy = self._sxy
        sxz = self._sxz
        syy = self._syy
        syz = self._syz
        szz = self._szz
        h = self._h
        N = self.n_layers

        a = szz[:, None, None]
        b = sxz[:, None, None] * wx + syz[:, None, None] * wy
        c = sxx[:, None, None] * wx * wx + 2 * sxy[:, None, None] * wx * wy + syy[:, None, None] * wy * wy

        al = np.sqrt(a * c - b * b)
        l = (al/a)[:-1]
        tanh = np.tanh(h[:, None, None] * l)

        Qs = np.empty((N, *wx.shape))
        Qs[-1] = 1/al[-1]
        for i in range(N-2, -1, -1):
            top = al[i] * Qs[i+1] + tanh[i]
            bot = al[i] * ( al[i] * Qs[i+1] * tanh[i] + 1)
            Qs[i] = top / bot

        Qs[0, ix0, iy0] = 0

        d_a = np.empty((N, *wx.shape))
        d_b = np.empty((N, *wx.shape))
        d_c = np.empty((N, *wx.shape))
        d_h = np.empty((N-1, *wx.shape))
        
        gout = 1.0
        gQ = 1.0/(2*np.pi)*gout
        for i in range(N-1):
            top = al[i] * Qs[i+1] + tanh[i]
            bot = al[i] * ( al[i] * Qs[i+1] * tanh[i] + 1)

            gtop = 1.0/bot * gQ
            gbot = -Qs[i] / bot * gQ
            
            gal = Qs[i+1] * gtop + (2 * al[i] * Qs[i+1] * tanh[i] + 1)*gbot
            gtanh = 1.0 * gtop + al[i] * al[i] * Qs[i+1] * gbot
            gQ = al[i] * gtop + al[i] * al[i] * tanh[i] * gbot
            
            # tanh = tanh(h * l)
            front = (1 - tanh[i] * tanh[i])
            d_h[i] = front * l[i] * gtanh
            gl = front * h[i] * gtanh
            
            # l = al/a
            gal += gl / a[i]
            d_a[i] = -gl * al[i] / (a[i] * a[i])
            
            # al = np.sqrt(a * c - b * b)
            d_a[i] += c[i] / (2 * al[i]) * gal
            d_b[i] = - b[i] / al[i] * gal
            d_c[i] = a[i] / (2 * al[i]) * gal

        # Qs[-1] = 1/al[-1]
        gal = -1/(al[-1] * al[-1]) * gQ
        d_a[-1] = c[-1] / (2 * al[-1]) * gal
        d_b[-1] = - b[-1] / al[-1] * gal
        d_c[-1] = a[-1] / (2 * al[-1]) * gal
        
        d_h[:, ix0, iy0] = 0.0
        d_a[:, ix0, iy0] = 0.0
        d_b[:, ix0, iy0] = 0.0
        d_c[:, ix0, iy0] = 0.0
        
        d_szz = d_a
        d_sxz = wx * d_b
        d_syz = wy * d_b
        d_sxx = wx * wx * d_c
        d_sxy = 2 * wx * wy * d_c
        d_syy = wy * wy * d_c
        return d_sxx, d_sxy, d_sxz, d_syy, d_syz, d_szz, d_h

    def J_params(self, Jsxx, Jsxy, Jsxz, Jsyy, Jsyz, Jszz, transpose=False):
        s1 = self._sig1
        s2 = self._sig2
        s3 = self._sig3
        cosa = np.cos(self._alpha)
        cosb = np.cos(self._beta)
        cosg = np.cos(self._gamma)
        sina = np.sin(self._alpha)
        sinb = np.sin(self._beta)
        sing = np.sin(self._gamma)
        if transpose:
            Jsxx = Jsxx.T
            Jsxy = Jsxy.T
            Jsxz = Jsxz.T
            Jsyy = Jsyy.T
            Jsyz = Jsyz.T
            Jszz = Jszz.T

        # DsxxD_params
        Js1 = Jsxx*(cosa**2*cosb**2)
        Js2 = Jsxx*(sina*cosg - sinb*sing*cosa)**2
        Js3 = Jsxx*(sina*sing + sinb*cosa*cosg)**2
        Ja = Jsxx*(2*(-s1*sina*cosa*cosb**2
                + s2*(sina*cosg - sinb*sing*cosa)*(sina*sinb*sing + cosa*cosg)
                + s3*(sina*sing + sinb*cosa*cosg)*(sing*cosa - sina*sinb*cosg)))
        Jb = Jsxx*(2*(-s1*sinb*cosa**2*cosb
                - s2*(sina*cosg - sinb*sing*cosa)*sing*cosa*cosb
                + s3*(sina*sing + sinb*cosa*cosg)*cosa*cosb*cosg))
        Jg = Jsxx*(2*(s2*(sina*sing + sinb*cosa*cosg)*(sinb*sing*cosa - sina*cosg)
                + s3*(sina*sing + sinb*cosa*cosg)*(sina*cosg - sinb*sing*cosa)))

        # DsxyD_params
        Js1 += Jsxy*(sina*cosa*cosb**2)
        Js2 += Jsxy*(-(sina*cosg-sinb*sing*cosa)*(sina*sinb*sing+cosa*cosg))
        Js3 += Jsxy*((sina*sing+sinb*cosa*cosg)*(sina*sinb*cosg-sing*cosa))
        Ja += Jsxy*(-s1*sina**2*cosb**2
               + s1*cosa**2*cosb**2
               - s2*(sinb*sing*cosa - sina*cosg)*(sina*cosg - sinb*sing*cosa)
               - s2*(sina*sinb*sing + cosa*cosg)**2
               + s3*(sina*sing + sinb*cosa*cosg)**2
               + s3*(sing*cosa - sina*sinb*cosg)*(sina*sinb*cosg - sing*cosa))
        Jb += Jsxy*(-2*s1*sina*sinb*cosa*cosb
               - s2*(sina*cosg - sinb*sing*cosa)*sina*sing*cosb
               + s2*(sina*sinb*sing + cosa*cosg)*sing*cosa*cosb
               + s3*(sina*sing + sinb*cosa*cosg)*sina*cosb*cosg
               + s3*(sina*sinb*cosg - sing*cosa)*cosa*cosb*cosg)
        Jg += Jsxy*(s2*(sina*sing + sinb*cosa*cosg)*(sina*sinb*sing + cosa*cosg)
               - s2*(sina*cosg - sinb*sing*cosa)*(sina*sinb*cosg - sing*cosa)
               - s3*(sina*sing + sinb*cosa*cosg)*(sina*sinb*sing + cosa*cosg)
               + s3*(sina*cosg - sinb*sing*cosa)*(sina*sinb*cosg - sing*cosa))

        # DsxzD_params
        Js1 += Jsxz*(-sinb*cosa*cosb)
        Js2 += Jsxz*((sinb*sing*cosa - sina*cosg)*sing*cosb)
        Js3 += Jsxz*((sina*sing + sinb*cosa*cosg)*cosb*cosg)
        Ja += Jsxz*((s1*sina*sinb
               - s2*(sina*sinb*sing + cosa*cosg)*sing
               + s3*(-sina*sinb*cosg + sing*cosa)*cosg)*cosb)
        Jb += Jsxz*(((s1*sinb*cosa
                + s2*(sina*cosg - sinb*sing*cosa)*sing
                - s3*(sina*sing + sinb*cosa*cosg)*cosg)*sinb
               + (-s1*cosa*cosb
                  + s2*sing**2*cosa*cosb
                  + s3*cosa*cosb*cosg**2)*cosb))
        Jg += Jsxz*((-s2*(-sina*sing - sinb*cosa*cosg)*sing
               - s2*(sina*cosg - sinb*sing*cosa)*cosg
               - s3*(sina*sing + sinb*cosa*cosg)*sing
               + s3*(sina*cosg - sinb*sing*cosa)*cosg)*cosb)

        # DsyyD_params
        Js1 += Jsyy*(sina**2*cosb**2)
        Js2 += Jsyy*((sina*sinb*sing + cosa*cosg)**2)
        Js3 += Jsyy*((sina*sinb*cosg - sing*cosa)**2)
        Ja += Jsyy*(2*(s1*sina*cosa*cosb**2
                 + s2*(sinb*sing*cosa - sina*cosg)*(sina*sinb*sing + cosa*cosg)
                 + s3*(sina*sing + sinb*cosa*cosg)*(sina*sinb*cosg - sing*cosa)))
        Jb += Jsyy*(2*(-s1*sina**2*sinb*cosb
                 + s2*(sina*sinb*sing + cosa*cosg)*sina*sing*cosb
                 + s3*(sina*sinb*cosg - sing*cosa)*sina*cosb*cosg))
        Jg += Jsyy*(2*(s2*(sina*sinb*sing + cosa*cosg)*(sina*sinb*cosg - sing*cosa)
                 - s3*(sina*sinb*sing + cosa*cosg)*(sina*sinb*cosg - sing*cosa)))

        # DsyzD_params
        Js1 += Jsyz*(-sina*sinb*cosb)
        Js2 += Jsyz*((sina*sinb*sing + cosa*cosg)*sing*cosb)
        Js3 += Jsyz*((sina*sinb*cosg - sing*cosa)*cosb*cosg)
        Ja += Jsyz*((- s1*sinb*cosa
               + s2*(sinb*sing*cosa - sina*cosg)*sing
               + s3*(sina*sing + sinb*cosa*cosg)*cosg)*cosb)
        Jb += Jsyz*((-(-s1*sina*sinb
                 + s2*(sina*sinb*sing + cosa*cosg)*sing
                 + s3*(sina*sinb*cosg - sing*cosa)*cosg)*sinb
               + (-s1*sina*cosb
                  + s2*sina*sing**2*cosb
                  + s3*sina*cosb*cosg**2)*cosb))
        Jg += Jsyz*((s2*(sina*sinb*sing + cosa*cosg)*cosg
               + s2*(sina*sinb*cosg - sing*cosa)*sing
               - s3*(sina*sinb*sing + cosa*cosg)*cosg
               - s3*(sina*sinb*cosg - sing*cosa)*sing)*cosb)

        #DszzD_params
        Js1 += Jszz*(sinb**2)
        Js2 += Jszz*(sing**2*cosb**2)
        Js3 += Jszz*(cosb**2*cosg**2)
        # Ja += np.zeros_like(Jszz.T)
        Jb += Jszz*(-2*(- s1*sinb*cosb
                  + s2*sinb*sing**2*cosb
                  + s3*sinb*cosb*cosg**2))
        Jg += Jszz*(2*(s2*sing*cosb**2*cosg
                 - s3*sing*cosb**2*cosg))

        if transpose:
            Js1 = Js1.T
            Js2 = Js2.T
            Js3 = Js3.T
            Ja = Ja.T
            Jb = Jb.T
            Jg = Jg.T
        return Js1, Js2, Js3, Ja, Jb, Jg

    def phi(self, nx=128, x_width=128, ny=None, y_width=None, return_grid=False):
        if(ny is None):
            ny = nx
        if(y_width is None):
            y_width = x_width

        dx = x_width/(nx-1)
        dy = y_width/(ny-1)
        wx = 2*np.pi*(fft.fftfreq(nx, dx))
        wy = 2*np.pi*(fft.fftfreq(ny, dy))
        dwx = wx[1]-wx[0]
        dwy = wy[1]-wy[0]

        phi_tilde = self.phi_tilde(wx, wy)
        scale = len(wx)*len(wy)*dwx*dwy/(2*np.pi)
        phi = fft.fftshift(fft.irfft2(phi_tilde, s=phi_tilde.shape))*scale
        if return_grid:
            x = np.linspace(-x_width/2, x_width/2, nx)
            y = np.linspace(-y_width/2, y_width/2, ny)
            return phi, x, y
        return phi

    def phi_ds(self, nx=128, x_width=128, ny=None, y_width=None, return_grid=False):
        if(ny is None):
            ny = nx
        if(y_width is None):
            y_width = x_width

        dx = x_width/(nx-1)
        dy = y_width/(ny-1)
        wx = 2*np.pi*(fft.fftfreq(nx, dx))
        wy = 2*np.pi*(fft.fftfreq(ny, dy))
        dwx = wx[1]-wx[0]
        dwy = wy[1]-wy[0]

        Js = self.phi_tilde_J(wx, wy)
        scale = len(wx)*len(wy)*dwx*dwy/(2*np.pi)

        Jsxx = fft.fftshift(fft.irfft2(Js[0], s=Js[0].shape[1:]), axes=(-2,-1))*scale
        Jsxy = fft.fftshift(fft.irfft2(Js[1], s=Js[1].shape[1:]), axes=(-2,-1))*scale
        Jsxz = fft.fftshift(fft.irfft2(Js[2], s=Js[2].shape[1:]), axes=(-2,-1))*scale
        Jsyy = fft.fftshift(fft.irfft2(Js[3], s=Js[3].shape[1:]), axes=(-2,-1))*scale
        Jsyz = fft.fftshift(fft.irfft2(Js[4], s=Js[4].shape[1:]), axes=(-2,-1))*scale
        Jszz = fft.fftshift(fft.irfft2(Js[5], s=Js[5].shape[1:]), axes=(-2,-1))*scale
        if return_grid:
            x = np.linspace(-x_width/2, x_width/2, nx)
            y = np.linspace(-y_width/2, y_width/2, ny)
            return Jsxx, Jsxy, Jsxz, Jsyy, Jsyz, Jszz, x, y
        return Jsxx, Jsxy, Jsxz, Jsyy, Jsyz, Jszz

    def phi_dh(self, nx=128, x_width=128, ny=None, y_width=None, return_grid=False):
        if(ny is None):
            ny = nx
        if(y_width is None):
            y_width = x_width

        dx = x_width/(nx-1)
        dy = y_width/(ny-1)
        wx = 2*np.pi*(fft.fftfreq(nx, dx))
        wy = 2*np.pi*(fft.fftfreq(ny, dy))
        dwx = wx[1]-wx[0]
        dwy = wy[1]-wy[0]

        Jh = self.phi_tilde_J(wx, wy)[-1]
        scale = len(wx)*len(wy)*dwx*dwy/(2*np.pi)
        Jh = fft.fftshift(fft.irfft2(Jh, s=Jh.shape[1:]), axes=(-2, -1))*scale
        if return_grid:
            x = np.linspace(-x_width/2, x_width/2, nx)
            y = np.linspace(-y_width/2, y_width/2, ny)
            return Jh, x, y
        return Jh

    def voltage(self, A, B, M, N, current=1):
        """
        Voltage measured at M and N, due to current sources at A and B.
        A : Tuple of numpy arrays for x and y position of current source
        B : Tuple of numpy arrays for x and y position of current sink
        M : Tuple of numpy arrays for x and y position of positive electrode
        N : Tuple of numpy arrays for x and y position of negative electrode
        """
        MAx = M[0]-A[0]
        MAy = M[1]-A[1]
        NAx = N[0]-A[0]
        NAy = N[1]-A[1]

        MBx = M[0]-B[0]
        MBy = M[1]-B[1]
        NBx = N[0]-B[0]
        NBy = N[1]-B[1]

        MA_potential = np.zeros_like(MAx)
        MB_potential = np.zeros_like(MAx)
        NA_potential = np.zeros_like(MAx)
        NB_potential = np.zeros_like(MAx)
        """
        x_width = max(np.abs(MAx).max(), np.abs(NAx).max(),
                      np.abs(MBx).max(), np.abs(NBx).max())
        y_width = max(np.abs(MAy).max(), np.abs(NAy).max(),
                      np.abs(MBy).max(), np.abs(NBy).max())
        width = max(x_width, y_width)*4
        phi, x, y = self.phi(nx=4096, x_width=width, return_grid=True)
        interp = Interpolator(x, y, phi)
        MA_potential = interp(MAx, MAy, grid=False)
        NA_potential = interp(NAx, NAy, grid=False)
        MB_potential = interp(MBx, MBy, grid=False)
        NB_potential = interp(NBx, NBy, grid=False)
        """
        n_obs = len(M[0])
        for i in range(n_obs):
            x_width = max(np.abs(MAx[i]), np.abs(NAx[i]),
                          np.abs(MBx[i]), np.abs(NBx[i]))
            y_width = max(np.abs(MAy[i]), np.abs(NAy[i]),
                          np.abs(MBy[i]), np.abs(NBy[i]))
            width = max(x_width, y_width)*5
            phi, x, y = self.phi(nx=128, x_width=width, return_grid=True)

            wsMA = sinc_weights(MAx[i], MAy[i], x, y)
            wsNA = sinc_weights(NAx[i], NAy[i], x, y)
            wsMB = sinc_weights(MBx[i], MBy[i], x, y)
            wsNB = sinc_weights(NBx[i], NBy[i], x, y)
            MA_potential[i] = np.sum(wsMA*phi)
            NA_potential[i] = np.sum(wsNA*phi)
            MB_potential[i] = np.sum(wsMB*phi)
            NB_potential[i] = np.sum(wsNB*phi)
            """
            ixMA, iyMA, wsMA = get_weights(MAx[i], MAy[i], x, y)
            ixNA, iyNA, wsNA = get_weights(NAx[i], NAy[i], x, y)
            ixMB, iyMB, wsMB = get_weights(MBx[i], MBy[i], x, y)
            ixNB, iyNB, wsNB = get_weights(NBx[i], NBy[i], x, y)
            # interp = Interpolator(x, y, phi)
            MA_potential[i] = wsMA.dot(phi[ixMA, iyMA])
            NA_potential[i] = wsNA.dot(phi[ixNA, iyNA])
            MB_potential[i] = wsMB.dot(phi[ixNA, iyNA])
            NB_potential[i] = wsNB.dot(phi[ixNB, iyNB])
            interp = Interpolator(x, y, phi)
            MA_potential[i] = interp(MAx[i], MAy[i], grid=False)
            NA_potential[i] = interp(NAx[i], NAy[i], grid=False)
            MB_potential[i] = interp(MBx[i], MBy[i], grid=False)
            NB_potential[i] = interp(NBx[i], NBy[i], grid=False)
            """
        volts = MA_potential-NA_potential-MB_potential+NB_potential
        return volts*current

    def voltage_ds(self, A, B, M, N, current=1):
        """
        Voltage measured at M and N, due to current sources at A and B.
        A : Tuple of numpy arrays for x and y position of current source
        B : Tuple of numpy arrays for x and y position of current sink
        M : Tuple of numpy arrays for x and y position of positive electrode
        N : Tuple of numpy arrays for x and y position of negative electrode
        """
        MAx = M[0]-A[0]
        MAy = M[1]-A[1]
        NAx = N[0]-A[0]
        NAy = N[1]-A[1]

        MBx = M[0]-B[0]
        MBy = M[1]-B[1]
        NBx = N[0]-B[0]
        NBy = N[1]-B[1]

        n_obs = len(M[0])
        n_layers = self.n_layers
        Jxx = np.empty((n_obs, n_layers))
        Jxy = np.empty(Jxx.shape)
        Jxz = np.empty(Jxx.shape)
        Jyy = np.empty(Jxx.shape)
        Jyz = np.empty(Jxx.shape)
        Jzz = np.empty(Jxx.shape)
        for i in range(n_obs):
            x_width = max(np.abs(MAx[i]), np.abs(NAx[i]),
                          np.abs(MBx[i]), np.abs(NBx[i]))
            y_width = max(np.abs(MAy[i]), np.abs(NAy[i]),
                          np.abs(MBy[i]), np.abs(NBy[i]))
            width = max(x_width, y_width)*5
            out = self.phi_ds(nx=128, x_width=width, return_grid=True)
            Jsxx, Jsxy, Jsxz, Jsyy, Jsyz, Jszz, x, y = out

            wsMA = sinc_weights(MAx[i], MAy[i], x, y)
            wsNA = sinc_weights(NAx[i], NAy[i], x, y)
            wsMB = sinc_weights(MBx[i], MBy[i], x, y)
            wsNB = sinc_weights(NBx[i], NBy[i], x, y)

            MA_potential = np.sum(Jsxx*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jsxx*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jsxx*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jsxx*wsNB, axis=(-2, -1))
            Jxx[i] = MA_potential-NA_potential-MB_potential+NB_potential

            MA_potential = np.sum(Jsxy*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jsxy*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jsxy*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jsxy*wsNB, axis=(-2, -1))
            Jxy[i] = MA_potential-NA_potential-MB_potential+NB_potential

            MA_potential = np.sum(Jsxz*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jsxz*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jsxz*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jsxz*wsNB, axis=(-2, -1))
            Jxz[i] = MA_potential-NA_potential-MB_potential+NB_potential

            MA_potential = np.sum(Jsyy*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jsyy*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jsyy*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jsyy*wsNB, axis=(-2, -1))
            Jyy[i] = MA_potential-NA_potential-MB_potential+NB_potential

            MA_potential = np.sum(Jsyz*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jsyz*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jsyz*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jsyz*wsNB, axis=(-2, -1))
            Jyz[i] = MA_potential-NA_potential-MB_potential+NB_potential

            MA_potential = np.sum(Jszz*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jszz*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jszz*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jszz*wsNB, axis=(-2, -1))
            Jzz[i] = MA_potential-NA_potential-MB_potential+NB_potential

        return Jxx, Jxy, Jxz, Jyy, Jyz, Jzz

    def voltage_dh(self, A, B, M, N, current=1):
        """
        Voltage measured at M and N, due to current sources at A and B.
        A : Tuple of numpy arrays for x and y position of current source
        B : Tuple of numpy arrays for x and y position of current sink
        M : Tuple of numpy arrays for x and y position of positive electrode
        N : Tuple of numpy arrays for x and y position of negative electrode
        """
        MAx = M[0]-A[0]
        MAy = M[1]-A[1]
        NAx = N[0]-A[0]
        NAy = N[1]-A[1]

        MBx = M[0]-B[0]
        MBy = M[1]-B[1]
        NBx = N[0]-B[0]
        NBy = N[1]-B[1]

        n_obs = len(M[0])
        n_layers = self.n_layers
        Jh = np.empty((n_obs, n_layers-1))
        for i in range(n_obs):
            x_width = max(np.abs(MAx[i]), np.abs(NAx[i]),
                          np.abs(MBx[i]), np.abs(NBx[i]))
            y_width = max(np.abs(MAy[i]), np.abs(NAy[i]),
                          np.abs(MBy[i]), np.abs(NBy[i]))
            width = max(x_width, y_width)*5
            Jhi, x, y = self.phi_dh(nx=128, x_width=width, return_grid=True)

            wsMA = sinc_weights(MAx[i], MAy[i], x, y)
            wsNA = sinc_weights(NAx[i], NAy[i], x, y)
            wsMB = sinc_weights(MBx[i], MBy[i], x, y)
            wsNB = sinc_weights(NBx[i], NBy[i], x, y)

            MA_potential = np.sum(Jhi*wsMA, axis=(-2, -1))
            NA_potential = np.sum(Jhi*wsNA, axis=(-2, -1))
            MB_potential = np.sum(Jhi*wsMB, axis=(-2, -1))
            NB_potential = np.sum(Jhi*wsNB, axis=(-2, -1))
            Jh[i] = MA_potential-NA_potential-MB_potential+NB_potential

        return Jh

def sinc_weights(xi, yi, x, y):
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    wsx = np.sinc((xi-x)/dx)
    wsy = np.sinc((yi-y)/dy)
    return np.outer(wsx, wsy)

def get_weights(xi, yi, x, y):
    # indices and weights for 2d bilinear interpolation.
    ix = np.searchsorted(x, xi)
    iy = np.searchsorted(y, yi)

    hx = (x[ix]-xi)/(x[ix]-x[ix-1])
    hy = (y[iy]-yi)/(y[iy]-y[iy-1])

    ixs = [ix-1, ix-1, ix, ix]
    iys = [iy-1, iy, iy-1, iy]
    ws = np.array([hx*hy, hx*(1-hy), (1-hx)*hy, (1-hx)*(1-hy)])

    return ixs, iys, ws
def apparent_conductivity(volts, A, B, M, N, current=1):
    """
    Calculates the apparent conductivity for a given array configuration.
    Bam!
    """
    R = (1/np.sqrt((M[0]-A[0])**2+(M[1]-A[1])**2)
         - 1/np.sqrt((M[0]-B[0])**2+(M[1]-B[1])**2)
         - 1/np.sqrt((N[0]-A[0])**2+(N[1]-A[1])**2)
         + 1/np.sqrt((N[0]-B[0])**2+(N[1]-B[1])**2))
    return current*R/(volts*2*np.pi)