import numpy as np

# Effective anisotropy transform from s1, s2, s3, a, b, g, to se1, se2, szz, a
def eff_anis_trans(sig1, sig2, sig3, alpha, beta, gamma):
    # Calculate Sigma Tensor
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
    
    #Then calculate eigs of A_eff
    ss, ang = eigSym2x2(sxx - sxz**2/szz, sxy-sxz*syz/szz, syy - syz**2/szz)
    return ss[0], ss[1], szz, ang

def eigSym2x2(a, bc, d):
    #Convention for eigenvalue ordering, and angle is to return
    # the values which results in a rotation angle closest to 0
    s1 = np.empty(len(a))
    s2 = np.empty(len(a))
    ang = np.empty(len(a))
    
    inds = bc==0
    s1[inds] = a[inds]
    s2[inds] = d[inds]
    ang[inds] = 0
    
    As = np.c_[a[~inds], bc[~inds], bc[~inds], d[~inds]].reshape(len(a[~inds]), 2, 2)
    
    eigs, vs = np.linalg.eigh(As)
    s1[~inds] = eigs[:,0]
    s2[~inds] = eigs[:,1]
    ang[~inds] = np.arctan(vs[:,0,1]/vs[:,0,0])
    
    inds = ang>np.pi/4
    ang[inds] -= np.pi/2
    s1_c = s1[inds]
    s1[inds] = s2[inds]
    s2[inds] = s1_c
    
    inds = ang<-np.pi/4
    ang[inds] += np.pi/2
    s1_c = s1[inds]
    s1[inds] = s2[inds]
    s2[inds] = s1_c
    return (s1, s2), ang