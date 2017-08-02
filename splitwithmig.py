import numpy
import dadi

# This was the best demographic model from our study
# Parameters listed below (e.g. nuE)

def split_mig(params, ns, pts):
    nuE,nuBG,nuB,nuG,mE_BG,mBG_E,mE_G,mE_B,mB_G,mG_B,mG_E,mB_E,T1,T2 = params
    fixed_params=[None,None,None,None,0,None,None,None,None,None,None,None,None,None]
    xx = yy = dadi.Numerics.default_grid(pts)
    phi = dadi.PhiManip.phi_1D (xx)
    phi = dadi.PhiManip.phi_1D_to_2D(xx, phi)
    phi = dadi.Integration.two_pops(phi,xx,T1,nu1=nuE,nu2=nuBG,m12=mBG_E,m21=mE_BG)
    phi = dadi.PhiManip.phi_2D_to_3D_split_2(xx,phi)
    phi = dadi.Integration.three_pops(phi, xx, T2, nu1=nuE, nu2=nuB, nu3=nuG,m12=mB_E,m21=mE_B,m13=mG_E,m31=mE_G,m23=mG_B,m32=mB_G)    
    fs = dadi.Spectrum.from_phi(phi, ns, (xx,xx,xx))
    return fs