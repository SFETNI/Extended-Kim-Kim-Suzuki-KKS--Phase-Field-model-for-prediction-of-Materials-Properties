import numpy as np
import itertools
from numba import jit  # use to speed up python
import pyfftw          # use for fast fourier transform
import time 
import math
import sys
from numpy import linalg as LA
from numpy import diff
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import datetime

@jit(nopython=True)
def micro_numba(Nx,Ny,Nz,dx,dy,dz,X0_mat,X0_pre,r,ox,oy,oz):
    """
    Microstructure initialization:
    inputs: the position and radius of sphere that represents the precipitate of 
    composition X0_pre for a matrix of compositon X0_mat
    outpus: value of the variable eta and X_mat for a given grid 

    """
    eta    = np.zeros((Nx,Ny,Nz))
    X      = np.ones((Nx,Ny,Nz))*X0_mat
    X_mat  = np.ones((Nx,Ny,Nz))*X0_mat
    X_pre  = np.zeros((Nx,Ny,Nz))    
    x      = np.arange(Nx)*dx
    y      = np.arange(Ny)*dy
    z      = np.arange(Nz)*dz
    
    for i_x in range(Nx):
        for i_y in range(Ny):
            for i_z in range(Nz):
                for i_cir in range(len(r)): 
                    norm=np.sqrt((ox[i_cir]-x[i_x])**2
                    +(oy[i_cir]-y[i_y])**2
                    +(oz[i_cir]-z[i_z])**2 )
                    #+(oz[i_cir]-zcor[i_coor])**2)
                    if norm<=r[i_cir]:
                        X_pre[i_x][i_y][i_z]=X0_pre # +=
                        X_mat[i_x][i_y][i_z]=0 # +=
                        X[i_x][i_y][i_z]    =X0_pre
                        eta[i_x][i_y][i_z]  =1

    return eta,X,X_mat,X_pre

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def put_nucle(idx,   # grid index where nucleus can be inserted 
              eta,   # order varaible  -> phase 
              X,     # conserved variable -> molar fraction of Si
              X_al,  # molar fraction of Si in al
              X_th,  # molar fraction of Si in diamond
              Nx,    # grid density in x direction 
              Ny,    # grid density in y direction 
              Nz     # # grid density in z direction  
              ):
   eta1d = eta.reshape(Nx*Ny*Nz)
   #X1d = X.reshape(Nx*Ny*Nz)
   #X_al1d = X_al.reshape(Nx*Ny*Nz)
   #X_th1d = X_th.reshape(Nx*Ny*Nz)
   #eta1dred = eta1d<0.05
   #idx = np.argwhere(eta1dred==True)
   rand = np.random.randint(0, len(idx)-1, size=1)
   eta1d[idx[rand]]=1
   eta = eta1d.reshape(Nx,Ny,Nz)
   #print("nucle : %i"%(idx[rand]))
   #X1d[idx[rand]]=0.99999
   #X = X1d.reshape(Nx,Ny,Nz)
   #X_al1d[idx[rand]]=0
   #X_th1d[idx[rand]]=0.99999
   #
   #X_al = X_al1d.reshape(Nx,Ny,Nz)
   #X_th = X_th1d.reshape(Nx,Ny,Nz)
   return eta,X,X_al,X_th

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def nucle(B,X_al_eq,t,dt,T,dx,dy,dz, Nx, Ny,Nz,D,
	A_alp,B_alp,C_alp,A_thet,B_thet,C_thet,
    sigma,X_al_mean,Xv
	):
   #B        # = heterogeneous nucleation factor [-]
   
 
   Na         = 6.02214086e23    # = Avogadro constant [mol-1]
   kB          = 1.380649e-23    # = Boltzmann constant [J.K-1]
   int_Jnuc = 0
   r_s = 0
   if X_al_mean>0.025:
      X_al_mean=0.025


   # Elastic constant
   C11_Si=elas_const(T=T,output="C11_Si")
   C12_Si=elas_const(T=T,output="C12_Si")
   C11_Al=elas_const(T=T,output="C11")
   C12_Al=elas_const(T=T,output="C12")
      
   # Poisson's ratio and moduli
   K_si = (C11_Si+2*C12_Si)/3
   mu_si = (C11_Si-C12_Si)/2
   K_al = (C11_Al+2*C12_Al)/3
   mu_al = (C11_Al-C12_Al)/2
   
   
   C6= (3*K_si)/(3*K_si+4*mu_al)
   
   # alpha-Al lattice with Si 
   Vm_fcc   = molar_volume(T0=298,T=T,X_Si=X_al_mean,output="Vmfcc")       # = molar volume [m3.mol-1]
   V_fcc    = Vm_fcc/Na                                               # = atomic volume [m3]
   nL_fcc   = 4
   a_fcc    = (nL_fcc*V_fcc)**(1/3)                                   # = lattice parameter [m]
   
   
   # d-Si lattice
   Vm_dia   = molar_volume(T0=298,T=T,X_Si=0.9999,output="Vmdia")     # = molar volume [m3.mol-1]
   V_dia    = Vm_dia/Na                                               # = atomic volume [m3]
   nL_dia   = 8
   
   # Molar volume of the domain
   
   Vm       = 0.1*Vm_dia+0.9*Vm_fcc                                   # = molar volume of domain
   
   # volume of the simulation
   if (Ny==1) and (Nz==1) :                   # 1D case 
      Vsim = ((Nx-1)*dx)**(3)
   elif (Nz==1):                              # 2D case 
      Vsim = ((Nx-1)*dx*(Ny-1)*dy)**(3/2)
   else:                                      # 3D case 
      Vsim = ((Nx-1)*dx*(Ny-1)*dy*(Nz-1)*dz) 
   
   # Nucleation free energy 
   f_al_p   = 2*A_alp*(X_al_mean)+ B_alp*(X_al_mean)
   f_th     = A_thet*0.99999**2+ B_thet*0.99999+C_thet
   f_al     = A_alp*X_al_mean**2+ B_alp*X_al_mean+C_alp
   
   delt_gV =(1/Na)*(f_th-(f_al+f_al_p*(0.99999-X_al_mean)))    # = driving force for nucleation
   
   A_fac_para = (36*np.pi*V_dia**2)**(1/3.)                    # = geometric factor for spherical precipitate
   
   
   eps = (1/3)*((Vm_dia-Vm_fcc)**2/(Vm_dia))/Na                # = elastic strain due to embedded incoherent 
   
   delgs = 2*mu_al*C6*eps                                       # = elastic energy  [J]
   
   
   deltaGhomo = (4/27)*((A_fac_para*sigma)**3/\
                           ((delt_gV+delgs)**2))         # = homogenous gibbs energy formation of stable precipitate 
                                                         #   [J]
   
   
   deltaGhete = B*deltaGhomo                            # heterogeneous nucleation
   
   
   
   n_s = (-2*A_fac_para*sigma/(3*(delt_gV+delgs)))**3    #= crital number of monomer in the nucleus 
   
   # = critical radius [m] 
   r_s = (1/(2*np.sqrt(np.pi)))*(36*np.pi)**(1/6)*\
                V_dia**(1/3)*n_s**(1/3)                      # = critical radius [m]   
   
   
   Zelch     = (3*(delt_gV+delgs)**2)/(4*np.sqrt(np.pi*kB*T)\
                  *(sigma*A_fac_para)**(3/2.))               # =  zelchovich factor [-]
   
   beta      =  (4*np.pi*r_s**2*X_al_mean*D)/(a_fcc**4)      # = atomic attachment rate [s-1] 
   
   tau = 1/(2*beta*Zelch**2)                                 # = incubation time [s]
   
   
   
   Jnuc = Zelch*Xv*beta*np.exp(-deltaGhete/(kB*T))*\
         np.exp(-tau/t)*(Na/Vm)*Vsim#*(0.5e-6)**3            # = nucleation rate [#.s-1]
   
   int_Jnuc = Jnuc*dt                                       # = nucleus increment [#]
   
   # Switch off nucleation if equilibrum 
   if X_al_mean<X_al_eq:
      int_Jnuc=0
      r_s     =0

   return int_Jnuc,r_s 
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
@jit(nopython=True)
def nucle_numba(Nx,Ny,Nz,dx,dy,dz,eta,X,X_mat,X_pre,r,ox,oy,oz):
    """
    put small nucleus inside the simulation domain
    inputs : same as  micro_numba
    outuput: same as  micro_numba
    """
      
    x      = np.arange(Nx)*dx
    y      = np.arange(Ny)*dy
    z      = np.arange(Nz)*dz
    
    for i_x in range(Nx):
        for i_y in range(Ny):
            for i_z in range(Nz):
                for i_cir in range(len(r)): 
                    norm=np.sqrt((ox[i_cir]-x[i_x])**2
                    +(oy[i_cir]-y[i_y])**2
                    +(oz[i_cir]-z[i_z])**2 )
                    #+(oz[i_cir]-zcor[i_coor])**2)
                    if norm<=r[i_cir]:
                        eta[i_x][i_y][i_z]  =1
                        X_pre[i_x][i_y][i_z]=0.99999 # +=
                        X_mat[i_x][i_y][i_z]=0 # +=
                        X[i_x][i_y][i_z]    =0.99999

    return eta,X,X_mat,X_pre


#@jit(nopython=True)
def compu_cell(Nx,Ny):
   """
   compute cells, needed to compute the surface of the precipitate and thus enthalpy of coarsening
   """
   gridlin = np.arange(0,Nx*Ny,1,dtype=int)#
   grid=np.resize(gridlin,(Nx,Ny))
   cell=[]
   for i in range(Nx-1):
      for j in range(Ny-1):
         cell.append([grid[i,j],grid[i+1,j],grid[i+1,j+1],grid[i,j+1]])
   cell = np.array(cell)
   return grid,cell


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def intersec_spher_plan(ox,oy,oz,r):
   """
   inputs :coordinates of sphere and the radius + coordinate of a plan
   outputs: intersection between the array of sphere and the plan  
   """
   r_new=[]
   ox_new=[]
   oy_new=[]
   oz_new =[]
   n=np.array([0,0,1])                               # = n normal vector to the plan P
   P = np.array([250,0,100])*1e-9                      # = point belonging to the plan P
   for i in range(len(r)):
      PO = np.array([ox[i]-P[0],oy[i]-P[1],oz[i]-P[2]])
      OH = (np.dot(PO,n)/np.linalg.norm(n))*n
      d= np.linalg.norm(OH)
      if d<r[i]:
         r_new.append(np.sqrt(r[i]**2-d**2))
         ox_new.append(-(OH[0]-ox[i]))
         oy_new.append(-(OH[1]-oy[i]))
         oz_new.append(-(OH[2]-oz[i]))
   return ox_new,oy_new,oz_new,r_new



def full_3x3_to_Voigt_6_index(i, j):
    """
    transform the index of 3x3 tension into voigt notation
    """
    if i == j:
        return i
    return 6-i-j

def fft_(a):
    """
    return a fft object from pyfftw library that will be use to compute fft
    """
    fft_object=pyfftw.builders.fftn(a,axes=(0,1,2), threads=12)
    return fft_object()

def ifft_(a):
    """
    return a inverse fft object from pyfftw library that will be use to compute inverse fft
    """
    ifft_object=pyfftw.builders.ifftn(a,axes=(0,1,2), threads=12)
    return ifft_object()

def elas_modul(C11, C12, C44):
    """
    Compute the compliance tensor of a material
    """
    C=np.array([[C11,C12,C12,  0,  0,  0],
                [C12,C11,C12,  0,  0,  0],
                [C12,C12,C11,  0,  0,  0],
                [  0,  0,  0,C44,  0,  0],
                [  0,  0,  0,  0,C44,  0],
                [  0,  0,  0,  0,  0,C44]])
    C = np.asarray(C)
    C_out = np.zeros((3,3,3,3), dtype=float)
    
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        Voigt_i = full_3x3_to_Voigt_6_index(i, j)
        Voigt_j = full_3x3_to_Voigt_6_index(k, l)
        C_out[i, j, k, l] = C[Voigt_i, Voigt_j]
    
    return C_out



#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def solve_elas(Nx,Ny,Nz,eta,Omega_khij,C_ijkl,kf,epsX,fft_object,ifft_object):
    """
    Solve elasticity problem and compute the elastic energy and derivative
    """
    # variable initialisation
    sigel_ij   = np.zeros((3,3,Nx,Ny,Nz))
    epsel_kl   = np.zeros((3,3,Nx,Ny,Nz))
    sigelk_ij  = np.zeros((3,3,Nx,Ny,Nz))*1j
    epsk_kl  = np.zeros((3,3,Nx,Ny,Nz))*1j
    eps_kl     = np.zeros((3,3,Nx,Ny,Nz))
    el_eta = np.zeros((Nx,Ny,Nz))
    #buff       = np.zeros((Nx,Ny,Nz))   
    
    h        = 3.*eta**2-2*eta**3
    h_eta    = 6.*eta   -6*eta**2 
    
    epsX_kl        = np.einsum('xyz,kl->klxyz',eta**2,np.eye(3))*epsX
    epsX_kl_eta    = np.einsum('xyz,kl->klxyz',2*eta,np.eye(3))*epsX
    #epsX_kl     = np.einsum('xyz,kl->klxyz',h,np.eye(3))*epsX
    
    epsel_kl   =eps_kl-epsX_kl
    
    sigel_ij=np.einsum('ijkl,klxyz->ijxyz',C_ijkl,epsel_kl)
    

    # calculate green's tensor

    sigelk_ij[0,0,:,:,:]=fft_object(sigel_ij[0,0,:,:,:])
    sigelk_ij[1,1,:,:,:]=fft_object(sigel_ij[1,1,:,:,:])
    sigelk_ij[2,2,:,:,:]=fft_object(sigel_ij[2,2,:,:,:])
    sigelk_ij[0,1,:,:,:]=fft_object(sigel_ij[0,1,:,:,:])
    sigelk_ij[1,0,:,:,:]=sigelk_ij[0,1,:,:,:]
    sigelk_ij[0,2,:,:,:]=fft_object(sigel_ij[0,2,:,:,:])
    sigelk_ij[2,0,:,:,:]=sigelk_ij[0,2,:,:,:]
    sigelk_ij[1,2,:,:,:]=fft_object(sigel_ij[1,2,:,:,:])
    sigelk_ij[2,1,:,:,:]=sigelk_ij[1,2,:,:,:]
    
    epsk_kl[0,0,:,:,:]=fft_object(eps_kl[0,0,:,:,:])
    epsk_kl[1,1,:,:,:]=fft_object(eps_kl[1,1,:,:,:])
    epsk_kl[2,2,:,:,:]=fft_object(eps_kl[2,2,:,:,:])
    epsk_kl[0,1,:,:,:]=fft_object(eps_kl[0,1,:,:,:])
    epsk_kl[1,0,:,:,:]=epsk_kl[0,1,:,:,:]
    epsk_kl[0,2,:,:,:]=fft_object(eps_kl[0,2,:,:,:])
    epsk_kl[2,0,:,:,:]=epsk_kl[0,2,:,:,:]
    epsk_kl[1,2,:,:,:]=fft_object(eps_kl[1,2,:,:,:])
    epsk_kl[2,1,:,:,:]=epsk_kl[1,2,:,:,:]
    
    epsk_kl=epsk_kl-\
    np.einsum('xyzklij,ijxyz->klxyz',Omega_khij,sigelk_ij)
    
    
    #inverse fourier transform
    eps_kl[0,0,:,:,:]=np.real(ifft_object(epsk_kl[0,0,:,:,:]))
    eps_kl[1,1,:,:,:]=np.real(ifft_object(epsk_kl[1,1,:,:,:]))
    eps_kl[2,2,:,:,:]=np.real(ifft_object(epsk_kl[2,2,:,:,:]))
    eps_kl[0,1,:,:,:]=np.real(ifft_object(epsk_kl[0,1,:,:,:]))
    eps_kl[1,0,:,:,:]=eps_kl[0,1,:,:,:]
    eps_kl[0,2,:,:,:]=np.real(ifft_object(epsk_kl[0,2,:,:,:]))
    eps_kl[2,0,:,:,:]=eps_kl[0,2,:,:,:]
    eps_kl[1,2,:,:,:]=np.real(ifft_object(epsk_kl[1,2,:,:,:]))
    eps_kl[2,1,:,:,:]=eps_kl[1,2,:,:,:]
    
    epsel_kl   =eps_kl-epsX_kl
    

    # last one 
    eps_ijkl= -np.einsum('ijxyz,klxyz->ijklxyz',eps_kl,epsX_kl_eta)\
              -np.einsum('ijxyz,klxyz->ijklxyz',epsX_kl_eta,eps_kl)+np.einsum('ijxyz,klxyz->ijklxyz',epsX_kl_eta,epsX_kl)\
              +np.einsum('ijxyz,klxyz->ijklxyz',epsX_kl,epsX_kl_eta)

    #epsel_kl_eta = eps_kl-np.einsum('ijkl,'epsX_kl*h_eta)

    sigel_ij=np.einsum('ijkl,klxyz->ijxyz',C_ijkl,epsel_kl)
    
    el    =0.5*np.einsum('ijxyz,ijxyz->xyz',sigel_ij,epsel_kl)
    el_eta=0.5*np.einsum('ijkl,ijklxyz->xyz',C_ijkl,eps_ijkl)

    return sigel_ij,epsel_kl, el_eta,el

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
@jit
def cal_det_comat(mat,opt):
    """
    compute the determinant and the comatrix of a 3x3 tensor
    """
    if opt=="2d":
        A=mat[0][0]
        B=mat[0][1]
        C=mat[1][0]
        D=mat[1][1]
        
        comat=np.array(
        [[D,-B],
         [-C,A]]
         )
        det=mat[0][0]*mat[1][1]-mat[1][0]*mat[0][1]
    if opt=="3d":
        comat=np.zeros((3,3))

        A=+(mat[1][1]*mat[2][2]-mat[2][1]*mat[1][2]) # comat[0][0]
        B=-(mat[1][0]*mat[2][2]-mat[2][0]*mat[1][2])
        C=+(mat[1][0]*mat[2][1]-mat[2][0]*mat[1][1])
    
        D=-(mat[0][1]*mat[2][2]-mat[2][1]*mat[0][2]) # comat[0][1]
        E=+(mat[0][0]*mat[2][2]-mat[2][0]*mat[0][2])
        F=-(mat[0][0]*mat[2][1]-mat[2][0]*mat[0][1])
    
        G=+(mat[0][1]*mat[1][2]-mat[1][1]*mat[0][2]) # comat[0][2]
        H=-(mat[0][0]*mat[1][2]-mat[1][0]*mat[0][2])
        I=+(mat[0][0]*mat[1][1]-mat[1][0]*mat[0][1])
    
        comat=np.array(
        [[A,D,G],
         [B,E,H],
         [C,F,I]]
         )
        det=mat[0][0]*A+mat[0][1]*B+mat[0][2]*C
    return det,comat

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
@jit(nopython=True)
def compute_green(Nx,Ny,Nz,C_ijkl,kf):
   """
   Compute green tensor 
   """
   # variable initialisation
   Omega_xyzklij = np.zeros((Nx,Ny,Nz,3,3,3,3))
   K_ik          = np.zeros((Nx,Ny,Nz,3,3)) 
   N_ki          = np.zeros((Nx,Ny,Nz,3,3))
   
   for i_x in range(Nx):
      for i_y in range(Ny):
         for i_z in range(Nz):
            for i in range(3):
               for j in range(3):
                  for k in range(3):
                     for l in range(3):
                        K_ik[i_x,i_y,i_z,i,k]+=C_ijkl[i,j,k,l]\
                           *kf[j,i_x,i_y,i_z]*kf[l,i_x,i_y,i_z]
               
            det,co_mat=cal_det_comat(K_ik[i_x,i_y,i_z,:,:],opt="3d")
               
            if det==0:
                det=1
               
            N_ki[i_x,i_y,i_z,:,:]=co_mat/det
               
            for k in range(3):
               for h in range(3):
                  for i in range(3):
                     for j in range(3):
                        Omega_xyzklij[i_x,i_y,i_z,k,h,i,j]+=0.25*(\
                        N_ki[i_x,i_y,i_z,h,i]*kf[j,i_x,i_y,i_z]*kf[k,i_x,i_y,i_z]+\
                        N_ki[i_x,i_y,i_z,k,i]*kf[j,i_x,i_y,i_z]*kf[h,i_x,i_y,i_z]+\
                        N_ki[i_x,i_y,i_z,h,j]*kf[i,i_x,i_y,i_z]*kf[k,i_x,i_y,i_z]+\
                        N_ki[i_x,i_y,i_z,k,j]*kf[i,i_x,i_y,i_z]*kf[h,i_x,i_y,i_z])
            
   return Omega_xyzklij



#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# good one 
@jit(nopython=True)
def prepar_fft(Nx,dx,Ny,dy,Nz,dz,opt): 
    """
    Compute spatial frequence term and derivative
    """
    # variable initialisation
    lin_x=np.zeros(Nx)
    lin_y=np.zeros(Ny)
    lin_z=np.zeros(Nz)
    
    k=np.zeros((3,Nx,Ny,Nz))
    k2=np.zeros((Nx,Ny,Nz))
    k4=np.zeros((Nx,Ny,Nz))
    
    
    if (Nx % 2) == 1 : # = number odd if remainers is one
        lin_x[:int((Nx-1)/2.0+1)]=np.arange(0, int((Nx-1)/2.0+1), 1)*2*np.pi/(Nx*dx)
        lin_x[int((Nx-1)/2.0+1):]=np.arange(int(-(Nx+1)/2.0 +1), 0, 1)*2*np.pi/(Nx*dx)
    if (Ny % 2) == 1 :
        lin_y[:int((Ny-1)/2.0+1)]=np.arange(0, int((Ny-1)/2.0+1), 1)*2*np.pi/(Ny*dy)
        lin_y[int((Ny-1)/2.0+1):]=np.arange(int(-(Ny+1)/2.0 +1), 0, 1)*2*np.pi/(Ny*dy)
    if (Nz % 2) == 1 :
        lin_z[:int((Nz-1)/2.0+1)]=np.arange(0, int((Nz-1)/2.0+1), 1)*2*np.pi/(Nz*dz)
        lin_z[int((Nz-1)/2.0+1):]=np.arange(int(-(Nz+1)/2.0 +1), 0, 1)*2*np.pi/(Nz*dz)
    if (Nx % 2) == 0 : # = number even if remainers is zero
        lin_x[0:int(Nx/2.0)]=np.arange(0, int(Nx/2.0), 1)*2*np.pi/(Nx*dx)
        lin_x[int(Nx/2.0 + 1):]=np.arange(int(-Nx/2.0 + 1), 0, 1)*2*np.pi/(Nx*dx)
    if (Ny % 2) == 0 :
        lin_y[0:int(Ny/2.0)]=np.arange(0, int(Ny/2.0), 1)*2*np.pi/(Ny*dy)
        lin_y[int(Ny/2.0 + 1):]=np.arange(int(-Ny/2.0 + 1), 0, 1)*2*np.pi/(Ny*dy)
    if (Nz % 2) == 0 :
        lin_z[0:int(Nz/2.0)]=np.arange(0, int(Nz/2.0), 1)*2*np.pi/(Nz*dz)
        lin_z[int(Nz/2.0 + 1):]=np.arange(int(-Nz/2.0 + 1), 0, 1)*2*np.pi/(Nz*dz)
    
    for i in range(Nx):
        for j in range(Ny):
            for l in range(Nz):
                k[0,i,j,l]= lin_x[i]
                k[1,i,j,l]= lin_y[j]
                k[2,i,j,l]= lin_z[l]                    
    

    k2=k[0]**2+k[1]**2+k[2]**2
    
    k4=k2**2
    
     
    return k,k2,k4


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def atomic_mobility(T,X_Si,Vm,opt):
    """
    Compute atomic mobility of Si atoms in Al 
    """
    out=0
    # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58â€“66.
    R     = 8.314472                  # = [J.K-1.mol-1]
    # mobility in the liquid
    liq_Phi_Al_Al   =-24830.3-130.62*T
    liq_Phi_Al_Si   = -41794.2-121.96*T
    liq_0Phi_Al_AlSi=17050.5
    X_Al = 1-X_Si
    # for Al
    liq_Phi_Al = X_Al*liq_Phi_Al_Al+X_Si*liq_Phi_Al_Si+\
        X_Al*X_Si*liq_0Phi_Al_AlSi
    liq_M_Al = np.exp(liq_Phi_Al/(R*T))*(1/(R*T))
    # mobility in FCC 
    # for Al

    Phi_Al_Al        = -123111.6-97.34*T
    Phi_Al_Si        = -155634.921-81.0257*T  
    _0Phi_Al_AlSi   =  1846710.24
    Phi_Si_Al      = -117600.0-93.05*T
    Phi_Si_Si      = -155634.9-81.03*T
    _0Phi_Si_AlSi  = -152613.88
    


    Phi_Al = X_Al*Phi_Al_Al+X_Si*Phi_Al_Si+\
        X_Si*X_Al*_0Phi_Al_AlSi#*(chi_Al_grid-chi_Si_grid)
    
    M_Al = np.exp(Phi_Al/(R*T))*(1/(R*T))
    D_Al_s = R*T*M_Al
    

    Phi_Si=X_Si*Phi_Si_Si+X_Al*Phi_Si_Al+\
        X_Si*X_Al*_0Phi_Si_AlSi#*(chi_Al_grid-chi_Si_grid)
    M_Si = np.exp(Phi_Si/(R*T))*(1/(R*T))
    mob = X_Al*X_Si*(X_Al*M_Si+X_Si*M_Al)

    return mob
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
@jit
def free_energy_para_old(c_alpha,c_theta,eta,T,R,C44,Omega0,w,
                     A_alp,B_alp,C_alp,A_thet,B_thet,C_thet):
    """
    Compute derivative of free energy function
    """
    # variable initialisation

    RT=R*T
    
    # chemical free energy 
    
    w_s=w/(C44*Omega0)
    
    fa    = A_alp*c_alpha**2+B_alp*c_alpha+C_alp  # = alpha phase 
    fa_s = fa/(C44*Omega0)                        # = its normalized value
    fa_ca = 2*A_alp*c_alpha+B_alp                 # = its first derivative  
    fa_ca_s = fa_ca/(C44*Omega0)                  # = its normalized value
    fa_caca = 2*A_alp
    fa_caca_s = fa_caca/(C44*Omega0)              # = its normalized value
    
    ft    = A_thet*c_theta**2+B_thet*c_theta+C_thet   # = theta' phase     
    ft_s  = ft/(C44*Omega0)                       # = its normalized value
    ft_ct = 2*A_thet*c_theta+B_thet                        # = first derivative
    ft_ct_s = ft_ct/(C44*Omega0)                       # = first derivative  
    ft_ctct = 2*A_thet
    ft_ctct_s = ft_ctct/(C44*Omega0)                  # = its normalized value
    
    # potential
    
    h     = 3.*eta**2-2*eta**3                    # = double-well potential
    h_eta = 6.*eta   -6*eta**2                    # = derivative of h with respect to eta
    
    g     = eta**2-2*eta**3+eta**4                # = monotonous function 
    g_eta = 2.*eta -6.*eta**2 + 4.*eta**3
    
    
    # smallest 
    
    G = (1-h)*fa+h*ft+w*g
    #G_eta_s = -h_eta*(fa_s-ft_s-(c_alpha-c_theta)*fa_ca_s)+w_s*g_eta  # Hu
    G_eta_s = -h_eta*(ft_s-fa_s+(c_alpha-c_theta)*fa_ca_s)+w_s*g_eta
    
    

    
    G_ceta_G_cc= h_eta*( c_alpha-c_theta)
    
    G_cc_s = (ft_ctct_s*fa_caca_s)/((1-h)*ft_ctct_s+h*fa_caca_s) # Hu
    
    #G_cc_s = h_eta* ft_ct* ((1-2*h)/(h*(1-h))) +w_s*g_eta
    
    return G_eta_s,G_cc_s,G_ceta_G_cc,G,fa_ca_s
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------    
def divergence(field):
    "return the divergence of a n-D field"
    return np.sum(np.gradient(field))

def partial(element, function):
	"""
	partial : sympy.core.symbol.Symbol * sympy.core.add.Add -> sympy.core.add.Add
	partial(element, function) Performs partial derivative of a function of several variables is its derivative with respect to one of those variables, with the others held constant. Return partial_diff.
	"""
	partial_diff = function.diff(element)

	return partial_diff


def gradient(partials):
	"""
	gradient : List[sympy.core.add.Add] -> numpy.matrix
	gradient(partials) Transforms a list of sympy objects into a numpy matrix. Return grad.
	"""
	grad = np.matrix([[partials[0]], [partials[1]]])

	return grad    
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def compute_residual_KKS(good_step,time_s_out,array_dtime_s,array_energy_s,X_mean_out,eta_mean_out,G_mean_out,array_fa_ca_s,Meta_s,array_G_s_adapt,T_out_s,array_D_adapt,\
    array_G_cc_adapt,array_G_ceta_adapt,\
    L_s,array_eta,array_X,array_X_al,array_X_th,\
    fa_ca_s,ft_ct_s,h_eta,c_alpha,c_theta,istep,dtime_s,Nx,Ny,Nz,dx,dx_s,dy_s,dz_s):     
    t_start_res=time.time()   
    #--------------------------------------------------------
    time_s_out=np.asarray(time_s_out[good_step:])
    X_mean_out=X_mean_out[good_step:]
    eta_mean_out=eta_mean_out[good_step:]
    G_mean_out=G_mean_out[good_step:]
    T_out_s=T_out_s[good_step:]
    array_energy_s=array_energy_s[good_step:istep]

    
    #--------------------------------------------------------
    eta_plus=array_eta[-1].reshape(Nx,Ny)
    eta_moins=array_eta[-2].reshape(Nx,Ny)
    X_plus=array_X[-1].reshape(Nx,Ny)
    X_moins=array_X[-2].reshape(Nx,Ny)
    c_alpha=c_alpha.reshape(Nx,Ny)
    c_theta=c_theta.reshape(Nx,Ny)
    #G_ceta_s=G_ceta_s.reshape(Nx,Ny)
    h_eta=h_eta.reshape(Nx,Ny)
    fa_ca_s=fa_ca_s.reshape(Nx,Ny)
    ft_ct_s=ft_ct_s.reshape(Nx,Ny)
    #Meta_s= D_s/G_cc_s # Hu 2007

    G_cc_s_plus=array_G_cc_adapt[-1].reshape(Nx,Ny) 
    G_cc_s_moins=array_G_cc_adapt[-2].reshape(Nx,Ny)    
    fa_ca_s_plus=array_fa_ca_s[-1].reshape(Nx,Ny)
    fa_ca_s_moins=array_fa_ca_s[-2].reshape(Nx,Ny)
    fa_ca_s_mean=np.asarray((fa_ca_s_plus+fa_ca_s_moins)/2)
    array_G_s_adapt=np.asarray(array_G_s_adapt).reshape(len(array_G_s_adapt),Nx,Ny)


    # gradient term--------------------------------------------------------
    #X_t= (X_plus-X_moins)/dtime_s  # first defintion
    """
    grad_mu_plus=np.asarray(np.gradient(np.asarray(fa_ca_s_plus)))  # good one
    grad_mu_moins=np.asarray(np.gradient(np.asarray(fa_ca_s_moins)))
    grad_mu=(grad_mu_plus+grad_mu_moins)/2
    Meta_s_plus= np.divide(array_D_adapt[-1] , G_cc_s_plus)
    Meta_s_moins=np.divide(array_D_adapt[-2] , G_cc_s_moins)
    Meta_s=(Meta_s_moins+Meta_s_plus)/2
    #gradient_term=  Meta_s* (grad_mu[0]**2+grad_mu[1]**2)  # first def 
    """
    # smooth derivative of X  # good def
    t=time_s_out
    T=T_out_s
    N= len(array_energy_s)  #; to speed up computing 
    """
    X_mean_out_s=UnivariateSpline(x=t[-N:],    # [-N:]: take just the last N values : to accelerate computing (we suppose that good step>N)
                       y=X_mean_out[-N:], 
                       w=None, bbox=[None, None], k=3, s=0, 
                       ext=0, check_finite=False)
    dXdt  = X_mean_out_s.derivative(n=1)(t)[-1]  
    """
    dXdt=(X_plus-X_moins)/dtime_s
    gradient_term=dXdt *fa_ca_s_mean   # good one 
    gradient_term_sum  =  gradient_term.sum()
  

    # energy dervative--------------------------------------------------------
    energy=array_energy_s
    """
    energy_s=UnivariateSpline(x=t[-N:],    
                       y=energy[-N:], 
                       w=None, bbox=[None, None], k=3, s=5.8e-6, 
                       ext=0, check_finite=False)
    #energy_term=(array_energy_s[istep]-array_energy_s[istep-1])/dtime_s   # first definition
    #energy_deriv  = energy_s.derivative(n=1)(t)[-1] #[-N:].mean() # good one 
    """
    energy_deriv  =  (np.diff(energy)/np.diff(time_s_out)).mean() #(energy[-1]-energy[-2])/dtime_s
     
    # eta term (AC)------------------------------------------------------------
    eta_plus=array_eta[-1].reshape(Nx,Ny)
    eta_moins=array_eta[-2].reshape(Nx,Ny)
    """
    eta_mean_out_s=UnivariateSpline(x=t[-N:],    
                    y=eta_mean_out[-N:], 
                    w=None, bbox=[None, None], k=3, s=0, 
                    ext=0, check_finite=False)

    detadt  = eta_mean_out_s.derivative(n=1)(t)[-1] 
    """
    detadt=(eta_plus-eta_moins)/dtime_s
    eta_term= (1/L_s)* (detadt)**2 
    eta_term_sum  =eta_term.sum()   

    # Temp term-----------------------------------------------------------------
    """
    T_s=UnivariateSpline(x=t[-N:], 
                       y=T_out_s[-N:], 
                       w=None, bbox=[None, None], k=3, s=0, 
                       ext=0, check_finite=False)
    
    G_mean_out_s=UnivariateSpline(x=T_out_s[-N:], 
                       y=G_mean_out[-N:], 
                       w=None, bbox=[None, None], k=3, s=0, 
                       ext=0, check_finite=False)
    dGdT=G_mean_out_s.derivative(n=1)(T)[-1]
    
    dGdT=(array_G_s_adapt[-1]-array_G_s_adapt[-2])/(T_s(t)[-1]-T_s(t)[-2])
    dTdt=(T_s(t)[-1]-T_s(t)[-2])/dtime_s  #T_s.derivative(n=1)(t)[-1]
    """
    temp_term=(array_G_s_adapt[-1]-array_G_s_adapt[-2]) /dtime_s # dGdT *dTdt
    temp_term_sum= temp_term.sum()  
 
    #--------------------------------------------------------
    res_KKS= energy_deriv + (-1* gradient_term+eta_term-temp_term).sum()
    

    if (istep % 1000 ==0):
        print('step, |RES-KKS|  , energy_term , gradient_term, eta_term, temp_term : ', istep, str('{0:.2e}'.format(np.abs(res_KKS))),\
            str('{0:.2e}'.format(energy_deriv)),\
            str('{0:.2e}'.format(gradient_term_sum)),\
            str('{0:.2e}'.format(eta_term_sum)),\
                str('{0:.2e}'.format(temp_term_sum))\
        )


    t_end_res=time.time()   
    time_res=t_end_res -t_start_res # required if need  to optimize computing time inside this function


    return res_KKS,energy_deriv,gradient_term_sum,eta_term_sum,temp_term_sum,time_res
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def normalize_parameters(
	#D0,Q,T,
    D,D0,Meta,Meta_0,
	C11,C12,C44,
	dx,dy,dz,
	sigma,alpha,
	A_thet,B_thet,C_thet,
	A_alp,B_alp,C_alp,Omega0,L,lambd):
   """
   Comupte the adimensional parameters 
   """
   # variable initialisation
   
   #R = 8.314472                  # = [J.K-1.mol-1]

   #D  = D0*np.exp(-Q/(R*T))         # = [m2.s-1]

   #lambd     = 3*dx                          # = half interfacial thickness [m]  
   kappa2    = 2*lambd*sigma*3/alpha
   w         = (sigma*3*alpha)/lambd       # = height of double-well potential [J.mol^{-1}]
   
   # last updata : w         = (sigma*3*alpha)/lambd

   # normalized variable

   L_s       = (L*C44*dx**2)/D0               # = kinetic mobility
   kappa2_s  = kappa2/(C44*dx**2)
   D_s       = D /D0
   Meta_s=Meta/Meta_0
   w_s       = w/(C44)

   A_thet_s  = A_thet/(C44*Omega0)
   B_thet_s  = B_thet/(C44*Omega0)
   C_thet_s  = C_thet/(C44*Omega0)

   A_alp_s  = A_alp/(C44*Omega0)
   B_alp_s  = B_alp/(C44*Omega0)
   C_alp_s  = C_alp/(C44*Omega0)

   C11_s    = C11/C44
   C12_s    = C12/C44 
   C44_s    = C44/C44 

   dx_s=dx/dx
   dy_s=dy/dy
   dz_s=dz/dz

   return L_s,kappa2_s,D_s,Meta_s,w_s,\
          A_thet_s,B_thet_s,C_thet_s,\
          A_alp_s,B_alp_s,C_alp_s,\
          dx_s,dy_s,dz_s,C11_s,C12_s,C44_s


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def free_energy_para_new(c_alpha,c_theta,eta,w_s,
                     A_alp_s,B_alp_s,C_alp_s,A_thet_s,B_thet_s,C_thet_s):
    """
    Compute derivative 
    """
    # variable initialisation

    # chemical free energy 
    
    fa_s   = A_alp_s*c_alpha**2+B_alp_s*c_alpha+C_alp_s  # = alpha phase 
    fa_ca_s = 2*A_alp_s*c_alpha+B_alp_s                 # = its first derivative  
    fa_caca_s = 2*A_alp_s

    
    ft_s    = A_thet_s*c_theta**2+B_thet_s*c_theta+C_thet_s   # = theta' phase     
    ft_ct_s = 2*A_thet_s*c_theta+B_thet_s                     # = first derivative
    ft_ctct_s = 2*A_thet_s
    
    # potential
    
    h     = 3.*eta**2-2*eta**3                    # = double-well potential
    h_eta = 6.*eta   -6*eta**2                    # = derivative of h with respect to eta
    
    g     = eta**2-2*eta**3+eta**4                # = monotonous function 
    g_eta = 2.*eta -6.*eta**2 + 4.*eta**3
    
    
    # smallest 
    
    #G = (1-h)*fa+h*ft+w*g
    G_s=(1-h)*fa_s+h*ft_s+w_s*g

    G_eta_s = -h_eta*(fa_s-ft_s-(c_alpha-c_theta)*fa_ca_s)+w_s*g_eta  # Hu 2007 - KKS 1999
    
    
    G_cc_s = (ft_ctct_s*fa_caca_s)/((1-h)*ft_ctct_s+h*fa_caca_s) # Hu2007 - KKS 1999

    
    G_ceta_G_cc= h_eta*( c_alpha-c_theta)

    # new def 
    #G_ceta_s=h_eta*( ft_ct_s-fa_ca_s)  # exp origin 

    #G_ceta_s= G_ceta_G_cc*G_cc_s
    
    denom= (1-h)*ft_ctct_s+h*fa_caca_s
    c_alpha_c =ft_ctct_s /denom
    c_theta_c =fa_caca_s /denom

    fa_c_s= fa_ca_s* c_alpha_c 
    ft_c_s= ft_ct_s* c_theta_c

    fa_ca_c_s= 2*A_alp_s*c_alpha_c

    G_ceta_s = -h_eta*(fa_c_s-ft_c_s-(c_alpha_c-c_theta_c)*fa_ca_s-(c_alpha-c_theta)*fa_ca_c_s)   # Hu 2007 - equation 27 : derive with respect to c
    #G_cc_s =G_ceta_s/G_ceta_G_cc  #another deficntion of G_cc
    
    return G_s,G_eta_s,G_cc_s,G_ceta_G_cc,G_ceta_s,h_eta,fa_ca_s,ft_ct_s
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#--------------------------------             Main                 -----------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------   
def calculate_energ(Nx,Ny,Nz,eta,G,kappa2,el):
    eta=eta.reshape(Nx,Ny)   # reshape eta.reshape(Nx) if 1D ...
    G=G.reshape(Nx,Ny)     
    el=el.reshape(Nx,Ny)
    #energ= np.sum(G)+0.5*kappa2*np.sum(np.asarray(np.gradient(eta))**2  ) +np.sum(el)
    energ=(G+0.5*kappa2* (np.asarray((np.gradient(eta)))**2 ) + el).sum()
    return energ           
 
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
def main(T_start,Xeq,eta,X,X_al,X_th,Lv2,tresh,
         Nx,Ny,Nz,dx,dy,dz,
         temp_tab2,epsX_tab2,D_inter_max_tab2,
	     A_thet_tab2,B_thet_tab2,C_thet_tab2,
	     A_alp_tab2,B_alp_tab2,C_alp_tab2,
	     C11_tab2,C12_tab2,C44_tab2,
	     L_tab2,gamma_tab2,alpha,
	     r,ox,oy,oz,
	     time_slm_tab1,temp_slm_tab1,X0_mat,X0_pre,
	     Vm_tab2,epsX,dtime_s,nstep,ask_step,ask_step2,
         D0,QD,Meta0T,QMetaT,X_Si_fcc_tab2,lambd,dtime_max,
         outfile,thresh_step):
   
   """
   main phase field program
   """
   

   t_start=time.time()     

   # to print residual evolution and follow iterations
   log_file = open('log.txt', 'w')
   log_file.close()
   time_step_file = open('dtime.txt', 'w')
   time_step_file.close()

   with open("log.txt", 'a') as f:
        f.write('start simulation\n')
        f.close() 

    # to print times steps during simulation
   with open("dtime.txt", 'a') as f_dtime:
        f_dtime.write('print time steps\n')
        f_dtime.close()

   # to compute fft in the main program
   a = pyfftw.empty_aligned((Nx,Ny,Nz), dtype='complex128') 
   b = pyfftw.empty_aligned((Nx,Ny,Nz), dtype='complex128')
   fft_object=pyfftw.builders.fftn(a,axes=(0,1,2), threads=12)
   ifft_object=pyfftw.builders.ifftn(b,axes=(0,1,2), threads=12)

   t_s           = 0
   t             = 0
   t_b           = 0
   T             = 0

   R             = 8.314472                  # = [J.K-1.mol-1]

   ttime=0   # compute the total time
   flag= 0   # to detect step when computing is good (fft method)

   # output  definition
   deta_mean_out = []  # = to get the average delta eta between 2 time step over the domain 
   eta_mean_out  = []  # = to get the average of eta over the domain   
   T_out         = []  # = to get the temperature
   dT_out        = []  # = to get the delta temperature between 2 time step
   dtime_out     = []  # = to get the real time step  
   time_out      = []  # = to get the real time   
   time_s_out      = []  # = to get the scaled time 
   surf_out      = []  # = to get the surface of the precipitate

   X_al_mean_out = []  # = to get the average concentration of Al in the matrix over the domain
   X_th_mean_out = []  # = to get the average concentration of Si in the matrix over the domain
   X_mean_out = []  # = to get the average concentration of Si in the matrix over the domain
   G_mean_out = []
   dX_al_mean_out= []  # = to get the average delta concentration of Al in the matrix over the domain
   dX_th_mean_out= []  # = to get the average delta concentration of Si in the matrix over the domain
   dsurf_out     = []  # = to get the average delta surface of the precipitate the domain
   L_out         = []  # = to get the mobility coefficient 

   # convert into adimensional grid
   dx_s=dx/dx
   dy_s=dy/dy
   dz_s=dz/dz
   
   # compute the spatial frequency term from fft
   k,k2,k4=prepar_fft(Nx,dx_s,Ny,dy_s,Nz,dz_s,opt="3d")
   
   # problem, the elastic stifness matrix is not updated
   T = np.interp(t, time_slm_tab1, temp_slm_tab1)
   C11 = np.interp(T, temp_tab2, C11_tab2)
   C12 = np.interp(T, temp_tab2, C12_tab2)
   C44 = np.interp(T, temp_tab2, C44_tab2)

   # compute the compliance matrix
   C_ijkl=elas_modul(C11, C12, C44)

   # compute the green tensor (here computed just once for the simulation because time consuming)
   Omega_xyzklij=compute_green(Nx,Ny,Nz,C_ijkl,k)
   
   # adimensional time step at the begining of the simulation
   dtime_s_ini = dtime_s
   
   # excess vacany parameters
   D0_Al   = 1.7e-4
   D0_manti= 3.66e-6
   Q_Al    = 142e3
   Q_manti = 110958

   HF1VA    = 63680
   SF1VA    = 2.49
   HB       = 10613
   SB_kB    = 42453
   Z        = 12
   
   surf = 0
   
   grid_pt,cell=compu_cell(Nx,Ny)

   Xv = np.exp(-(HF1VA-850*SF1VA)/(R*850))*(1-Z*0.01+Z*0.01*np.exp((HB -850*SB_kB)/(R*850)))
   
   # to earsed  
   sigel_ij, epsel_kl,el_eta,el=solve_elas(Nx,Ny,Nz,eta,Omega_xyzklij,C_ijkl,k,epsX,fft_object,ifft_object)
   # to earse
   el_eta_s = 0
   el = 0


   #----------------------------------------------------------------------------
   #----------------------------------------------------------------------------
   #----------------------------------------------------------------------------
   #for adaptive scheme
   array_energy_s=np.zeros(nstep)    # to stock normalized energy values, needed to compute residual
   array_dtime_s=np.zeros(nstep) # to store normalized time steps
   iter_max = 50  # maximum number of iterations 
   resmax=1e-1 # trials 
   resmin= 1e-4
   dtime_s_min=dtime_s_ini/20
   theta_coef=1.25  # a constant (not confuse with theta phase)

   #arrays 
   array_residus_KKS=np.zeros(nstep) # to store residual and its associated terms 
   array_residus_KKS_energ_deriv=np.zeros(nstep)
   array_residus_KKS_CH=np.zeros(nstep)
   array_residus_KKS_AC=np.zeros(nstep)
   array_residus_KKS_temp_term=np.zeros(nstep)
   array_eta_adapt=[] # to save eta values needeed to compute residual (temporary array)
   array_X_adapt=[] 
   array_X_al_adapt=[]
   array_X_th_adapt=[]
   array_D_adapt=[]
   array_G_cc_adapt=[]   
   array_G_ceta_adapt=[] 
   array_temp_ask_step=[]  
   array_G_s_adapt=[]
   array_fa_ca_s= []
   array_time_res=[]
   #----------------------------------------------------------------------------
   #good_step=1  # to comment if no nucelation

   # start the loop over the desired number of time step
   for istep in range(nstep):
    iter=1
    while True:
        #----------------------------------------------------------------------------
        #--------------------------- While loop -------------------------------------
        #----------------------------------------------------------------------------


        # Compute the temperature by linear interpolation for values given in time_slm_tab1, temp_slm_tab1
        T = np.interp(t, time_slm_tab1, temp_slm_tab1)
        #T= 438 # constant if nucleation (put your  temp)
        
        # if the desire time step is in the variable ask_step2
        # compute the surface of the precipitate
        if any(istep+1==ask_step2):
            selec_cell = grid_pt[(eta[:,:,0]>=0.1)&(eta[:,:,0]<=0.9)]
            ans_bool = np.isin(cell, selec_cell)
            
            if Ny ==1:
                surf=0
            else:
                surf=np.sum(np.all(ans_bool, axis=1))*(dx**2/(2*lambd))    

            # compute the temperature 
            T_a = T
            t_a = t


        
        
        # Compute the excess vacancy concentration
        
        Xve = np.exp(-(HF1VA-T*SF1VA)/(R*T))*(1-Z*0.01+Z*0.01*np.exp((HB -T*SB_kB)/(R*T))) 
        
        # material properties vs slm temperature 
        
        
        
        if (np.abs(1+(Xv-Xve)/Xve))>tresh:
            D  = tresh*D0*np.exp(-QD/(R*T))
            #Meta=tresh*Meta0T*np.exp(-QMetaT/(R*T))
        else :
            D  = np.abs(1+(Xv-Xve)/Xve)*D0*np.exp(-QD/(R*T))
            #Meta=np.abs(1+(Xv-Xve)/Xve)*Meta0T*np.exp(-QMetaT/(R*T))
        if (np.abs(1+(Xv-Xve)/Xve))<1:
            D  = D0*np.exp(-QD/(R*T))
            #Meta=Meta0T*np.exp(-QMetaT/(R*T))


        Meta=Meta0T*np.exp(-QMetaT/(R*T))   
        DAl     = D0_Al*np.exp(-Q_Al/(R*T))
        D_manti  = D0_manti*np.exp(-Q_manti/(R*T))
        Deff =np.abs(1+(Xv-Xve)/Xve)*(0.03*D_manti+0.97*DAl)
        
        if (istep==0):
            D_0=D
            T_0=T  # to normalize temperature 
            Meta_0=Meta
            
        
        
        if (istep==0):
                dtime = (dx**2/D)*dtime_s_ini    #  = compute the real time step with the original adimentional time step
                dtime_s=dtime_s_ini        
        # new dtime structure --------------
        

        
        if (dtime_s>dtime_max):
            dtime_s=dtime_max   # to avoid big time steps 

        if (dtime_s<dtime_s_min):
            dtime_s=dtime_s_min
        

        dtime = (dx**2/D_0)*dtime_s_ini    #  = compute the real time step with the original adimentional time step
        
        ttime=ttime+dtime
      

        #compute the evolution of vacancy concentration with the time 
        dXv     = (1-np.exp(-dtime*(Deff/Lv2)))*(Xve-Xv)
        Xv      = Xv+dXv 


        # Compute the coefficient of the free energy represented by a parabolic function

        A_alp   = np.interp(T, temp_tab2, A_alp_tab2)
        B_alp   = np.interp(T, temp_tab2, B_alp_tab2)
        C_alp   = np.interp(T, temp_tab2, C_alp_tab2)
        
        # Compute the coefficient of the siffness matrix if we want to take into account its evolution with T

        C11     = np.interp(T, temp_tab2, C11_tab2)
        C12     = np.interp(T, temp_tab2, C12_tab2)
        C44     = np.interp(T, temp_tab2, C44_tab2)

        A_thet  = np.interp(T, temp_tab2, A_thet_tab2)
        B_thet  = np.interp(T, temp_tab2, B_thet_tab2)
        C_thet  = np.interp(T, temp_tab2, C_thet_tab2)

        # Compute interface energy with T
        sigma   = np.interp(T, temp_tab2, gamma_tab2)
        # Compute molar volume with T
        Omega0  =  np.interp(T, temp_tab2, Vm_tab2)
        # Compute free strain bewteen the precipitate and the matrix with T
        epsX    =  np.interp(T, temp_tab2, epsX_tab2)
        # Compute the equilibrium concentration of Si in al matrix with T
        X_Si_fcc=np.interp(T, temp_tab2, X_Si_fcc_tab2)
        
        # = compute the height of kappa double-well potential [J.mol^{-1}]
        kappa2    = 2*lambd*sigma*3/alpha
        kappa=np.sqrt(kappa2)
        w         = (sigma*3*alpha)/lambd  

        # = compute the interface coefficient
        xi    =(19/30)*(1/Omega0)*2*A_alp*(X_Si_fcc-0.9999)**2    # unit [J.m-3]
        L = (sigma/kappa**2)*(1/(1/Meta+kappa*xi/(D*np.sqrt(2*w)))) *0.01
        


        #-----------------------------------------
        # write if file updated key values    
        if math.fmod(istep,100)==0:
            with open("dtime.txt", 'a') as f_dtime:
                f_dtime.write('step: '+str(istep)+", T= "+str('{0:.3f} '.format(T))+" D= "+str('{0:.2e}'.format(D))+ " L= "+str('{0:.2e}'.format(L))+ " Meta= "+str('{0:.2e}'.format(Meta))+ " Vm= "+ str('{0:.2e}'.format(Omega0))+ ' dtime_s= '+ str('{0:.2f}'.format(dtime_s))+' \n')
                f_dtime.close()

        # compute adimensional parameters in the phase field
        L_s,kappa2_s,D_s,Meta_s,w_s,A_thet_s,B_thet_s,C_thet_s,\
        A_alp_s,B_alp_s,C_alp_s,dx_s,dy_s,dz_s,C11_s,C12_s,C44_s=\
        normalize_parameters(D,D_0,\
                Meta,Meta_0,C11,C12,C44,dx,dy,dz,sigma,alpha,
                A_thet,B_thet,C_thet,A_alp,B_alp,C_alp,Omega0,L,lambd)

        # compute adimensional free energy terms + derivative
        G_s,G_eta_s,G_cc_s,G_ceta_G_cc,G_ceta_s,h_eta,fa_ca_s,ft_ct_s=free_energy_para_new(X_al,X_th,eta,w_s,
                        A_alp_s,B_alp_s,C_alp_s,A_thet_s,B_thet_s,C_thet_s)
        
        # compute elastic energy contribution to total energy
        sigel_ij,epsel_kl, el_eta,el=solve_elas(Nx,Ny,Nz,eta,Omega_xyzklij,C_ijkl,k,epsX,fft_object,ifft_object)
        
        # compute fourier transform of
        etak =fft_(eta)
        etakp=etak
        G_etak=fft_(G_eta_s)        

        el_eta_s=el_eta/C44                      # to reactivated
        el_s=el/(C44)


        el_etak=fft_(el_eta_s)                   # to reactivated
        G_etak_el_etak = fft_(G_eta_s+el_eta_s)  # to reactivated


        # solve cahn - hilliard in fourier space
        etak=(-dtime_s*L_s*G_etak_el_etak+etak)/(1+dtime_s*L_s*kappa2_s*k2)  
        eta_b = np.real(ifft_(etak))



        Xk   =fft_(X)

        # solve allen cahn in fourier space
        Xk=(dtime_s*D_s*(1j*k[0]*fft_(G_ceta_G_cc*np.real(ifft_(1j*k[0]*etakp)))\
        +1j*k[1]*fft_(G_ceta_G_cc*np.real(ifft_(1j*k[1]*etakp)))\
        +1j*k[2]*fft_(G_ceta_G_cc*np.real(ifft_(1j*k[2]*etakp)))  )\
            +Xk)\
            /(1+dtime_s*k2*D_s)   # semi implicit    
        

        # concentration variable at time tb
        X_b   = np.real(ifft_(Xk))  
        


        
        h_b        = 3.*eta_b**2-2*eta_b**3
    
        # concentration of Si in al phase at time tb from equality of chemical potential
        X_al_b=(X-h_b*((B_alp-B_thet)/(2*A_thet)))/((1-h_b)+h_b*(A_alp/A_thet))


        # if some weird value appear (infinite or nan), replace the following ones  
        X_al_b[ X_al_b == np.inf]=1e-6
        X_al_b[X_al_b == -np.inf]=-1e-6
        X_al_b[np.isnan(X_al_b)]=1e-6

        # concentration of Si in diamond phase at time tb
        X_th_b=(A_alp/A_thet)*X_al_b+(B_alp-B_thet)/(2*A_thet)

        X_th_b[ X_th_b == np.inf]=1e-6
        X_th_b[X_th_b == -np.inf]=-1e-6
        X_th_b[np.isnan(X_th_b)]=1e-6

        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------      
        #-------------------------------------------------------------------------------------------------
        # store variables needed for adaptive time stepping (compute residual)
        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------     
        #-------------------------------------------------------------------------------------------------   
        
        energy_s=calculate_energ(Nx,Ny,Nz,eta_b,G_s,kappa2_s,el_s) # dimensionless energy, 
        array_energy_s[istep]=energy_s
        array_dtime_s[istep]=dtime_s
        
        # compute adimensional free energy terms + derivative (updated after FFT resolution)
        G_b_s,G_eta_b_s,G_cc_b_s,G_ceta_G_cc,G_ceta_b_s,h_eta,fa_ca_s,ft_ct_s=free_energy_para_new(X_al_b,X_th_b,eta_b,w_s,A_alp_s,B_alp_s,C_alp_s,A_thet_s,B_thet_s,C_thet_s)

        # use these array values to adapt the time stepping
        if (flag==1):
            array_fa_ca_s.append(fa_ca_s)
            array_X_adapt.append(X_b)
            array_X_al_adapt.append(X_al_b)
            array_X_th_adapt.append(X_th_b)
            array_eta_adapt.append(eta_b)
            array_D_adapt.append(D_s)
            array_G_cc_adapt.append(G_cc_b_s)
            array_G_ceta_adapt.append(G_ceta_b_s)
            array_G_s_adapt.append(G_b_s)

        num_elem=2
        if (flag==1) and (istep>=good_step+1):  # prepare lists before the activation of the adapt scheme 
            array_X_adapt=array_X_adapt[-num_elem:]  # keep the last useful num_elem elements of the list (to avoid memory consumption)  (here to smoothly compute dGdX)
            array_X_al_adapt=array_X_al_adapt[-num_elem:] 
            array_X_th_adapt=array_X_th_adapt[-num_elem:] 
            array_eta_adapt=array_eta_adapt[-num_elem:] 
            array_D_adapt=array_D_adapt[-num_elem:] 
            array_G_cc_adapt=array_G_cc_adapt[-num_elem:] 
            array_G_ceta_adapt=array_G_ceta_adapt[-num_elem:] 
            array_G_s_adapt= array_G_s_adapt[-num_elem:]  
            array_fa_ca_s=array_fa_ca_s[-num_elem:]
        
        #-------------------------------------------------------------------------------------------------
        #------------------------   adaptive time stepping -----------------------------------------------
        #-------------------------------------------------------------------------------------------------     
        # stop criterion of the simulation
        if  (iter==iter_max):
            print('maximum criterion reached to stop the simulation'+'\n'+ 'dtime_s= '+str('{0:.2e}'.format(dtime_s))+'\n'+\
                'number of iterations: '+str(iter))
            plt.plot(T_out[:istep],array_dtime_s[:istep])
            plt.title('time step evolution')
            plt.show()
            import sys
            sys.exit('maximum number of iterations reached')
        
        
        if (flag== 0) or (istep<good_step+num_elem): # the value num_elem=N=2 to get enough data (minimum 02) to correctly execute def compute_residual-KKS
            break # break the while loop (nothing to do) 
        else:   # activate the adaptive scheme : when computing becomes stable
                N=num_elem # need to have already two elements of each list (array_X_adapt, array_...) before activating the adapt scheme
                res,energy_deriv,gradient_term,eta_term,temp_term,time_res=compute_residual_KKS(good_step,time_s_out,array_dtime_s,array_energy_s,X_mean_out,\
                     eta_mean_out,G_mean_out,array_fa_ca_s,Meta_s,array_G_s_adapt,T_out/T_0,array_D_adapt,array_G_cc_adapt,array_G_ceta_adapt,L_s,array_eta_adapt,\
                    array_X_adapt,array_X_al_adapt,array_X_th_adapt,fa_ca_s,ft_ct_s,h_eta,X_al_b,X_th_b,istep,dtime_s,Nx,Ny,Nz,dx,dx_s,dy_s,dz_s)
                
                
                array_residus_KKS[istep]=res
                array_residus_KKS_energ_deriv[istep]=energy_deriv  
                array_residus_KKS_CH[istep]=gradient_term  
                array_residus_KKS_AC[istep]=eta_term  
                array_residus_KKS_temp_term[istep]=temp_term  
                array_time_res.append(time_res)

                # resmax definition
                
                if (istep==good_step+N) and (iter==1):  
                    #resmax= np.max(np.asarray([np.abs(energy_deriv),np.abs(gradient_term),np.abs(eta_term),np.abs(temp_term)])) # trial 
                    print('resmax= ',str('{0:.2e}'.format(resmax)),' resmin= ',str('{0:.2e}'.format(resmin)),'\n')
                    print('good step: ',good_step,'\n')
                  
                
                #-----------------------------------------------------------------------
                if (np.abs(res)>resmax):
                    with open("log.txt", 'a') as f:
                        f.write('step '+ str(istep)+': RE>resmax\n')
                        f.write ('Residual = '+ str('{0:.2e}'.format(np.abs(res))) + ','+ ' Energy_deriv= ' + str('{0:.2e}'.format(energy_deriv))+\
                             ','+ ' Potentiel_grad= '+str('{0:.2e}'.format(gradient_term ))+ ',' + ' eta_term= '+str('{0:.2e}'.format(eta_term))+\
                                ' temp_term= '+str('{0:.2e}'.format(temp_term))+',' +'         Energy= '+ str('{0:.2e}'.format(energy_s) )+'\n')
                        f.close()

                    dtime_s/= theta_coef
                    with open("dtime.txt", 'a') as f_dtime:
                        f_dtime.write('RE>resmax\n')
                        f_dtime.write('repeat istep '+str(istep)+ ' : iteration '+ str(iter)+ ' dtime_s:  '+str('{0:.2e}'.format(dtime_s))+'\n'+'\n' )
                        f_dtime.close()
                    
                    iter +=1                
                    
                    if (iter>iter_max):
                        with open("dtime.txt", 'a') as f_dtime:
                            f_dtime.write('\n'+str(iter_max)+' iterations reached ; stop the simulation'+'\n')
                            f_dtime.close()

                        with open("log.txt", 'a') as f:
                            f.write ('step: '+str(istep)+', Residual = '+ str('{0:.2e}'.format(res)) + ','+ ' Energy_deriv= ' + str('{0:.2e}'.format(energy_deriv))+ ','+\
                                 ' Potentiel_grad= '+str('{0:.ef}'.format(gradient_term ))+ ',' + ' eta_term= '+str('{0:.2e}'.format(eta_term))+ ',' +\
                                    ' temp_term= '+str('{0:.2e}'.format(temp_term))+','+'         Energy= '+ str('{0:.2e}'.format(energy_s) )+'\n')
                            f.close()

                        break  # break the while loop
                        #import sys
                        #sys.exit('maximum number of iterations reached')


                    
                    continue   # continue the while loop =>  restart the "for loop" actual step with a new dtime=> recompute X and RE

                #-----------------------------------------------------------------------
                elif (np.abs(res)<=resmax) and (np.abs(res)>resmin) :
                    with open("log.txt", 'a') as f:
                        f.write('step: '+str(istep)+', We get  (resmin <RE <resmax) ;  go to step '+str(istep+1)+'\n')
                        f.write ('Residual = '+ str('{0:.2e}'.format(np.abs(res))) + ','+ ' Energy_deriv= ' + str('{0:.2e}'.format(energy_deriv))+ ','+\
                                ' Potentiel_grad= '+str('{0:.2e}'.format(gradient_term ))+ ',' + ' eta_term= '+str('{0:.2e}'.format(eta_term))+ ',' +\
                                ' temp_term= '+str('{0:.2e}'.format(temp_term))+','+'         Energy= '+ str('{0:.2e}'.format(energy_s) )+'\n')
                        f.close()
                        
                    break # break the while loop => go to next "for loop" iteration with a new dtime 
                
                #-----------------------------------------------------------------------
                elif (np.abs(res)<resmin):
                    with open("log.txt", 'a') as f:
                        f.write(' We get  :RE <resmin;  go to step '+str(istep)+'\n')
                        f.write ('Residual = '+ str('{0:.2e}'.format(np.abs(res))) + ','+ ' Energy_deriv= ' + str('{0:.2e}'.format(energy_deriv))+ ','+\
                         ' Potentiel_grad= '+str('{0:.2e}'.format(gradient_term ))+ ',' + ' eta_term= '+str('{0:.2e}'.format(eta_term))+ ',' +\
                            ' temp_term= '+str('{0:.2e}'.format(temp_term))+'         Energy= '+ str('{0:.2e}'.format(energy_s) )+'\n')
                        f.close()

                    dtime_s*=theta_coef
                             
                #-----------------------------------------------------------------------

                    break # break the while loop => go to next "for loop" iteration with a new dtime_s 
        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------
        #-------------------------------------------------------------------------------------------------
        #-----------------------------------------
        #break    # uncomment if the adaptive scheme is desactivated
        #----------------------------------------------------------------------------
        #--------------------------- end While loop ---------------------------------
        #----------------------------------------------------------------------------
        
     
    # make sure to store good values 
    if (flag==1):
      array_X_adapt[-1]=X_b
      array_X_al_adapt[-1]=X_al_b
      array_X_th_adapt[-1]=X_th_b
      array_eta_adapt[-1]=eta_b
      array_D_adapt[-1]=D_s
      array_G_cc_adapt[-1]=G_cc_s
      array_G_ceta_adapt[-1]=G_ceta_s
      array_G_s_adapt[-1]=G_s
      array_energy_s[istep]=energy_s
      array_dtime_s[istep]=dtime_s
    #print('len(array_X_adapt) ',len(array_X_adapt))

    # update the time step
    t_s+=dtime_s
    dtime= (dx**2/D_0)*dtime_s
    t_b+=(dx**2/D_0)*dtime_s
      


    # get the output from ask_step2 array
    if any(istep+1==ask_step2):
        # slm temperature vs time 
        T_b = np.interp(t_b, time_slm_tab1, temp_slm_tab1)
        # output :
        selec_cell = grid_pt[(eta_b[:,:,0]>=0.1)&(eta_b[:,:,0]<=0.9)]
        ans_bool = np.isin(cell, selec_cell)
        
        # compute surface of the precipitate 
        if Ny ==1:
            surf_b = 0
        else:
            surf_b=np.sum(np.all(ans_bool, axis=1))*(dx**2/(2*lambd))


    X_al_mean = np.mean(X[eta<0.05])

    if (X_al_mean <=Xeq) and (flag== 0) and (istep>1):    # when istep> 1 ; the concentration increase (not physical) : take 10 by security
        good_step=istep+thresh_step        # at this step the computing is stable (no more fft method numerica issues ) ; this value (thresh_step) is a threshhold and could be more adjustable
        flag= 1


    eta_mean = np.mean(eta)
    eta_mean_b =np.mean(eta_b)

    X_al_mean_out.append(X_al_mean)
    X_th_mean_out.append(np.mean(X[eta>0.95]))
    X_mean_out.append(np.mean(X[(eta>0.05) & (eta>0.95) ]) )
    dX_al_mean_out.append(np.mean(X_b[eta_b<0.05])-np.mean(X[eta<0.05]))
    dX_th_mean_out.append(np.mean(X_b[eta_b>0.95])-np.mean(X[eta>0.95]))
    deta_mean_out.append(eta_mean_b-eta_mean)
    dT_out.append(T_b-T) 
    eta_mean_out.append(eta_mean)
    G_mean_out.append(np.mean(G_s.reshape(Nx,Ny)))
    surf_out.append(surf_b)
    dsurf_out.append(surf_b-surf)
    T_out.append(T)
    dtime_out.append(dtime)
    time_out.append(t)
    time_s_out.append(t_s)
    L_out.append(L) 
    #dis_inta = gridx[(X_a[:,0,0]<0.95) & (X_a[:,0,0]>0.05)][5:]
    #X_inta=X_a[:,0,0][(X_a[:,0,0]<0.95) & (X_a[:,0,0]>0.05)][5:]
    #dis_intb = gridx[(X[:,0,0]<0.95) & (X[:,0,0]>0.05)][5:]
    #X_intb=X[:,0,0][(X[:,0,0]<0.95) & (X[:,0,0]>0.05)][5:]
    #pos_a = np.interp(0.5,X_inta,dis_inta)
    #pos_b = np.interp(0.5,X_intb,dis_intb)
    #vel = (pos_b-pos_a)/((dx**2/D)*dtime_s)
    #inter_pos_b =gridx[h_eta[:,0,0]==np.max(h_eta[0:1020,0,0])]
    #vel = inter_pos_b#(inter_pos_b-inter_pos_a)/((dx**2/D)*dtime_s)
    #velocity.append([pos_b,t])#(inter_pos_b-inter_pos_a)/((dx**2/D)*dtime_s))

    # update the variables
    eta=eta_b
    X=X_b
    X_al=X_al_b
    X_th=X_th_b
    t   =t_b

    # print values if time step is in ask_step
    

    if any(istep+1==ask_step):
        import os
        Save_path=os.path.join(os.getcwd(),'saved_maps')
        outfile="map"
        T_max=np.asarray(T_out).max()
        file_name=outfile+""+'{0:.2f}'.format(ttime)+"s_"+ str(T_start)+"_"+'{0:.2f}'.format(T_max)+"K"
        np.savez_compressed(os.path.join(Save_path, file_name), ttime=ttime,time_out=time_out,T_out=T_out,\
X_al_mean_out=X_al_mean_out,X_th_mean_out=X_th_mean_out,eta_mean_out=eta_mean_out,surf_out=surf_out,X=X,X_al=X_al,X_th=X_th,eta=eta)
    """
        #print values
        #inter_pos_a = np.sum(gridx[(X_a[:,0,0]<0.8) & (X_a[:,0,0]>0.2)][3:])/3
        #inter_pos_b = np.sum(gridx[(X[:,0,0]<0.8) & (X[:,0,0]>0.2)][3:])/3  
        #inter_pos_a =gridx[h_eta_a[:,0,0]==np.max(h_eta_a[0:1020,0,0])]
        
        #print("posa-b:%12.9e \n" %(inter_pos_a-inter_pos_b))
        
        print("true values \n")
        print("D:%E \nT:%f \n" %(D,T))
        print("Meta:%E \n" %(Meta))
        print("A_alp:%E \nB_alp:%E \nC_alp:%E \n" %(A_alp,B_alp,C_alp))
        print("A_thet:%E \nB_thet:%E \nC_thet:%E \n" %(A_thet,B_thet,C_thet))
        print("C11:%E \nC12:%E \nC44:%E \n" %(C11,C12,C44))
        print("sigma:%E \nOmega0:%E \nL:%E \n" %(sigma,Omega0,L))
        print("epsX:%E \nX0_mat:%E \nX0_pre:%E \n" %(epsX,X0_mat,X0_pre))
        print("normalized true values \n")
        print("D_s:%E \nT:%f \n" %(D_s,T))
        print("A_alp_s:%E \nB_alp_s:%E \nC_alp_s:%E \n" %(A_alp_s,B_alp_s,C_alp_s))
        print("A_thet:%E \nB_thet:%E \nC_thet:%E \n" %(A_thet_s,B_thet_s,C_thet_s))
        print("C11_s:%E \nC12_s:%E \nC44_s:%E \n" %(C11_s,C12_s,C44_s))
        print("sigma:%E \nOmega0:%E \nL_s:%E \n" %(sigma,Omega0,L_s))
        print("epsX:%E \nX0_mat:%E \nX0_pre:%E \n" %(epsX,X0_mat,X0_pre))
        print("save step : %i \n"%(istep))
        print("real time : %E s \n"%(t))
        print("t_adapt : %f \n"%(dtime_s))
        #np.savez_compressed(outfile+'{:06.2e}'.format(t)+"s", t=t,sig=sigel_ij,X=X,eta=eta)
    """

   #-----------------------------------------  

   
   print("Temperature reached : %s K ---" % ( np.asarray(T_out).max() ))
   simulation_time=time.time() - t_start
   print("simulation time : " + (str(datetime.timedelta(seconds=simulation_time))) ) 
   return X,X_al,X_th,eta,sigel_ij,epsel_kl,eta_mean_out,deta_mean_out,X_mean_out,X_al_mean_out,dX_al_mean_out,\
          X_th_mean_out,dX_th_mean_out,dT_out,T_out,dtime_out,time_out,surf_out,dsurf_out,el_eta_s,el,G_eta_s,G_ceta_G_cc,L_out,\
          array_energy_s,array_dtime_s,good_step,ttime,array_residus_KKS,array_residus_KKS_energ_deriv,array_residus_KKS_CH,\
                array_residus_KKS_AC, array_residus_KKS_temp_term,array_time_res
          



def stopSimulation(self):

    # Disable relevant buttons to reset
    print("stop criterion reached, exit")
    self.__enableSteeringButtons(False)
    self.__enableSimulationButtons(False)

    # Reset controls
    self.clearControls()



