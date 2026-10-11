# %%
"""
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#description" data-toc-modified-id="description-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>description</a></span></li><li><span><a href="#Import" data-toc-modified-id="Import-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Import</a></span><ul class="toc-item"><li><span><a href="#external-library" data-toc-modified-id="external-library-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>external library</a></span></li><li><span><a href="#my-function" data-toc-modified-id="my-function-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>my function</a></span></li></ul></li><li><span><a href="#input-for-the-code" data-toc-modified-id="input-for-the-code-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>input for the code</a></span><ul class="toc-item"><li><span><a href="#load-the-microstructure" data-toc-modified-id="load-the-microstructure-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>load the microstructure</a></span></li></ul></li><li><span><a href="#run-the-code" data-toc-modified-id="run-the-code-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>run the code</a></span></li><li><span><a href="#save-results" data-toc-modified-id="save-results-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>save results</a></span></li><li><span><a href="#load-the-results" data-toc-modified-id="load-the-results-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>load the results</a></span></li></ul></div>
"""

# %%
"""
# Phase-field model for growth and coarsening of Si precipitate in AlSi10Mg SLM in a super-saturated matrix
This model is based on the Kim-Kim-Suzuki model [1]
References:


[1] Kim, Kim, and Suzuki. "Phase-field model for binary alloys." Physical Review E 60:6;7186-7197 (1999).

"""

# %%
"""
# Import
"""

# %%
"""
## external library
"""

# %%
import os
import numpy as np
import importlib
import collections
from os import chdir
from scipy.interpolate import UnivariateSpline #pour ma spline
import sys
import random
import importlib
from scipy.interpolate import splev, splrep
from itertools import cycle
import pyFi_adapt
importlib.reload(pyFi_adapt)
from pyFi_adapt import *
import pyvista as pv
import matplotlib.pyplot as plt


# %%
"""
# input for the code
"""

# %%
# 2 precipitates , small grid

# grid dimension
# points
Nx = 735
Ny = 735

Nz =1  # Nz=1 -> 2D 
# spacing
dx=  0.5*1e-9 # [m]

dy= 1*dx      # [m]

dz= dx # [m]


# %%
"""
### new grid 
"""

# %%
micro=np.load("micro_final.npz")

# %%
xyz_r=np.asarray([micro['ox_cyl'],micro['oy_cyl'],micro['oz_cyl'] ,micro['r_cyl'] ]).transpose()

# %%
#np.savetxt('xyz_r.txt',xyz_r)

# %%
import numpy as np
T_end=850
T_start=600
total_time=3* (T_end -T_start)
total_time

# %%
Meta0 = 5.60e-8
QMeta = 1.10e+05

# time and temperature to reproduce DSC conditions

time_slm_tab1      = np.array([0,total_time])             # = time [s] 
temp_slm_tab1      = np.array([T_start,T_end])           # = temperature [K] 

X0_mat,X0_pre=0.025,0.99999       # = molar composition of the Al matrix and precipitate in SI [mol.mol-1]

tresh = 1e9

dtime_s  = 50    # = adimensional time step [-]
dtime_max=5*  dtime_s           # = maximum dimensionless time step bound [s]

nstep=1e3
print_times=10
ask_step=np.linspace(1,int(nstep),print_times,dtype=int)   # = step to save eta and Xsi 2d map 
ask_step2=np.linspace(1,3e3,400
                      ,dtype=int) # = step to save averag Xsi in the matrix, average eta

Lv2      = (250e-9)**2            # = vacancies mean free path [m2]


nstep   = np.max(ask_step)        # = maximum step in the simulation

alpha=2.2
Xeq  =0.025
epsX =0


lambd = 2.5*dx                    # = half of interface thickness [m] 

outfile = "precip_si"            # 


# diffusion coefficient 

D0=3.66e-6                       # = pre-exponential factor [m2.s-1]
QD=110958                        # = activation energy [J.mol-1]

c_Meta  = 1.                     # = factor for mobility []
c_D = 1.                         # = factor for diffusion [-]
c_eps0 = 0.01                        # = factor for epsilon [-]
c_gamma = 2.15                   # = factor for interface energy [-]

Meta0bis=Meta0*c_Meta            #  = interface mobility coefficient pre-exponential coefficent 


# import data

# data2 : data containing all the temperature dependant parameters
data2 = np.load("final_export.npz")
print(data2.files)

R     = 8.314472                  # = perfect gas constant[J.K-1.mol-1]

temp_tab2      = data2["T"]        # = temperature table for which the properties are given [K]
X_Si_fcc_tab2   =data2["X_Si_fcc"] # = concentration of Si in the matrix 
                                   #   at equiblrium at temperature temp_tab2 given by the phase diagram

# Free energy parabola parameter
# F=A*Xsi**2+B*XSi+C
# alpha -> matrix
# thet  -> precipitate

A_alp_tab2  = data2["A_fcc"]    # = A for alpha
B_alp_tab2  = data2["B_fcc"]    # = B for alpha
C_alp_tab2  = data2["C_fcc"]    # = C for alpha

A_thet_tab2 = data2["A_dia"]    # = A for thet
B_thet_tab2 = data2["B_dia"]    # = B for thet
C_thet_tab2 = data2["C_dia"]    # = C for thet

# stiffness coeficient for AlSi

C11_tab2         = data2["C11"]     # = [MPa]
C12_tab2         = data2["C12"]     # = [MPa]
C44_tab2         = data2["C44"]     # = [MPa]

D_inter_max_tab2 = data2["D_inter_max"]

epsX_tab2        = data2["delta"]   # = free stress strain 
L_tab2           = data2["L_max"]   # = mobility coeffient for allen-cahn

Vm_tab2          = data2["Vm"]      # = molar volume
gamma_tab2       = data2["gamma"]   # = surface enegy


D0    = data2["D0"]                 # = pre-exponential factor for diffusion [m2.s-1]
QD    =data2["QD"]                  # = activation energy for diffusion [J.mol-1]
Meta0ori =data2["Meta0"]            # = pre-exponential factor for mobility  
QMetaori =data2["QMeta"]            # = activation energy for interface velocity [J.mol-1]





# %%
@jit(nopython=True)
def micro_numba_2(Nx,Ny,Nz,dx,dy,dz,X0_mat,X0_pre,r,ox,oy,oz):
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

# %%
"""
## load the microstructure 
"""

# %%
# cylinder
# from a file in blender 

micro_ini ="manual"

if micro_ini =="fromfile":

   # load file, format npz
    
   cyl = np.load("blender/cyl_packing.npz") # print(cyl.files)

   # assign coordinate x,y,z 

   ox_cyl=np.transpose(cyl['xyz'])[2]    # = x coordinate of the precipitate [nm]

   oy_cyl=np.transpose(cyl['xyz'])[0]    # = y coordinate of the precipitate [nm]

   oz_cyl=np.zeros(len(ox_cyl))          # = z coordinate of the precipitate [nm] -> 2D =0

   r_cyl = cyl['radius']*0.8            # = radius of the precipitate [nm]
   

   # thickness of the precipitate wall
  

   up_limit = -18.4e-9          # = before   -18.4e-9 / last -15e-9

   low_limi = -86e-9            # = before -65e-9  / last -86.5e-9


   # new coorindate of the wall

   ox_cyl_sor  =ox_cyl[(ox_cyl>low_limi)&(ox_cyl<up_limit)]

   oy_cyl_sor  =oy_cyl[(ox_cyl>low_limi)&(ox_cyl<up_limit)]

   oz_cyl_sor  =oz_cyl[(ox_cyl>low_limi)&(ox_cyl<up_limit)]

   r_cyl_sor   = r_cyl[(ox_cyl>low_limi)&(ox_cyl<up_limit)]

elif micro_ini=="manual":   
 
   [ox_cyl_sor, oy_cyl_sor, oz_cyl_sor, r_cyl_sor ]=  [xyz_r[:,0],xyz_r[:,1],xyz_r[:,2],xyz_r[:,3]] 
   #oy_cyl_sor= np.zeros(len(oy_cyl_sor))
  


eta_micro,X_micro,X_al_micro,X_th_micro=micro_numba_2(Nx,Ny,Nz,dx,dy,dz,X0_mat,

                                                    X0_pre,r_cyl_sor,ox_cyl_sor

                                                    ,oy_cyl_sor,oz_cyl_sor)


eta,X,X_al,X_th =eta_micro,X_micro,X_al_micro,X_th_micro
                                                   

# if nucleation is desired 
"""
nb_nuc=0
ox_s=np.zeros(nb_nuc)
r_s =np.ones(nb_nuc)*1e-9
ox_s[:int(nb_nuc/2)]=np.random.random_sample(int(nb_nuc/2))*200e-9
ox_s[int(nb_nuc/2):]=300e-9+np.random.random_sample(int(nb_nuc/2))*200e-9
oy_s=np.random.random_sample(nb_nuc)*200e-9
oz_s=np.zeros(nb_nuc)
eta,X,X_al,X_th = nucle_numba(Nx,Ny,Nz,dx,dy,dz,eta_micro,X_micro,X_al_micro,X_th_micro,r_s,ox_s,oy_s,oz_s)
"""

plt.imshow(eta.reshape(Nx,Ny).transpose())
plt.savefig('eta_init')
plt.close()


# %%
Meta0_c_Meta=Meta0*c_Meta
gamma_tab2_c_gamma=gamma_tab2*c_gamma
epsX_tab2_c_eps0=epsX_tab2*c_eps0
D0_c_D=D0*c_D
epsX_tab2_c_eps0=epsX_tab2*c_eps0
thresh_step=0

# %%
"""
# run the code 
"""

# %%
X_out,X_al_out,X_th_out,eta_out,sigel_ij_out,epsel_kl_out,eta_mean_out,deta_mean_out,X_mean_out,X_al_mean_out,dX_al_mean_out,\
          X_th_mean_out,dX_th_mean_out,dT_out,T_out,dtime_out,time_out,surf_out,dsurf_out,el_eta_s,el,G_eta_s,G_ceta_G_cc,L_out,\
           array_energy_s,array_dtime_s,good_step,ttime,array_residus_KKS,array_residus_KKS_energ_deriv,array_residus_KKS_CH,\
                array_residus_KKS_AC, array_residus_KKS_temp_term,array_time_res=\
            pyFi_adapt.main(T_start,Xeq,eta,X ,X_al,X_th,Lv2,tresh,
         Nx,Ny,Nz,dx,dy,dz,\
         temp_tab2,epsX_tab2_c_eps0,D_inter_max_tab2,\
         A_thet_tab2,B_thet_tab2,C_thet_tab2,\
         A_alp_tab2,B_alp_tab2,C_alp_tab2,\
         C11_tab2,C12_tab2,C44_tab2,\
         L_tab2,gamma_tab2_c_gamma,alpha,\
         r_cyl_sor,ox_cyl_sor,oy_cyl_sor,oz_cyl_sor,\
         time_slm_tab1,temp_slm_tab1,X0_mat,X0_pre,\
         Vm_tab2,epsX,dtime_s,nstep,ask_step,ask_step2,\
         D0_c_D,QD,Meta0_c_Meta,QMeta,X_Si_fcc_tab2,lambd,dtime_max,\
         outfile,thresh_step)   


# %%
# plot microstructure
plt.imshow(eta_out.reshape(Nx,Ny).transpose())
plt.savefig('eta_out')
plt.close()


# %%
"""
#### Energy evolution
"""

# %%
plt.plot(time_out[good_step:50:-1], array_energy_s[good_step:50:-1])
plt.xlabel('time')
plt.title('Energy evolution')
plt.savefig('Energy evolution')
plt.close()

# %%
"""
### Mean concentrations of Si in alpha and diamond phases
"""

# %%
plt.plot(time_out[good_step:-1],X_al_mean_out[good_step:-1],'r',label='X_al_alpha')
#plt.plot(time_out[good_step:-1],X_th_mean_out[good_step:-1],'c',label='X_Si_diamond')
plt.axhline(y=Xeq)
plt.xlabel('time [s]')
#plt.ylim(0.01,0.04)
plt.legend()
plt.savefig('X Si in alpha phase')
plt.close()

# %%
"""
#### Residuals evolution
"""

# %%
"""
### KKS
"""

# %%
# C-H
N=5 # pas
plt.plot(array_residus_KKS[good_step+1:-1:N] , label=' Total', color='black')
plt.plot(array_residus_KKS_energ_deriv[good_step+1:-1:N] , label=' Energy derivative', color='b')
plt.plot(array_residus_KKS_CH[good_step+1:-1:N] , label=' CH', color='r')
plt.plot(array_residus_KKS_AC[good_step+1:-1:N] , label=' AC', color='c')
plt.plot(array_residus_KKS_temp_term[good_step+1:-1:N] , label=' Temperature term', color='m')
plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.8))
#plt.xlabel('time')
plt.ylabel(' Residual') 
plt.title('Evolution of the residual terms')
plt.savefig('Residual')
plt.close()

# %%
plt.plot(T_out,array_dtime_s,linestyle='dashdot')
plt.xlabel(' dt') 
plt.ylabel(' Temp (K)') 
plt.savefig('Time_steps evolution')
plt.close()

# C-H
N=5 # pas
plt.plot(T_out[good_step+1:-1:N],array_residus_KKS[good_step+1:-1:N] , label=' Total', color='black')
plt.plot(T_out[good_step+1:-1:N],array_residus_KKS_energ_deriv[good_step+1:-1:N] , label=' Energy derivative', color='b')
plt.plot(T_out[good_step+1:-1:N],array_residus_KKS_CH[good_step+1:-1:N] , label=' CH', color='r')
plt.plot(T_out[good_step+1:-1:N],array_residus_KKS_AC[good_step+1:-1:N] , label=' AC', color='c')
plt.plot(T_out[good_step+1:-1:N];array_residus_KKS_temp_term[good_step+1:-1:N] , label=' Temperature term', color='m')
plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.8))
#plt.xlabel('time')
plt.ylabel(' Residual vs Temperature') 
plt.title('Evolution of the residual terms')
plt.savefig('Residual 2')
plt.close()

# C-H
N=5 # pas
plt.plot(T_out[good_step+1:-1:N],array_residus_KKS[good_step+1:-1:N] , label=' Total', color='blue')
plt.legend(loc='center left', bbox_to_anchor=(0.55, 0.8))
#plt.xlabel('time')
plt.ylabel(' Total Residual ') 
plt.title('Evolution of the residual terms')
plt.savefig('Residual 3')
plt.close()