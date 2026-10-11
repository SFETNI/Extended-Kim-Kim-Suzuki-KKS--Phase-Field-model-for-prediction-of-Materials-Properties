import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np


system = "linux"

if system=="linux":
    sys.path.insert(0,"/media/jo/docs/ulieg/fonct_pytho/")


from os import chdir
chdir("/media/jo/docs/ulieg/oral/fig/fig_test/")

import jupyter_style
importlib.reload(jupyter_style)
from jupyter_style import *


import free_nrj_mob
importlib.reload(free_nrj_mob)
from free_nrj_mob import *

cm_to_inch = 1/2.54
fig_wid_inch=7.5*cm_to_inch
golden_ratio = 0.75
prop = 0.75

figLatexSize(fig_width_cm=7.5,fontSize=8,legFontSiz=6,lineSize=1,markerSize=3, golden_mean=prop)


#dic=dict(np.load("4d02_durci_expli.npz"))
#dic=dict(np.load("4d02_durci_expli2.npz"))

#print(dic.keys())
# traitement des données 


#


#%%

dout=dict(np.load("3d02_dsc_mean.npz"))

 
dout["T_mean"]
 
dout["X_al_mean"]


dout["eta_mean"]


dout["t_mean"]

dout["surf_mean"]

#plt.plot(dout["T_mean"],dout["X_al_mean"])

plt.plot(dout["T_Xv_mean"],dout["Xv_mean"])



# %% pour le Eta

t = dout["t_mean"]
T = dout["T_mean"]
eta = dout["eta_mean"]
X_al = dout["X_al_mean"]
surf = dout["surf_mean"]


eta_s=UnivariateSpline(x=t, 
                       y=eta, 
                       w=None, bbox=[None, None], k=3, s=5.8e-6, 
                       ext=0, check_finite=False)



plt.plot(dout["T_mean"],dout["eta_mean"],color="b",label="brut")
plt.plot(dout["T_mean"],eta_s(dout["t_mean"]),color="r",label="spline")
plt.xlabel("T [K]")
plt.ylabel(r"$\eta$")
plt.legend()

# %% pourle X_AL

t[0] = 6
X_al[0] = 0.025




t_X_al_mod = t[X_al<=0.025]
X_al_mod=X_al[X_al<=0.025]



X_al_s=UnivariateSpline(x=t_X_al_mod, 
                       y=X_al_mod, 
                       w=None, bbox=[None, None], k=3, s=5.8e-6, 
                       ext=0, check_finite=False)

plt.plot(t*(1/3)+400,X_al,color="b",label="brut")
 
plt.plot(t_X_al_mod*(1/3)+400,X_al_mod,color="g",label="brut modfi")
plt.plot(dout["T_mean"],X_al_s(t),color="r",label="spline")

plt.xlabel("T [K]")
plt.ylabel(r"$X_{Si}$")
 
plt.legend()


# %% pour le surf

t[0] = 6
surf[0] = 0




t_surf_mod = t[t>=150]
surf_mod=surf[t>=150]

surf_mod2 = np.hstack(( 0.4e-6,0.5e-6, surf_mod))
t_surf_mod2 = np.hstack(( 0,75, t_surf_mod))

surf_s=UnivariateSpline(x=t_surf_mod2 , 
                        y=surf_mod2 , 
                        w=None, bbox=[None, None], k=3, s=1.6e-13, 
                        ext=0, check_finite=False)




dsurf_dt_s  = surf_s.derivative(n=1)(t)

plt.plot(t*(1/3)+400, surf ,label="brut")
plt.plot(t_surf_mod2*(1/3)+400, surf_mod2,color="g",label="brut modfi")

#plt.plot(t,surf)
plt.plot(T,surf_s(t),color="r",label="spline")

plt.xlabel("T [K]")
plt.ylabel(r"$surf$")

plt.legend()

# %%

deta_dt_s  = eta_s.derivative(n=1)(t)
dX_al_dt_s = X_al_s.derivative(n=1)(t)


cp_fcc = np.zeros(len(T))
cp_dia = np.zeros(len(T))
Hm_dia = np.zeros(len(T))
Hm_fcc = np.zeros(len(T))
Vm_dia = np.zeros(len(T))
Vm_fcc = np.zeros(len(T))
d_Hm_fcc_dSi = np.zeros(len(T))

for i in range(len(T)):
    cp_fcc[i] = gibbs_energy(T=T[i], X_Si=X_al_s(t[i]), output ="cp_fcc" )
    Hm_fcc[i] = gibbs_energy(T=T[i], X_Si=X_al_s(t[i]), output ="Hm_fcc" )
    cp_dia[i] = gibbs_energy(T=T[i], X_Si=0.9999, output ="cp_dia" )
    Hm_dia[i] = gibbs_energy(T=T[i], X_Si=0.9999, output ="Hm_dia" )
    Vm_dia[i] = molar_volume(T0=298, T=T[i],X_Si=0.9999, output ="Vmdia" )
    Vm_fcc[i] = molar_volume(T0=298, T=T[i],X_Si=X_al_s(t[i]), output ="Vmfcc" )
    d_Hm_fcc_dSi[i] = gibbs_energy(T=T[i], X_Si=X_al_s(t[i]), output ="d_Hm_fcc_dSi" )
#
#
T_dot = 20/60   # K.s-1
umAl = 26.98154 # g.mol-1
umSi = 28.0855  # g.mol-1
gam_H = 1.5     # J.m-1


# 

V_sim = (360e-9)**2   # m2  
rho_alloy = eta_s(t)*umSi/Vm_dia + (1-eta_s(t))*umAl/Vm_fcc # g.m-3

 








#dsurf_dt_s[T<430]=0
 


dsc_bkg = T_dot*(eta_s(t)*cp_dia*(1/umSi)+(1-eta_s(t))*cp_fcc*(1/umAl))


dsc_surf = gam_H*dsurf_dt_s/(V_sim*rho_alloy)  # 

dsc_preci = (1-eta_s(t))*d_Hm_fcc_dSi*(1/umAl)*dX_al_dt_s+\
             deta_dt_s*(Hm_dia*(1/umSi))+(-deta_dt_s)*(Hm_fcc*(1/umAl))

 

plt.plot(T,dsc_bkg*3)
plt.plot(T,(dsc_bkg+dsc_preci)*3)

plt.plot(T,(dsc_bkg+dsc_surf)*3)

plt.plot(T,(dsc_bkg+dsc_surf+dsc_preci)*3)

plt.xlim([350,800])


# exporter 

dic_exp={}


dic_exp["T"]=T
dic_exp["surf"]=surf_s(t)/(V_sim*rho_alloy)
dic_exp["X_al_mean"]=X_al_s(t)
dic_exp["eta_mean"]=eta_s(t)
dic_exp["dsc_preci"]=dsc_preci
dic_exp["dsc_surf"]=dsc_surf
dic_exp["dsc_bkg"]=dsc_bkg
title = "3d02a_dsc_mean"

np.savez(
     title              ,
    **dic_exp    
  )  






#%% calcul





#%%

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T443"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T443"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=443 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01a.pdf",dpi =200)





#%% 

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T483"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T483"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.5, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=483 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01b.pdf",dpi =200)

 

#%% 

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T525"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T525"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=525 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01c.pdf",dpi =200)


#%% 

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T566"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T566"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=566 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01d.pdf",dpi =200)



#%% 

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T580"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T580"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=580 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01e.pdf",dpi =200)


#%% 

#print(dic.keys())

python_part = "part1"

bar_width=0.3


if python_part=="part1":
   # mpl.use("pgf")
   # mpl.use("Agg")
   # plt.rcParams.update({"text.usetex": True })
   plt.rcParams.update({"text.usetex": False }) 
   cm_to_inch = 1/2.54
   fig_wid_inch=7.5*cm_to_inch
   golden_ratio = 0.75
   prop = 0.8#0.75
   i=1
   fig = plt.figure(figsize=(fig_wid_inch,fig_wid_inch*golden_ratio))  # sets the window to 8 x 6 inches  if my figure is 7.5cm
   big_ax = fig.add_axes([0, 0, 1, 1])
   small_ax = fig.add_axes([(1-prop)*0.99, (1-prop)*0.99, prop*0.9, prop]) # left, bottom, width, height (range 0 to 1)



   im2=small_ax.imshow(dout["X_T650"].reshape((dout["Nx"] ,dout["Ny"]))*100,extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,2.5) 
   small_ax.contour(dout["eta_T650"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{X_{Si}} \ $"  + r"$\mathrm{[mol.\% ]}$", fontsize=8)
 
   cbar.ax.tick_params(labelsize=8)

   small_ax.set_ylabel(r"$\mathrm{x} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   small_ax.set_xlabel(r"$\mathrm{y} \ $"  + r"$\mathrm{[ nm ]}$", fontsize=10) #+"\n"
   # small_ax.set_xlabel(r"$\mathrm{hkl \ [-]}$")
    
   #bbox_to_anchor=(0.0, 1),
#    small_ax.set_yticks(np.linspace(3,21,7))
# # #    small_ax.set_xticks([1,2,3])

#   small_ax.set_xticks(np.linspace(1,3,3))
#   small_ax.set_xticklabels(["35","165","200"] )
#   small_ax.set_ylim([0,325])
#    # small_ax.set_ylim([4.045,4.057])
#   small_ax.text(0.5,205, r"$\mathrm{eutectic }$",rotation=90,color="k",size=6)
 
   small_ax.text(380,-75, r"$\mathrm{T=650 \ K}$",color="k",size=8)
#    small_ax.text(2.5,4.1, r"$\mathrm{200^{o} \ C}$",color="b",size=6)
#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="s",label="bottom"
#            )

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="d",label="middle"
#            )  

#    small_ax.plot([],
#             [],
#            linestyle=" ",color="k",markeredgecolor="k",markerfacecolor="None",
#            alpha=0.8,marker="o",label="top"
#            )   


   #get handles and labels
   #handles, labels = plt.gca().get_legend_handles_labels()

   #specify order of items in legend
   #order = [0,3,1,4,2]

   # add legend to plot
   # small_ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
   #                 loc='upper right',ncol=3, fontsize=6,frameon=False) 

   # small_ax.legend( loc='upper right',ncol=4, fontsize=6,
   #             #labelcolor=["k","g","r"],
   #             frameon=False)

   big_ax.axis("off")
   big_ax.tick_params(axis='both', which='both', length=0)
   plt.setp(big_ax.get_yticklabels(), visible=False)
   plt.setp(big_ax.get_xticklabels(), visible=False)

   #plt.title(r"$FEM \ calibration \ for \ the\ 4 \ sets \ of \ $"+"\n"+r"$  thermo-physical \ properties$", fontsize=18,pad=15)
   plt.savefig("3f01f.pdf",dpi =200)
