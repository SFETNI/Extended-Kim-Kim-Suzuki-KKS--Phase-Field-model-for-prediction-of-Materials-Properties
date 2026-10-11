import importlib
import sys
import matplotlib.pyplot as plt
import numpy as np


system = "linux"

if system=="linux":
    sys.path.insert(0,"/media/jo/docs/ulieg/fonct_pytho/")


from os import chdir
chdir("/media/jo/docs/ulieg/these/chap3/3_valid/fig/3f01_2dmap/")

import jupyter_style
importlib.reload(jupyter_style)
from jupyter_style import *


cm_to_inch = 1/2.54
fig_wid_inch=7.5*cm_to_inch
golden_ratio = 0.75
prop = 0.75

figLatexSize(fig_width_cm=7.5,fontSize=8,legFontSiz=6,lineSize=1,markerSize=3, golden_mean=prop)


#dic=dict(np.load("4d02_durci_expli.npz"))
#dic=dict(np.load("4d02_durci_expli2.npz"))

#print(dic.keys())
# traitement des données 

#%%

dout=dict(np.load("3d01_2dmap.npz"))

x  = np.linspace(0,dout["Nx"] ,dout["Nx"]  )*dout["dx"]*1e9  


# Set up a regular grid of interpolation points
xi = np.linspace(0,dout["Nx"] ,dout["Nx"] +1)*dout["dx"]*1e9   
yi = np.linspace(0,dout["Ny"] ,dout["Ny"]+1)*dout["dx"]*1e9    
Xi, Yi = np.meshgrid(xi, yi )
extent = np.min(yi), np.max(yi), np.min(xi), np.max(xi)


 



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



   im2=small_ax.imshow(dout["eta_T443"].reshape((dout["Nx"] ,dout["Ny"])),extent=extent)
   cbar=fig.colorbar(im2,shrink = 0.9)
   im2.set_clim(0,1) 
   small_ax.contour(dout["eta_T443"].reshape((dout["Nx"] ,dout["Ny"])), linewidths=0.3, 
                 levels=[0.1,0.9], colors='r', origin='image', extent=extent)
   cbar.set_label(label=r"$\mathrm{\eta} \ $"  + r"$\mathrm{[-]}$", fontsize=8)
 
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
 
#   small_ax.text(380,-75, r"$\mathrm{T=443 \ K}$",color="k",size=8)
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
   plt.savefig("3f04a.pdf",dpi =200)



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
