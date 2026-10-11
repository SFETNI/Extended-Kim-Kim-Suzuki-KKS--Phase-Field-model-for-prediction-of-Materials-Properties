import numpy as np

def atomic_mobility(T,X_Si,output):
    out=0
    # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.
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

    Phi_Al = X_Al*Phi_Al_Al+X_Si*Phi_Al_Si+\
        X_Si*X_Al*_0Phi_Al_AlSi#*(chi_Al_grid-chi_Si_grid)
    
    M_Al = np.exp(Phi_Al/(R*T))*(1/(R*T))
    D_Al_s = R*T*M_Al
    
    # for Si 
    Phi_Si_Al      = -117600.0-93.05*T
    Phi_Si_Si      = -155634.9-81.03*T
    _0Phi_Si_AlSi = -152613.88
    
    Phi_Si=X_Si*Phi_Si_Si+X_Al*Phi_Si_Al+\
        X_Si*X_Al*_0Phi_Si_AlSi#*(chi_Al_grid-chi_Si_grid)
    M_Si = np.exp(Phi_Si/(R*T))*(1/(R*T))
    # output 
    if output=="M_Al":
       out = M_Al
    elif output=="M_Si":
       out = M_Si
    elif output=="LIQ_M_Al":
       out = liq_M_Al
    elif output=="FCC_D_Al_s":
       out = D_Al_s
    return out


def gibbs_energy(T,X_Si,output):
   out=0
   # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.   X_Al = 1-X_Si
   R     = 8.314472                  # = [J.K-1.mol-1]
   
   # so as to X_SI different from 0 or 1
   
   #if X_Si>=1:
   #   X_Si = 0.00000001 
   #elif X_Si<=1:
   #   X_Si = 9.99999999e-1 
   
   X_Al = 1-X_Si
   
   _0G_dia_Si=0
   d_0G_dia_Si_dT = 0
   d2_0G_dia_Si_dT2 = 0
   _0G_fcc_Al = 0
   d_0G_fcc_Al_dT = 0
   d2_0G_fcc_Al_dT2 = 0
   _0G_liq_Si =0
   d_0G_liq_Si_dT = 0
   d2_0G_liq_Si_dT2 = 0
   _0G_liq_Al =0
   d_0G_liq_Al_dT = 0
   d2_0G_liq_Al_dT2 = 0

   #
   # free energy for G0_FCC-A1
   # for AL 
   if T>298.15 and T<=700:
      _0G_fcc_Al=-7976.15+137.093038*T-24.3671976*T*np.log(T)-1.884662e-3*T**2-0.877664e-6*T**3\
        +74092*T**(-1)
      d_0G_fcc_Al_dT= 137.093038-24.3671976*(1+np.log(T))-(2)*1.884662e-3*T-(3)*0.877664e-6*T**2\
        +(-1)*74092*T**(-2)
      d2_0G_fcc_Al_dT2= -24.3671976*(1/T)-(2)*1.884662e-3-(3*2)*0.877664e-6*T\
        +(-1)*(-2)*74092*T**(-3)
   elif T>700 and T<=933.47:
      _0G_fcc_Al=-11276.24+223.048446*T-38.5844296*T*np.log(T)+18.531982e-3*T**2-5.764227e-6*T**3\
        +74092*T**(-1)
      d_0G_fcc_Al_dT=  223.048446-38.5844296*(1+np.log(T))+(2)*18.531982e-3*T-(3)*5.764227e-6*T**2\
        +(-1)*74092*T**(-2) 
      d2_0G_fcc_Al_dT2= -38.5844296*(1/T)+(2)*18.531982e-3-(3)*(2)*5.764227e-6*T\
      +(-1)*(-2)*74092*T**(-3)
   elif T>933.47 and T<2900.0:
      _0G_fcc_Al=-11278.378+188.6841536*T-31.748192*T*np.log(T)-1230.524e25*T**(-9)
      d_0G_fcc_Al_dT=188.6841536-31.748192*(1+np.log(T))-(-9)*1230.524e25*T**(-10) 
      d2_0G_fcc_Al_dT2= -31.748192*(1/T)-(-9)*(-10)*1230.524e25*T**(-11)  
   #
   # free energy for G0_FCC-A1
   # for Si 
   if T>298.15 and T<=1687:
      _0G_dia_Si=-8162.609+137.236859*T-22.8317533*T*np.log(T)-1.912904e-3*T**2-0.003552e-6*T**3\
        +176667*T**(-1)
      d_0G_dia_Si_dT= 137.236859-22.8317533*(1+np.log(T))-(2)*1.912904e-3*T-(3)*0.003552e-6*T**2\
        +(-1)*176667*T**(-2)
      d2_0G_dia_Si_dT2=-22.8317533*(1/T)-(2)*1.912904e-3-(3*2)*0.003552e-6*T\
        +(-1)*(-2)*176667*T**(-3)
   elif T>1687 and T<3600:
      _0G_dia_Si=-9457.642+167.281367*T-27.196*T*np.log(T)-420.369e28*T**(-9)
      d_0G_dia_Si_dT= 167.281367-27.196*(1+np.log(T))-(-9)*420.369e28*T**(-10)
      d2_0G_dia_Si_dT2=-27.196*(1/T)-(-9)*(-10)*420.369e28*T**(-11)

   _0G_fcc_Si     = 51000.00-21.8*T+_0G_dia_Si
   d_0G_fcc_Si_dT = -21.8+d_0G_dia_Si_dT
   d2_0G_fcc_Si_dT2 = d2_0G_dia_Si_dT2
   _0G_dia_Al     = 30*T+_0G_fcc_Al
   d_0G_dia_Al_dT = 30 + d_0G_fcc_Al_dT
   d2_0G_dia_Al_dT2 = d2_0G_fcc_Al_dT2

   # free energy for liquid
   # for Si

   if T>298.15 and T<=1687:
      _0G_liq_Si=50696.4-30.0994*T+2.09307e-21*T**7+_0G_dia_Si
      d_0G_liq_Si_dT= -30.0994+7*2.09307e-21*T**6+d_0G_dia_Si_dT
      d2_0G_liq_Si_dT2=6*7*2.09307e-21*T**5+d2_0G_dia_Si_dT2
   elif T>1687 and T<6000:
      _0G_liq_Si=49828.2-29.5591*T+4.20369e30*T**(-9)+_0G_dia_Si
      d_0G_liq_Si_dT= -29.5591+(-9)*4.20369e30*T**(-10)+d_0G_dia_Si_dT
      d2_0G_liq_Si_dT2=(-10)*4.20369e30*T**(-11)+d2_0G_dia_Si_dT2

   # for Al:


   if T>298.15 and T<=933.47:
      _0G_liq_Al= 11005.029-11.84187*T+7.934e-20*T**7+_0G_fcc_Al
      d_0G_liq_Al_dT=-11.84187+(7)*7.934e-20*T**6 +d_0G_fcc_Al_dT
      d2_0G_liq_Al_dT2=(6*7)*7.934e-20*T**5+d2_0G_fcc_Al_dT2
   elif T>933.47 and T<2900:
      _0G_liq_Al=10482.382-11.253974*T+1.231e+28*T**(-9)+_0G_fcc_Al
      d_0G_liq_Al_dT= -11.253974+(-9)*1.231e+28*T**(-10)+d_0G_fcc_Al_dT
      d2_0G_liq_Al_dT2=(-10)*1.231e+28*T**(-11)+d2_0G_fcc_Al_dT2


   #fcc
   L0_fcc_AlSi     = -3143.78+0.3929*T
   dL0_fcc_AlSi_dT = 0.3929
   d2L0_fcc_AlSi_dT2 = 0
   
   #dia
   L0_dia_AlSi     = 113246.16-47.5551*T 
   dL0_dia_AlSi_dT = -47.5551
   d2L0_dia_AlSi_dT2=0
   
   #liq
   L0_liq_AlSi = -11340.1-1.23394*T
   dL0_liq_AlSi_dT = -1.23394
   d2L0_liq_AlSi_dT2 = 0
   
   L1_liq_AlSi = -3530.93+1.35993*T 
   dL1_liq_AlSi_dT = 1.35993
   d2L1_liq_AlSi_dT2 = 0

   L2_liq_AlSi = 2265.39
   dL2_liq_AlSi_dT = 0
   d2L2_liq_AlSi_dT2 = 0
   
   # free energy for the liquid phase

   Gm_liq = X_Al*_0G_liq_Al+X_Si*_0G_liq_Si+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*(L0_liq_AlSi+L1_liq_AlSi*(X_Al-X_Si)+L2_liq_AlSi*(X_Al-X_Si)**2)

   dGm_liq_dT = X_Al*d_0G_liq_Al_dT+X_Si*d_0G_liq_Si_dT+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
      X_Al*X_Si*(dL0_liq_AlSi_dT+dL1_liq_AlSi_dT*(X_Al-X_Si)+dL2_liq_AlSi_dT*(X_Al-X_Si)**2)
   
   d2Gm_liq_dT2 =  X_Al*d2_0G_liq_Al_dT2+X_Si*d2_0G_liq_Si_dT2+X_Al*X_Si*(d2L0_liq_AlSi_dT2+\
    d2L1_liq_AlSi_dT2*(X_Al-X_Si)+d2L2_liq_AlSi_dT2*(X_Al-X_Si)**2)

   Hm_liq = Gm_liq-T*dGm_liq_dT
   
   dGm_liq_dSi = (-1)*_0G_liq_Al+_0G_liq_Si+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+(-2*X_Si+1)*L0_liq_AlSi\
                 +(6*X_Si**2-6*X_Si+1)*L1_liq_AlSi+\
                 (-16*X_Si**3+24*X_Si**2-10*X_Si+1)*L2_liq_AlSi # = to check because not good  old (-6*X_Si**5+5*X_Si**4+8*X_Si**3-6*X_Si**2-2*X_Si+1)
   
   d2Gm_liq_dSi2 = R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_liq_AlSi+(12*X_Si-6)*L1_liq_AlSi+(-48*X_Si**2+48*X_Si-10)*L2_liq_AlSi

   d2Gm_liq_dTSi = (-1)*d_0G_liq_Al_dT+d_0G_liq_Si_dT+R*(-1-np.log(1-X_Si)+1+np.log(X_Si))+(-2*X_Si+1)*dL0_liq_AlSi_dT\
                 +(6*X_Si**2-6*X_Si+1)*dL1_liq_AlSi_dT+(-6*X_Si**5+5*X_Si**4+8*X_Si**3-6*X_Si**2-2*X_Si+1)*dL2_liq_AlSi_dT

   d_Hm_liq_dSi = dGm_liq_dSi-T*d2Gm_liq_dTSi
   
   #
   # free energy for fcc phase
   #
   Gm_fcc = X_Al*_0G_fcc_Al+X_Si*_0G_fcc_Si+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*L0_fcc_AlSi
   # derivative with respect to T
   dGm_fcc_dT    = X_Al*d_0G_fcc_Al_dT+X_Si*d_0G_fcc_Si_dT+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*dL0_fcc_AlSi_dT

   d2Gm_fcc_dT2    = X_Al*d2_0G_fcc_Al_dT2+X_Si*d2_0G_fcc_Si_dT2+\
     X_Al*X_Si*d2L0_fcc_AlSi_dT2

   d2Gm_fcc_dTSi = (-1)*d_0G_fcc_Al_dT+d_0G_fcc_Si_dT+R*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
     (-2*X_Si+1)*dL0_fcc_AlSi_dT

   
   dGm_fcc_dSi = (-1)*_0G_fcc_Al+_0G_fcc_Si+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+(-2*X_Si+1)*L0_fcc_AlSi


   d2Gm_fcc_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_fcc_AlSi  # good
   dGm_fcc_dAl   = -dGm_fcc_dSi
   d2Gm_fcc_dAl2 = d2Gm_fcc_dSi2
   d2Gm_fcc_dSiAl = -d2Gm_fcc_dSi2
   d2Gm_fcc_dAlSi = -d2Gm_fcc_dSi2
   # molar enthalpy of fcc

   Hm_fcc       = Gm_fcc-T*dGm_fcc_dT
   d_Hm_fcc_dSi = dGm_fcc_dSi-T*d2Gm_fcc_dTSi

   #
   # free energy for diamond phase
   Gm_dia = X_Al*_0G_dia_Al+X_Si*_0G_dia_Si+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*L0_dia_AlSi    # ok
   # derivative with respect to T
   dGm_dia_dT = X_Al*d_0G_dia_Al_dT+X_Si*d_0G_dia_Si_dT+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*dL0_dia_AlSi_dT    # ok
   d2Gm_dia_dT2 = X_Al*d2_0G_dia_Al_dT2+X_Si*d2_0G_dia_Si_dT2+\
     X_Al*X_Si*d2L0_dia_AlSi_dT2    # ok  

   # its 1st and 2nd derivative with respect to Si
   dGm_dia_dSi  = (-1)*_0G_dia_Al+_0G_dia_Si+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
   (-2*X_Si+1)*L0_dia_AlSi    # ok
   
   # Heat capacity

   cp_liq =-T*d2Gm_liq_dT2
   cp_dia = -T*d2Gm_dia_dT2
   cp_fcc =-T*d2Gm_fcc_dT2

   d_Hm_fcc_dT  = cp_fcc
   d_Hm_dia_dT  = cp_dia

   # molar enthalpy of diam

   Hm_dia = Gm_dia-T*dGm_dia_dT


   d2Gm_dia_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_dia_AlSi
   dGm_dia_dAl   = -dGm_dia_dSi
   d2Gm_dia_dAl2 = d2Gm_dia_dSi2
   d2Gm_dia_dSiAl = -d2Gm_dia_dSi2
   d2Gm_dia_dAlSi = -d2Gm_dia_dSi2

   dGm_liq_dAl = -dGm_liq_dSi
   d2Gm_liq_dAlSi = -d2Gm_liq_dSi2
   #dGm_dia_dAl = _0G_dia_Al+(-1)*_0G_dia_Si+R*T*(1+np.log(X_Al)-1-np.log(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #d2Gm_dia_dAl2 =  R*T*(1/X_Al+1/(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #
   #
   
   # chemical potential 
   # for liquid
   mu_liq_Si =  dGm_liq_dSi*(1-X_Si)+Gm_liq
   mu_liq_Al =  dGm_liq_dAl*(1-X_Al)+Gm_liq

   dmu_liq_Si_dSi = d2Gm_liq_dSi2*(1-X_Si)
   dmu_liq_Al_dSi = d2Gm_liq_dAlSi*(1-X_Al)+dGm_liq_dAl*1+dGm_liq_dSi
   # for dia
   mu_dia_Si =  dGm_dia_dSi*(1-X_Si)+Gm_dia                            #  ok
   mu_dia_Al =  dGm_dia_dAl*(1-X_Al)+Gm_dia                            # ok
   
   dmu_dia_Si_dSi = d2Gm_dia_dSi2*(1-X_Si)+dGm_dia_dSi*(-1)+dGm_dia_dSi
   dmu_dia_Al_dSi = d2Gm_dia_dAlSi*(1-X_Al)+dGm_dia_dAl*1+dGm_dia_dSi

   # chemical potential  
   mu_fcc_Si      = dGm_fcc_dSi*(1-X_Si)+Gm_fcc                           # ok
   dmu_fcc_Si_dSi =  d2Gm_fcc_dSi2*(1-X_Si)+dGm_fcc_dSi*(-1)+dGm_fcc_dSi  # ok
   dmu_fcc_Si_dAl =  d2Gm_fcc_dSiAl*(1-X_Si)+dGm_fcc_dSi*1+dGm_fcc_dAl    #+dGm_fcc_dSi+dGm_fcc_dAl
   #dmu_fcc_SidAl =  -dmu_fcc_SidSi 

   mu_fcc_Al      = dGm_fcc_dAl*(1-X_Al)+Gm_fcc                              
   #
   #dmu_fcc_AldSi = d2Gm_fcc_dSiAl+dGm_fcc_dAl
   #dmu_fcc_AldAl = d2Gm_fcc_dAl2-(dGm_fcc_dAl+X_Al*d2Gm_fcc_dAl2)+dGm_fcc_dAl
   dmu_fcc_Al_dAl = d2Gm_fcc_dAl2*(1-X_Al)+dGm_fcc_dAl*(-1)+dGm_fcc_dAl
   dmu_fcc_Al_dSi = d2Gm_fcc_dAlSi*(1-X_Al)+dGm_fcc_dAl*1+dGm_fcc_dSi#-dmu_fcc_Al_dAl# ok
   
   #
   # derivatives of chemical potential
   # dmu_fcc_Si_dSi =  d2Gm_fcc_dSi2-(dGm_fcc_dSi+X_Si*d2Gm_fcc_dSi2)+dGm_fcc_dSi # ok new activation
   # derivatives of chemical potential
   dmu_dia_Si_dSi =  d2Gm_dia_dSi2*(1-X_Si)
   #dmu_dia_SidSi =  dGm_dia_dSi-(dGm_dia_dSi+X_Si*d2Gm_dia_dSi2)+dGm_dia_dSi 
   dmu_fdia_Si_dAl =  -dmu_dia_Si_dSi # ok

   if output=="Gm_fcc":
      out = Gm_fcc
   elif output=="Hm_fcc":
      out = Hm_fcc
   elif output=="d_Hm_fcc_dSi":
      out = d_Hm_fcc_dSi
   elif output=="d_Hm_liq_dSi":
      out = d_Hm_liq_dSi  
   elif output=="d_Hm_dia_dSi":
      out = d_Hm_dia_dSi   
   elif output=="dGm_fcc_dSi":
      out = dGm_fcc_dSi
   elif output=="d2Gm_fcc_dSi2":
      out = d2Gm_fcc_dSi2
   elif output=="dGm_fcc_dAl":
      out = dGm_fcc_dAl
   elif output=="d2Gm_fcc_dAl2":
      out = d2Gm_fcc_dAl2
   elif output=="cp_fcc":
      out = cp_fcc
   elif output=="cp_liq":
      out = cp_liq
   elif output=="cp_dia":
      out = cp_dia
   elif output=="Gm_dia":
      out = Gm_dia
   elif output=="Hm_dia":
      out = Hm_dia
   elif output=="dmu_liq_Si_dSi":
      out = dmu_liq_Si_dSi
   elif output=="dmu_liq_Al_dSi":
      out = dmu_liq_Al_dSi
   elif output=="dGm_dia_dSi":
      out = dGm_dia_dSi
   elif output=="d2Gm_fcc_dSiAl":
      out = d2Gm_fcc_dSiAl
   elif output=="mu_fcc_Si":
      out = mu_fcc_Si
   elif output=="mu_liq_Si":
      out = mu_liq_Si
   elif output=="mu_liq_Al":
      out = mu_liq_Al   
   elif output=="dmu_fcc_Si_dSi":
      out = dmu_fcc_Si_dSi
   elif output=="dmu_fcc_Al_dSi":
      out = dmu_fcc_Al_dSi
   elif output=="dmu_dia_Si_dSi":
      out = dmu_dia_Si_dSi
   elif output=="dmu_dia_Al_dSi":
      out = dmu_dia_Al_dSi
   elif output=="mu_fcc_Al":
      out = mu_fcc_Al
   elif output=="mu_dia_Si":
      out = mu_dia_Si
   elif output=="mu_dia_Al":
      out = mu_dia_Al
   #elif output=="dmu_fcc_Si_dAl":
   #   out = dmu_fcc_Si_dAl
   #elif output=="dmu_fcc_Al_dAl":
   #   out = dmu_fcc_AldAl
   elif output=="Hm_liq":
      out=Hm_liq
   elif output=="Gm_liq":
      out=Gm_liq
   elif output=="dGm_liq_dSi":
      out=dGm_liq_dSi
   return out





def gibbs_energy_updated(T,X_Al,X_Si,X_Mg,X_Fe,output):
   out=0
   # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.   X_Al = 1-X_Si
   R     = 8.314472                  # = [J.K-1.mol-1]
   #X_Al = 1-(X_Si+X_Mg+X_Fe)
   #
   # free energy for G0_FCC-A1
   




   # for beta" Mg5Si6

   a_Mg5Si6=-5000.         # = T**0
   b_Mg5Si6=-30           # = T**1
   c_Mg5Si6= -0.0096    # = T**2
   d_Mg5Si6= 0              # = T**(-2)
   e_Mg5Si6=0       # = Tln(T)
   f_Mg5Si6=-1e-7    # = T**3
   g_Mg5Si6 = 0        # = T**(-1)
   h_Mg5Si6 = 0             # = T**(-9)

   _0G_Mg5Si6=a_Mg5Si6+b_Mg5Si6*T+c_Mg5Si6*T**2+d_Mg5Si6*T**(-2)+e_Mg5Si6*T*np.log(T)+f_Mg5Si6*T**3\
     +g_Mg5Si6*T**(-1)+h_Mg5Si6*T**(-9)
   d_0G_Mg5Si6_dT= b_Mg5Si6+(2)*c_Mg5Si6*T+e_Mg5Si6*(1+np.log(T))+(3)* f_Mg5Si6*T**2\
     +(-1)*g_Mg5Si6*T**(-2)+(-9)*h_Mg5Si6*T**(-10) 
   d2_0G_Mg5Si6_dT2= (2)*c_Mg5Si6+e_Mg5Si6*(1/T)+(3*2)*f_Mg5Si6*T\
     +(-1)*(-2)*g_Mg5Si6*T**(-3)+(-9)*(-10)*h_Mg5Si6*T**(-11)  

   # for beta Mg2si

   a_Mg2Si=-92250.0         # = T**0
   b_Mg2Si=440.4       # = T**1
   c_Mg2Si= -0.0018    # = T**2
   d_Mg2Si= 0              # = T**(-2)
   e_Mg2Si=-75.9        # = Tln(T)
   f_Mg2Si=0    # = T**3
   g_Mg2Si = 630000         # = T**(-1)
   h_Mg2Si = 0             # = T**(-9)

   _0G_Mg2Si=a_Mg2Si+b_Mg2Si*T+c_Mg2Si*T**2+d_Mg2Si*T**(-2)+e_Mg2Si*T*np.log(T)+f_Mg2Si*T**3\
     +g_Mg2Si*T**(-1)+h_Mg2Si*T**(-9)
   d_0G_Mg2Si_dT= b_Mg2Si+(2)*c_Mg2Si*T+e_Mg2Si*(1+np.log(T))+(3)* f_Mg2Si*T**2\
     +(-1)*g_Mg2Si*T**(-2)+(-9)*h_Mg2Si*T**(-10) 
   d2_0G_Mg2Si_dT2= (2)*c_Mg2Si+e_Mg2Si*(1/T)+(3*2)*f_Mg2Si*T\
     +(-1)*(-2)*g_Mg2Si*T**(-3)+(-9)*(-10)*h_Mg2Si*T**(-11)  


   # for beta ' Mg1_8Si

   a_Mg1_8Si= 24250         # = T**0
   b_Mg1_8Si= -40.4      # = T**1
   c_Mg1_8Si= -0.0042    # = T**2
   d_Mg1_8Si=  0              # = T**(-2)
   e_Mg1_8Si=  5.9       # = Tln(T)
   f_Mg1_8Si=  0  # = T**3
   g_Mg1_8Si =  -130000        # = T**(-1)
   h_Mg1_8Si =  0             # = T**(-9)

   _0G_Mg1_8Si=a_Mg1_8Si+b_Mg1_8Si*T+c_Mg1_8Si*T**2+d_Mg1_8Si*T**(-2)+e_Mg1_8Si*T*np.log(T)+f_Mg1_8Si*T**3\
     +g_Mg1_8Si*T**(-1)+h_Mg1_8Si*T**(-9)+_0G_Mg2Si
   d_0G_Mg1_8Si_dT= b_Mg1_8Si+(2)*c_Mg1_8Si*T+e_Mg1_8Si*(1+np.log(T))+(3)* f_Mg1_8Si*T**2\
     +(-1)*g_Mg1_8Si*T**(-2)+(-9)*h_Mg1_8Si*T**(-10)+d_0G_Mg2Si_dT 
   d2_0G_Mg1_8Si_dT2= (2)*c_Mg1_8Si+e_Mg1_8Si*(1/T)+(3*2)*f_Mg1_8Si*T\
     +(-1)*(-2)*g_Mg1_8Si*T**(-3)+(-9)*(-10)*h_Mg1_8Si*T**(-11)+d2_0G_Mg2Si_dT2  

   # for Mg hcp-A3(paramagnetic)
   if T>298.15 and T<=923.00:
      a_hcpMg=-8367.34          # = T**0
      b_hcpMg=+143.675547        # = T**1
      c_hcpMg= 0.4858e-3    # = T**2
      d_hcpMg= 0              # = T**(-2)
      e_hcpMg=-26.1849782        # = Tln(T)
      f_hcpMg=-1.393669e-6    # = T**3
      g_hcpMg = 78950         # = T**(-1)
      h_hcpMg = 0             # = T**(-9)
      #
      a_fccMg=2600          # = T**0
      b_fccMg=-0.90        # = T**1
      c_fccMg= 0   # = T**2
      d_fccMg= 0              # = T**(-2)
      e_fccMg=0       # = Tln(T)
      f_fccMg=0    # = T**3
      g_fccMg = 0      # = T**(-1)
      h_fccMg = 0             # = T**(-9)


   if T>923.00 and T<3000.00:
      a_hcpMg=-14130.185         # = T**0
      b_hcpMg=204.716215        # = T**1
      c_hcpMg= 0   # = T**2
      d_hcpMg= 0              # = T**(-2)
      e_hcpMg=-34.3088       # = Tln(T)
      f_hcpMg=0    # = T**3
      g_hcpMg = 0        # = T**(-1)
      h_hcpMg = 1038.192e25             # = T**(-9)
      #
      a_fccMg=2600          # = T**0
      b_fccMg=-0.90        # = T**1
      c_fccMg= 0   # = T**2
      d_fccMg= 0              # = T**(-2)
      e_fccMg=0       # = Tln(T)
      f_fccMg=0    # = T**3
      g_fccMg = 0      # = T**(-1)
      h_fccMg = 0             # = T**(-9)


   _0G_hcp_Mg=a_hcpMg+b_hcpMg*T+c_hcpMg*T**2+d_hcpMg*T**(-2)+e_hcpMg*T*np.log(T)+f_hcpMg*T**3\
     +g_hcpMg*T**(-1)+h_hcpMg*T**(-9)
   d_0G_hcp_Mg_dT= b_hcpMg+(2)*c_hcpMg*T+e_hcpMg*(1+np.log(T))+(3)* f_hcpMg*T**2\
     +(-1)*g_hcpMg*T**(-2)+(-9)*h_hcpMg*T**(-10) 
   d2_0G_hcp_Mg_dT2= (2)*c_hcpMg+e_hcpMg*(1/T)+(3*2)*f_hcpMg*T\
     +(-1)*(-2)*g_hcpMg*T**(-3)+(-9)*(-10)*h_hcpMg*T**(-11)  

   _0G_fcc_Mg=a_fccMg+b_fccMg*T+c_fccMg*T**2+d_fccMg*T**(-2)+e_fccMg*T*np.log(T)+f_fccMg*T**3\
     +g_fccMg*T**(-1)+h_fccMg*T**(-9)+_0G_hcp_Mg
   d_0G_fcc_Mg_dT= b_fccMg+(2)*c_fccMg*T+e_fccMg*(1+np.log(T))+(3)* f_fccMg*T**2\
     +(-1)*g_fccMg*T**(-2)+(-9)*h_fccMg*T**(-10)+d_0G_hcp_Mg_dT 
   d2_0G_fcc_Mg_dT2= (2)*c_fccMg+e_fccMg*(1/T)+(3*2)*f_fccMg*T\
     +(-1)*(-2)*g_fccMg*T**(-3)+(-9)*(-10)*h_fccMg*T**(-11)+d2_0G_hcp_Mg_dT2  

   # for Fe bcc-A2(paramagnetic)

   if T>298.15 and T<=1811.00:
      a_bccFe=1225.7          # = T**0   
      b_bccFe=124.134         # = T**1   
      c_bccFe= -4.39752e-3    # = T**2   
      d_bccFe= 0              # = T**(-2)
      e_bccFe=-23.5143        # = Tln(T) 
      f_bccFe=-0.058927e-6    # = T**3   
      g_bccFe = 77359         # = T**(-1)
      h_bccFe = 0             # = T**(-9)
      #
      a_fccFe= -1462.4          # = T**0
      b_fccFe= 8.8282        # = T**1
      c_fccFe= 0.00064     # = T**2
      d_fccFe= 0              # = T**(-2)
      e_fccFe= -1.15        # = Tln(T)
      f_fccFe= 0    # = T**3
      g_fccFe = 0         # = T**(-1)
      h_fccFe = 0             # = T**(-9)

   if T>1811.00 and T<6000.00:
      a_bccFe=-25383.581          # = T**0
      b_bccFe=299.31255         # = T**1
      c_bccFe= 0   # = T**2
      d_bccFe= 0              # = T**(-2)
      e_bccFe=  -46        # = Tln(T)
      f_bccFe=  0    # = T**3
      g_bccFe = 0         # = T**(-1)
      h_bccFe = 2296.03e28             # = T**(-9)
      #
      a_fccFe= -713.815         # = T**0
      b_fccFe= 0.94001        # = T**1
      c_fccFe= 0    # = T**2
      d_fccFe= 0              # = T**(-2)
      e_fccFe= 0       # = Tln(T)
      f_fccFe= 0    # = T**3
      g_fccFe = 0         # = T**(-1)
      h_fccFe = 0.49251             # = T**(-9)

   _0G_bcc_Fe=a_bccFe+b_bccFe*T+c_bccFe*T**2+d_bccFe*T**(-2)+e_bccFe*T*np.log(T)+f_bccFe*T**3\
     +g_bccFe*T**(-1)+h_bccFe*T**(-9)
   d_0G_bcc_Fe_dT= b_bccFe+(2)*c_bccFe*T+e_bccFe*(1+np.log(T))+(3)* f_bccFe*T**2\
     +(-1)*g_bccFe*T**(-2)+(-9)*h_bccFe*T**(-10) 
   d2_0G_bcc_Fe_dT2= (2)*c_bccFe+e_bccFe*(1/T)+(3*2)*f_bccFe*T\
     +(-1)*(-2)*g_bccFe*T**(-3)+(-9)*(-10)*h_bccFe*T**(-11)  

   _0G_fcc_Fe=a_fccFe+b_fccFe*T+c_fccFe*T**2+d_fccFe*T**(-2)+e_fccFe*T*np.log(T)+f_fccFe*T**3\
     +g_fccFe*T**(-1)+h_fccFe*T**(-9)+_0G_bcc_Fe
   d_0G_fcc_Fe_dT= b_fccFe+(2)*c_fccFe*T+e_fccFe*(1+np.log(T))+(3)* f_fccFe*T**2\
     +(-1)*g_fccFe*T**(-2)+(-9)*h_fccFe*T**(-10)+d_0G_bcc_Fe_dT 
   d2_0G_fcc_Fe_dT2= (2)*c_fccFe+e_fccFe*(1/T)+(3*2)*f_fccFe*T\
     +(-1)*(-2)*g_fccFe*T**(-3)+(-9)*(-10)*h_fccFe*T**(-11)+d2_0G_bcc_Fe_dT2

   # for AL liq
   if T>298.15 and T<=933.47:
      a_liqAl=11005.029              # = T**0     
      b_liqAl=-11.841867            # = T**1    
      c_liqAl= 0         # = T**2     
      d_liqAl= 0                    # = T**(-2)  
      e_liqAl=0          # = Tln(T) 
      f_liqAl=0          # = T**3    
      g_liqAl = 0               # = T**(-1)
      h_liqAl = 0                   # = T**(-9)
      i_liqAl = 7.934e-20 # = T**(7)
   elif T>933.47 and T<=2900.00:
      a_liqAl=10482.382              # = T**0     
      b_liqAl=-11.253974            # = T**1    
      c_liqAl= 0         # = T**2     
      d_liqAl= 0                    # = T**(-2)  
      e_liqAl=0          # = Tln(T) 
      f_liqAl=0          # = T**3    
      g_liqAl = 0               # = T**(-1)
      h_liqAl = 1.231e28                   # = T**(-9)
      i_liqAl = 0         # = T**(7)

   # for AL 
   if T>298.15 and T<=700:
      a_fccAl=-7976.15              # = T**0     
      b_fccAl=137.093038            # = T**1    
      c_fccAl= -1.884662e-3         # = T**2     
      d_fccAl= 0                    # = T**(-2)  
      e_fccAl=-24.3671976           # = Tln(T) 
      f_fccAl=-0.877664e-6          # = T**3    
      g_fccAl = 74092               # = T**(-1)
      h_fccAl = 0                   # = T**(-9)    
   elif T>700 and T<=933.47:
      a_fccAl=-11276.24
      b_fccAl=223.048446
      c_fccAl= 18.531982e-3
      d_fccAl= 0
      e_fccAl=-38.5844296
      f_fccAl=-5.764227e-6
      g_fccAl = 74092
      h_fccAl = 0
   elif T>933.47 and T<2900.0:
      a_fccAl=-11278.378
      b_fccAl=188.6841536
      c_fccAl= 0
      d_fccAl= 0
      e_fccAl=-31.748192
      f_fccAl=0
      g_fccAl = 0
      h_fccAl = -1230.524e25
   
   _0G_fcc_Al=a_fccAl+b_fccAl*T+c_fccAl*T**2+d_fccAl*T**(-2)+e_fccAl*T*np.log(T)+f_fccAl*T**3\
     +g_fccAl*T**(-1)+h_fccAl*T**(-9)
   d_0G_fcc_Al_dT= b_fccAl+(2)*c_fccAl*T+e_fccAl*(1+np.log(T))+(3)* f_fccAl*T**2\
     +(-1)*g_fccAl*T**(-2)+(-9)*h_fccAl*T**(-10) 
   d2_0G_fcc_Al_dT2= (2)*c_fccAl+e_fccAl*(1/T)+(3*2)*f_fccAl*T\
     +(-1)*(-2)*g_fccAl*T**(-3)+(-9)*(-10)*h_fccAl*T**(-11)  

   _0G_liq_Al= a_liqAl+b_liqAl*T+h_liqAl*T**(-9)+i_liqAl*T**(7)+_0G_fcc_Al
   d_0G_liq_Al_dT = b_liqAl+(-9)*h_liqAl*T**(-10)+7*i_liqAl*T**(6)+d_0G_fcc_Al_dT
   d2_0G_liq_Al_dT2 =(-9)*(-10)*h_liqAl*T**(-11)+7*6*i_liqAl*T**(5)+d2_0G_fcc_Al_dT2
   #
   # free energy for G0_FCC-A1
   # for Si 
   if T>298.15 and T<=1687:
      a_diaSi=-8162.609         # = T**0   
      b_diaSi=137.236859        # = T**1   
      c_diaSi= -1.912904e-3     # = T**2    
      d_diaSi= 0                # = T**(-2)
      e_diaSi=-22.8317533       # = Tln(T) 
      f_diaSi=-0.003552e-6      # = T**3   
      g_diaSi = 176667          # = T**(-1)
      h_diaSi = 0               # = T**(-9)
   elif T>1687 and T<3600:
      a_diaSi=-9457.642
      b_diaSi=167.281367
      c_diaSi= 0
      d_diaSi= 0
      e_diaSi=-27.196
      f_diaSi= 0
      g_diaSi = 0
      h_diaSi = -420.369e28

   _0G_dia_Si=a_diaSi+b_diaSi*T+c_diaSi*T**2+d_diaSi*T**(-2)+e_diaSi*T*np.log(T)+f_diaSi*T**3\
     +g_diaSi*T**(-1)+h_diaSi*T**(-9)
   d_0G_dia_Si_dT= b_diaSi+(2)*c_diaSi*T+e_diaSi*(1+np.log(T))+(3)* f_diaSi*T**2\
     +(-1)*g_diaSi*T**(-2)+(-9)*h_diaSi*T**(-10) 
   d2_0G_dia_Si_dT2= (2)*c_diaSi+e_diaSi*(1/T)+(3*2)*f_diaSi*T\
     +(-1)*(-2)*g_diaSi*T**(-3)+(-9)*(-10)*h_diaSi*T**(-11)  


   # for Si liq
   if T>298.15 and T<=1687:
      a_liqSi=50696.4              # = T**0     
      b_liqSi=-30.0994           # = T**1    
      c_liqSi= 0         # = T**2     
      d_liqSi= 0                    # = T**(-2)  
      e_liqSi=0          # = Tln(T) 
      f_liqSi=0          # = T**3    
      g_liqSi = 0               # = T**(-1)
      h_liqSi = 0                   # = T**(-9)
      i_liqSi = 2.09607e-21 # = T**(7)
   elif T>1687 and T<=6000.00:
      a_liqSi=49828.2              # = T**0     
      b_liqSi=-29.5591            # = T**1    
      c_liqSi= 0         # = T**2     
      d_liqSi= 0                    # = T**(-2)  
      e_liqSi=0          # = Tln(T) 
      f_liqSi=0          # = T**3    
      g_liqSi = 0               # = T**(-1)
      h_liqSi = 420369e30                   # = T**(-9)
      i_liqSi = 0 # = T**(7)

   

   _0G_liq_Si= a_liqSi+b_liqSi*T+h_liqSi*T**(-9)+i_liqSi*T**(7)+_0G_dia_Si
   d_0G_liq_Si_dT = b_liqSi+(-9)*h_liqSi*T**(-10)+7*i_liqSi*T**(6)+d_0G_dia_Si_dT
   d2_0G_liq_Si_dT2 =(-9)*(-10)*h_liqSi*T**(-11)+7*6*i_liqSi*T**(5)+d2_0G_dia_Si_dT2

   # for beta AlSiFe

   a_Al15Si3Fe3=-391310.9         # = T**0
   b_Al15Si3Fe3=55.84756           # = T**1
   c_Al15Si3Fe3= 0    # = T**2
   d_Al15Si3Fe3= 0              # = T**(-2)
   e_Al15Si3Fe3=0       # = Tln(T)
   f_Al15Si3Fe3=0   # = T**3
   g_Al15Si3Fe3 = 0        # = T**(-1)
   h_Al15Si3Fe3 = 0             # = T**(-9)

   _0G_Al14Si3Fe3=a_Al15Si3Fe3+b_Al15Si3Fe3*T+c_Al15Si3Fe3*T**2+d_Al15Si3Fe3*T**(-2)+e_Al15Si3Fe3*T*np.log(T)+f_Al15Si3Fe3*T**3\
     +g_Al15Si3Fe3*T**(-1)+h_Al15Si3Fe3*T**(-9)+14*_0G_fcc_Al+3*_0G_bcc_Fe+3*_0G_dia_Si
   d_0G_Al14Si3Fe3_dT= b_Al15Si3Fe3+(2)*c_Al15Si3Fe3*T+e_Al15Si3Fe3*(1+np.log(T))+(3)* f_Al15Si3Fe3*T**2\
     +(-1)*g_Al15Si3Fe3*T**(-2)+(-9)*h_Al15Si3Fe3*T**(-10)+14*d_0G_fcc_Al_dT+3*d_0G_bcc_Fe_dT+3*d_0G_dia_Si_dT 
   d2_0G_Al14Si3Fe3_dT2= (2)*c_Al15Si3Fe3+e_Al15Si3Fe3*(1/T)+(3*2)*f_Al15Si3Fe3*T\
     +(-1)*(-2)*g_Al15Si3Fe3*T**(-3)+(-9)*(-10)*h_Al15Si3Fe3*T**(-11)+14*d2_0G_fcc_Al_dT2+3*d2_0G_bcc_Fe_dT2+3*d2_0G_dia_Si_dT2  

   _0G_fcc_Si     = 51000.00-21.8*T+_0G_dia_Si
   d_0G_fcc_Si_dT = -21.8+d_0G_dia_Si_dT
   d2_0G_fcc_Si_dT2 = d2_0G_dia_Si_dT2
   _0G_dia_Al     = 30*T+_0G_fcc_Al
   d_0G_dia_Al_dT = 30 + d_0G_fcc_Al_dT
   d2_0G_dia_Al_dT2 = d2_0G_fcc_Al_dT2

   # liq phase
   L0_liq_AlSi = -11340.1-1.23394*T 
   L1_liq_AlSi = -3530.93+1.35993*T
   L2_liq_AlSi = 2265.39
   dL0_liq_AlSi_dT = -1.23394
   dL1_liq_AlSi_dT = 1.35993
   d2L0_liq_AlSi_dT2 = 0
   d2L1_liq_AlSi_dT2 = 0   


   # fcc phase
   L0_fcc_AlSi     = -3143.78+0.3929*T
   L0_fcc_AlMg     = 4971-3.5*T
   L1_fcc_AlMg     = 900+0.423*T 
   L2_fcc_AlMg     = 950 
   L0_fcc_AlFe     = -76066.1+18.6758*T
   L1_fcc_AlFe     = 21167.4+1.3398*T
   L0_fcc_SiMg     =   -7148.79+0.89361*T
   L0_fcc_SiFe     =   -125247.7+41.166*T 
   L1_fcc_SiFe     =   -142707.6
   L2_fcc_SiFe     =   89907.3 
   L0_fcc_MgFe     =   65200.0 


   dL0_fcc_AlSi_dT = 0.3929
   dL0_fcc_AlMg_dT = -3.5
   dL1_fcc_AlMg_dT = 0.423
   dL2_fcc_AlMg_dT = 0
   dL0_fcc_AlFe_dT = 18.6758
   dL1_fcc_AlFe_dT = 1.3398
   dL0_fcc_SiMg_dT = 0.89361
   dL0_fcc_SiFe_dT = 41.166
   dL1_fcc_SiFe_dT = 0
   dL2_fcc_SiFe_dT = 041.166
   dL0_fcc_MgFe_dT = 0

   d2L0_fcc_AlSi_dT2 = 0
   L0_dia_AlSi     = 113246.16-47.5551*T 
   dL0_dia_AlSi_dT = -47.5551
   d2L0_dia_AlSi_dT2=0
   
   Gm_Mg2Si=_0G_Mg2Si
   dGm_Mg2Si_dT=d_0G_Mg2Si_dT
   d2Gm_Mg2Si_dT2=d2_0G_Mg2Si_dT2

   Gm_Mg1_8Si=_0G_Mg1_8Si
   dGm_Mg1_8Si_dT=d_0G_Mg1_8Si_dT
   d2Gm_Mg1_8Si_dT2=d2_0G_Mg1_8Si_dT2   
   Hm_Mg1_8Si       = Gm_Mg1_8Si-T*dGm_Mg1_8Si_dT

   Gm_Mg5Si6=_0G_Mg5Si6
   dGm_Mg5Si6_dT=d_0G_Mg5Si6_dT
   d2Gm_Mg5Si6_dT2=d2_0G_Mg5Si6_dT2

   Hm_Mg5Si6       = Gm_Mg5Si6-T*dGm_Mg5Si6_dT

   # molar enthalpy of fcc
   Hm_Mg2Si       = Gm_Mg2Si-T*dGm_Mg2Si_dT

   Hm_Al14Si3Fe3       = _0G_Al14Si3Fe3-T*d_0G_Al14Si3Fe3_dT

   # free energy for liq phase
   Gm_liq = X_Al*_0G_liq_Al+X_Si*_0G_liq_Si+ R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
    X_Al*X_Si*(L0_liq_AlSi+L1_liq_AlSi*(X_Al-X_Si)+L2_liq_AlSi*(X_Al-X_Si)**2)

   d2Gm_liq_dT2    = X_Al*d2_0G_liq_Al_dT2+X_Si*d2_0G_liq_Si_dT2+\
     X_Al*X_Si*d2L0_liq_AlSi_dT2#+d2L0_liq_AlSi_dT2*(X_Al-X_Si)

   cp_liq = -T*d2Gm_liq_dT2

   #
   # free energy for fcc phase
   #
   Gm_fcc = X_Al*_0G_fcc_Al+X_Si*_0G_fcc_Si+X_Mg*_0G_fcc_Mg+X_Fe*_0G_fcc_Fe+\
          R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si)+X_Mg*np.log(X_Mg)+X_Fe*np.log(X_Fe))+\
     X_Al*X_Si*L0_fcc_AlSi+\
     X_Al*X_Mg*(L0_fcc_AlMg+L1_fcc_AlMg*(X_Al-X_Mg)+L2_fcc_AlMg*(X_Al-X_Mg)**2)+\
     X_Al*X_Fe*(L0_fcc_AlFe+L1_fcc_AlFe*(X_Al-X_Fe))+\
     X_Si*X_Mg*L0_fcc_SiMg+\
     X_Si*X_Fe*(L0_fcc_SiFe+L1_fcc_SiFe*(X_Si-X_Fe)+L2_fcc_SiFe*(X_Si-X_Fe)**2 )+\
     X_Mg*X_Fe*L0_fcc_MgFe

   # derivative with respect to T
   dGm_fcc_dT    = X_Al*d_0G_fcc_Al_dT+X_Si*d_0G_fcc_Si_dT+X_Mg*d_0G_fcc_Mg_dT+X_Fe*d_0G_fcc_Fe_dT+\
          R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si)+X_Mg*np.log(X_Mg)+X_Fe*np.log(X_Fe))+\
     X_Al*X_Si*dL0_fcc_AlSi_dT+\
     X_Al*X_Mg*(dL0_fcc_AlMg_dT+dL1_fcc_AlMg_dT*(X_Al-X_Mg)+dL2_fcc_AlMg_dT*(X_Al-X_Mg)**2)+\
     X_Al*X_Fe*(dL0_fcc_AlFe_dT+dL1_fcc_AlFe_dT*(X_Al-X_Fe))+\
     X_Si*X_Mg*dL0_fcc_SiMg_dT+\
     X_Si*X_Fe*(dL0_fcc_SiFe_dT+dL1_fcc_SiFe_dT*(X_Si-X_Fe)+dL2_fcc_SiFe_dT*(X_Si-X_Fe)**2 )+\
     X_Mg*X_Fe*dL0_fcc_MgFe_dT

   #dGm_fcc_dT    = X_Al*d_0G_fcc_Al_dT+X_Si*d_0G_fcc_Si_dT+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
   #  X_Al*X_Si*dL0_fcc_AlSi_dT

   d2Gm_fcc_dT2    = X_Al*d2_0G_fcc_Al_dT2+X_Si*d2_0G_fcc_Si_dT2+\
     X_Al*X_Si*d2L0_fcc_AlSi_dT2

   d2Gm_fcc_dTSi = (-1)*d_0G_fcc_Al_dT+d_0G_fcc_Si_dT+R*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
     (-2*X_Si+1)*dL0_fcc_AlSi_dT

   
   dGm_fcc_dSi = (-1)*_0G_fcc_Al+_0G_fcc_Si+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+(-2*X_Si+1)*L0_fcc_AlSi
   d2Gm_fcc_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_fcc_AlSi
   dGm_fcc_dAl   = -dGm_fcc_dSi
   d2Gm_fcc_dAl2 = d2Gm_fcc_dSi2
   d2Gm_fcc_dSiAl = -d2Gm_fcc_dSi2

   # molar enthalpy of fcc
   Hm_fcc       = Gm_fcc-T*dGm_fcc_dT
   d_Hm_fcc_dSi = dGm_fcc_dSi-T*d2Gm_fcc_dTSi

   #
   # free energy for diamond phase
   Gm_dia = X_Al*_0G_dia_Al+X_Si*_0G_dia_Si+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*L0_dia_AlSi    # ok
   # derivative with respect to T
   dGm_dia_dT = X_Al*d_0G_dia_Al_dT+X_Si*d_0G_dia_Si_dT+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*dL0_dia_AlSi_dT    # ok
   d2Gm_dia_dT2 = X_Al*d2_0G_dia_Al_dT2+X_Si*d2_0G_dia_Si_dT2+\
     X_Al*X_Si*d2L0_dia_AlSi_dT2    # ok  

   # its 1st and 2nd derivative with respect to Si
   dGm_dia_dSi  = (-1)*_0G_dia_Al+_0G_dia_Si+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
   (-2*X_Si+1)*dL0_dia_AlSi_dT    # ok
   
   cp_dia = -T*d2Gm_dia_dT2
   cp_fcc =-T*d2Gm_fcc_dT2

   # molar enthalpy of diam

   Hm_dia = Gm_dia-T*dGm_dia_dT


   d2Gm_dia_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_dia_AlSi
   dGm_dia_dAl   = -dGm_dia_dSi
   d2Gm_dia_dAl2 = d2Gm_dia_dSi2
   d2Gm_dia_dSiAl = -d2Gm_dia_dSi2
   #dGm_dia_dAl = _0G_dia_Al+(-1)*_0G_dia_Si+R*T*(1+np.log(X_Al)-1-np.log(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #d2Gm_dia_dAl2 =  R*T*(1/X_Al+1/(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #
   #
   mu_dia_si =  dGm_dia_dSi*(1-X_Si)+Gm_dia                            #  ok
   mu_dia_Al =  dGm_dia_dAl*(1-X_Al)+Gm_dia                            # ok
   # chemical potential  
   mu_fcc_Si = dGm_fcc_dSi*(1-X_Si)+Gm_fcc                             # ok
   dmu_fcc_SidSi =  d2Gm_fcc_dSi2*(1-X_Si) # ok
   dmu_fcc_SidAl =  d2Gm_fcc_dSiAl*(1-X_Si)#+dGm_fcc_dSi+dGm_fcc_dAl
   #dmu_fcc_SidAl =  -dmu_fcc_SidSi 

   mu_fcc_Al = dGm_fcc_dAl*(1-X_Al)+Gm_fcc                              
   #
   #dmu_fcc_AldSi = d2Gm_fcc_dSiAl+dGm_fcc_dAl
   #dmu_fcc_AldAl = d2Gm_fcc_dAl2-(dGm_fcc_dAl+X_Al*d2Gm_fcc_dAl2)+dGm_fcc_dAl
   dmu_fcc_AldAl = d2Gm_fcc_dAl2*(1-X_Al)
   dmu_fcc_AldSi =  -dmu_fcc_AldAl # ok
   
   #
   # derivatives of chemical potential
   #dmu_fcc_SidSi =  d2Gm_fcc_dSi2-(dGm_fcc_dSi+X_Si*d2Gm_fcc_dSi2)+dGm_fcc_dSi # ok
   # derivatives of chemical potential
   dmu_dia_Si_dSi =  d2Gm_dia_dSi2*(1-X_Si)
   #dmu_dia_SidSi =  dGm_dia_dSi-(dGm_dia_dSi+X_Si*d2Gm_dia_dSi2)+dGm_dia_dSi 
   dmu_fdia_SidAl =  -dmu_dia_Si_dSi # ok

   if output=="Gm_fcc":
      out = Gm_fcc
   elif output=="Hm_fcc":
      out = Hm_fcc
   elif output=="Hm_Mg2Si":
      out = Hm_Mg2Si
   elif output=="Hm_Mg1_8Si":
      out = Hm_Mg1_8Si
   elif output=="Hm_Al14Si3Fe3":
      out = Hm_Al14Si3Fe3
   elif output=="Hm_Mg5Si6":
      out = Hm_Mg5Si6
   elif output=="d_Hm_fcc_dSi":
      out = d_Hm_fcc_dSi
   elif output=="dGm_fcc_dSi":
      out = dGm_fcc_dSi
   elif output=="d2Gm_fcc_dSi2":
      out = d2Gm_fcc_dSi2
   elif output=="dGm_fcc_dAl":
      out = dGm_fcc_dAl
   elif output=="d2Gm_fcc_dAl2":
      out = d2Gm_fcc_dAl2
   elif output=="cp_fcc":
      out = cp_fcc
   elif output=="cp_dia":
      out = cp_dia
   elif output=="cp_liq":
      out = cp_liq
   elif output=="Gm_dia":
      out = Gm_dia
   elif output=="Hm_dia":
      out = Hm_dia
   elif output=="dGm_dia_dSi":
      out = dGm_dia_dSi
   elif output=="d2Gm_fcc_dSiAl":
      out = d2Gm_fcc_dSiAl
   elif output=="mu_fcc_Si":
      out = mu_fcc_Si
   elif output=="dmu_fcc_SidSi":
      out = dmu_fcc_SidSi
   elif output=="mu_fcc_Al":
      out = mu_fcc_Al
   elif output=="dmu_fcc_AldSi":
      out = dmu_fcc_AldSi
   elif output=="mu_dia_si":
      out = mu_dia_si
   elif output=="mu_dia_Al":
      out = mu_dia_Al
   elif output=="dmu_fcc_SidAl":
      out = dmu_fcc_SidAl
   elif output=="d2_0G_liq_Al_dT2":
      out = d2_0G_liq_Al_dT2
   elif output=="dmu_fcc_AldAl":
      out = dmu_fcc_AldAl
   return out




def thermal_conduc(T,X_Si,Np_dia,output):
   Np_fcc = 1-Np_dia
   X_Al=1-X_Si  
   a_fccAl = 311.72511   
   b_fccAl = -0.09661
   c_fccAl = -13510.26
   lamb_fccAl =  a_fccAl+b_fccAl*T+c_fccAl*T**(-1)

   a_diaSi = -43.46041   
   b_diaSi = 0.01953
   c_diaSi = 55279.27
   lamb_diaSi =  a_diaSi+b_diaSi*T+c_diaSi*T**(-1)

   L0_fcc_AlSi = 34954.92-24.601*T
   L1_fcc_AlSi = -39974.45+28.122*T
   
   if output=="lamb_fccdia_lin":
      if T ==400:
         lamb_fccAlSi = -3746.66431597*X_Si+239.30545389239452
      elif T==600:
         lamb_fccAlSi = -2979.61668715*X_Si+231.19290757411102
      elif T ==800:
         lamb_fccAlSi = -1589.01296397*X_Si+213.30713922739255
   else :   
      lamb_fccAlSi =X_Al*lamb_fccAl+X_Si*lamb_diaSi+X_Al*X_Si*\
      (L0_fcc_AlSi+L1_fcc_AlSi*(X_Al-X_Si))
   
   M0fccdia = -378.95+0.072*T
   M1fccdia = 1442.77-0.635*T
   


   lamb_fccdia = Np_fcc*lamb_fccAlSi+Np_dia*lamb_diaSi-Np_fcc*Np_dia*\
   (M0fccdia+M1fccdia*(Np_fcc-Np_dia)) 

   lamb_fccdia_wo_hc = Np_fcc*lamb_fccAlSi+Np_dia*lamb_diaSi     

   if output=="lamb_fccAlSi":
      out = lamb_fccAlSi
   elif output=="lamb_fccdia":
      out = lamb_fccdia
   elif output=="lamb_fccdia_wo_hc":
      out = lamb_fccdia_wo_hc
   elif output=="lamb_fccdia_lin":
      out = lamb_fccdia   
   return out  


def molar_volume(T0,T,X_Si,output):
   X_Al=1-X_Si

   V0_fccAl = 9.99987596e-6#9.77430e-6
   a_fccAl  = 2.17303609e-05 # 6.91213e-5
   b_fccAl  = 2.63550737e-09 #0
   c_fccAl  = 1.47000588e-11#4.86802e-11
   d_fccAl  = -9.03030712e-02#0.413484
   #0V_fccAl =                      # interaction parameters 
   #
   V0_fccSi = 9.20000852e-6
   a_fccSi  = 4.50736462e-06#1.12466e-5
   b_fccSi  = 2.00632966e-09#8.92959e-9
   c_fccSi  = -8.07642813e-13#-3.09709e-12
   d_fccSi  = -1.50848370e-01#-0.251151
   #
   V0_diaSi =1.20588085e-05#12.0588e-6
   a_diaSi  = 3.43321444e-06#8.56511e-6
   b_diaSi  = 1.54351184e-09#6.85005e-9
   c_diaSi  = -6.13087337e-13#-2.35401e-12
   d_diaSi  = -1.14950173e-01#-0.191371
   # al liquid
   V0_liqAl =1.03371674e-05#12.0588e-6
   a_liqAl  = 4.66770167e-05#8.56511e-6
   b_liqAl  =-6.07205185e-09 #6.85005e-9
   c_liqAl  =5.29104281e-13#-2.35401e-12
   d_liqAl  = 3.51948259e-02 #-0.191371
   # si liquid
   V0_liqSi =9.05196044e-06#12.0588e-6
   a_liqSi  = 5.77061979e-05#8.56511e-6
   b_liqSi  = -9.03455430e-09#6.85005e-9
   c_liqSi  = 8.94822667e-13#-2.35401e-12
   d_liqSi  =7.31878142e-02#-0.191371
   
   # fcc al
   alpha_fccAl    = (1/3.)*(a_fccAl+b_fccAl*T+c_fccAl*T**2+d_fccAl*T**(-2))
   CLE_fccAl      = alpha_fccAl*(T-T0)
   s_alpha_dT_fccAl = (a_fccAl*T+(b_fccAl/2)*T**2+(c_fccAl/3)*T**3\
              +d_fccAl*(-T**(-1)))-\
              (a_fccAl*T0+(b_fccAl/2)*T0**2+(c_fccAl/3)*T0**3\
              +d_fccAl*(-T0**(-1)))
   Vm_fccAl       = V0_fccAl*np.exp(3*s_alpha_dT_fccAl)

   # liq al

   alpha_liqAl    = (1/3.)*(a_liqAl+b_liqAl*T+c_liqAl*T**2+d_liqAl*T**(-2))
   CLE_liqAl      = alpha_liqAl*(T-T0)
   s_alpha_dT_liqAl = (a_liqAl*T+(b_liqAl/2)*T**2+(c_liqAl/3)*T**3\
              +d_liqAl*(-T**(-1)))-\
              (a_liqAl*T0+(b_liqAl/2)*T0**2+(c_liqAl/3)*T0**3\
              +d_liqAl*(-T0**(-1)))
   Vm_liqAl       = V0_liqAl*np.exp(3*s_alpha_dT_liqAl)

   # fcc si
   alpha_fccSi    = (1/3.)*(a_fccSi+b_fccSi*T+c_fccSi*T**2+d_fccSi*T**(-2))
   CLE_fccSi      = alpha_fccSi*(T-T0)
   s_alpha_dT_fccSi = (a_fccSi*T+(b_fccSi/2)*T**2+(c_fccSi/3)*T**3\
              +d_fccSi*(-T**(-1)))-\
              (a_fccSi*T0+(b_fccSi/2)*T0**2+(c_fccSi/3)*T0**3\
              +d_fccSi*(-T0**(-1)))
   Vm_fccSi       = V0_fccSi *np.exp(3*s_alpha_dT_fccSi)


   # liq si

   alpha_liqSi    = (1/3.)*(a_liqSi+b_liqSi*T+c_liqSi*T**2+d_liqSi*T**(-2))
   CLE_liqSi      = alpha_liqSi*(T-T0)
   s_alpha_dT_liqSi = (a_liqSi*T+(b_liqSi/2)*T**2+(c_liqSi/3)*T**3\
              +d_liqSi*(-T**(-1)))-\
              (a_liqSi*T0+(b_liqSi/2)*T0**2+(c_liqSi/3)*T0**3\
              +d_liqSi*(-T0**(-1)))
   Vm_liqSi       = V0_liqSi*np.exp(3*s_alpha_dT_liqSi)

   #dia si
   alpha_diaSi    = (1/3.)*(a_diaSi+b_diaSi*T+c_diaSi*T**2+d_diaSi*T**(-2))
   CLE_diaSi      = alpha_diaSi*(T-T0)
   s_alpha_dT_diaSi = (a_diaSi*T+(b_diaSi/2)*T**2+(c_diaSi/3)*T**3\
              +d_diaSi*(-T**(-1)))-\
              (a_diaSi*T0+(b_diaSi/2)*T0**2+(c_diaSi/3)*T0**3\
              +d_diaSi*(-T0**(-1)))
   Vm_diaSi       = V0_diaSi*np.exp(3*s_alpha_dT_diaSi)

   Vmfcc = X_Al*Vm_fccAl+X_Si*Vm_fccSi
   Vmdia = Vm_diaSi

   Vmliq = X_Al*Vm_liqAl+X_Si*Vm_liqSi 

   #alpha    = (1/3.)*(a_fccAl+b_fccAl*T+c_fccAl*T**2+d_fccAl*T**(-2))
   #CLE      = alpha*(T-T0)
   #s_alpha_dT = (a_fccAl*T+(b_fccAl/2)*T**2+(c_fccAl/3)*T**3\
   #           +d_fccAl*(-T**(-1)))-\
   #           (a_fccAl*T0+(b_fccAl/2)*T0**2+(c_fccAl/3)*T0**3\
   #           +d_fccAl*(-T0**(-1)))
   #Vm_fccAl = V0_fccAl*np.exp(3*s_alpha_dT)
   if output=="Vmfcc":
      out = Vmfcc
   elif output=="Vmdia":
      out = Vmdia
   elif output=="Vmliq":
      out = Vmliq
   elif output=="CLE_fccAl":
      out = CLE_fccAl
   return out

def vm_dia_hallstedt(T,output):
   vm = (11.99189 + 1.485e-4*T + 7.0e-9*T**2 + 7.84*T**(-1) - 380*T**(-2))*1e-6
   lnvm = np.log((11.99189 + 1.485e-4*T + 7.0e-9*T**2 + 7.84*T**(-1) - 380*T**(-2))*1e-6)
   if output=="vm":
      out = vm
   elif output=="lnvm":
      out = lnvm
   return out  

def fit_vm(T,T0,V0,a,b,c,d):
   return (a*T+(b/2)*T**2+(c/3)*T**3\
              +d*(-T**(-1)))-\
              (a*T0+(b/2)*T0**2+(c/3)*T0**3\
              +d*(-T0**(-1)))

def vm_fcc_hallstedt(T,output):
   vm = (9.13309 + 1.485e-4*T + 7.0e-9*T**2 + 7.84*T**(-1) - 380*T**(-2))*1e-6
   lnvm = np.log((9.13309 + 1.485e-4*T + 7.0e-9*T**2 + 7.84*T**(-1) - 380*T**(-2))*1e-6)
   if output=="vm":
      out = vm
   elif output=="lnvm":
      out = lnvm
   return out  

def elas_const(T,output):
   # B = bulk modulus
   # E = young modulus
   a_B_al = 8.030e1*1e9
   b_B_al=-1.319e-2*1e9
   c_B_al=-6.815e-6*1e9

   a_E_al = 7.028e1*1e9
   b_E_al=2.114e-3*1e9
   c_E_al=-4.903e-5*1e9
   
   # data from http://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html
   a_C11_si = 16.38*1e10
   b_C11_si = -1.28e-3*1e10
   c_C11_si = 0.

   a_C44_si = 8.17*1e10
   b_C44_si = -0.59e-3*1e10
   c_C44_si = 0.

   a_C12_si = 5.92*1e10
   b_C12_si = -0.48e-3*1e10
   c_C12_si = 0.


   B_al =a_B_al+b_B_al*T+c_B_al*T**2
   E_al =a_E_al+b_E_al*T+c_E_al*T**2

   C11_Si = a_C11_si+b_C11_si*T+c_C11_si*T**2
   C44_Si = a_C44_si+b_C44_si*T+c_C44_si*T**2
   C12_Si = a_C12_si+b_C12_si*T+c_C12_si*T**2
   #G = -3*E_al*B_al/(E_al-9*B_al) #shear modulus
   #C44 = G
   #C11 = (3*B_al+4*C44)/3
   #C12 = C11-2*C44
   C44 = -3*E_al*B_al/(E_al-9*B_al) #shear modulus
   C11 = B_al-4*E_al*B_al/(E_al-9*B_al) #shear modulus
   C12 = B_al+2*E_al*B_al/(E_al-9*B_al) #shear modulus

   if output=="C11":
      out = C11
   elif output=="C12":
      out = C12
   elif output=="C44":
      out = C44
   elif output=="C11_Si":
      out = C11_Si
   elif output=="C44_Si":
      out = C44_Si
   elif output=="C12_Si":
      out = C12_Si
   elif output=="B":
      out = B_al
   elif output=="E":
      out = E_al
   return out  


def gibbs_energy_ref(T,X_Si,output):
   out=0
   # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.   X_Al = 1-X_Si
   R     = 8.314472                  # = [J.K-1.mol-1]
   X_Al = 1-X_Si
   #
   # free energy for G0_FCC-A1
   # for AL 
   if T>298.15 and T<=700:
      _0G_fcc_Al=-7976.15+137.093038*T-24.3671976*T*np.log(T)-1.884662e-3*T**2-0.877664e-6*T**3\
        +74092*T**(-1)
      d_0G_fcc_Al_dT= 137.093038-24.3671976*(1+np.log(T))-(2)*1.884662e-3*T-(3)*0.877664e-6*T**2\
        +(-1)*74092*T**(-2)
      d2_0G_fcc_Al_dT2= -24.3671976*(1/T)-(2)*1.884662e-3-(3*2)*0.877664e-6*T\
        +(-1)*(-2)*74092*T**(-3)
   elif T>700 and T<=933.47:
      _0G_fcc_Al=-11276.24+223.048446*T-38.5844296*T*np.log(T)+18.531982e-3*T**2-5.764227e-6*T**3\
        +74092*T**(-1)
      d_0G_fcc_Al_dT=  223.048446-38.5844296*(1+np.log(T))+(2)*18.531982e-3*T-(3)*5.764227e-6*T**2\
        +(-1)*74092*T**(-2)
      d2_0G_fcc_Al_dT2= -38.5844296*(1/T)+(2)*18.531982e-3-(3)*(2)*5.764227e-6**T\
      +(-1)*(-2)*74092*T**(-3)
   elif T>933.47 and T<2900.0:
      _0G_fcc_Al=-11278.378+188.6841536*T-31.748192*T*np.log(T)-1230.524e25*T**(-9)
      d_0G_fcc_Al_dT=188.6841536-31.748192*(1+np.log(T))-(-9)*1230.524e25*T**(-10)
      d2_0G_fcc_Al_dT2= -31.748192*(1/T)-(-9)*(-10)*1230.524e25*T**(-11)
   #
   # free energy for G0_FCC-A1
   # for Si 
   if T>298.15 and T<=1687:
      _0G_dia_Si=-8162.609+137.236859*T-22.8317533*T*np.log(T)-1.912904e-3*T**2-0.003552e-6*T**3\
        +176667*T**(-1)
      d_0G_dia_Si_dT= 137.236859-22.8317533*(1+np.log(T))-(2)*1.912904e-3*T-(3)*0.003552e-6*T**2\
        +(-1)*176667*T**(-2)
   elif T>1687 and T<3600:
      _0G_dia_Si=-9457.642+167.281367*T-27.196*T*np.log(T)-420.369e28*T**(-9)
      d_0G_dia_Si_dT= 167.281367-27.196*(1+np.log(T))-(-9)*420.369e28*T**(-10)
   

   _0G_fcc_Si     = 51000.00-21.8*T+_0G_dia_Si
   d_0G_fcc_Si_dT = -21.8+d_0G_dia_Si_dT
   #d2_0G_fcc_Si_dT2 = 
   _0G_dia_Al     = 30*T+_0G_fcc_Al
   d_0G_dia_Al_dT = 30 + d_0G_fcc_Al_dT
   d2_0G_dia_Al_dT2 = d2_0G_fcc_Al_dT2
   #
   L0_fcc_AlSi     = -3143.78+0.3929*T
   dL0_fcc_AlSi_dT = 0.3929
   L0_dia_AlSi     = 113246.16-47.5551*T 
   dL0_dia_AlSi_dT = -47.5551
   #
   # free energy for fcc phase
   #
   Gm_fcc = X_Al*(_0G_fcc_Al-_0G_fcc_Al)+X_Si*(_0G_fcc_Si-_0G_dia_Si)+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*L0_fcc_AlSi
   # derivative with respect to T
   dGm_fcc_dT    = X_Al*(d_0G_fcc_Al_dT-d_0G_fcc_Al_dT)+X_Si*(d_0G_fcc_Si_dT-d_0G_dia_Si_dT)+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*dL0_fcc_AlSi_dT

   d2Gm_fcc_dTSi = (-1)*(d_0G_fcc_Al_dT-d_0G_fcc_Al_dT)+(d_0G_fcc_Si_dT-d_0G_dia_Si_dT)+R*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
     (-2*X_Si+1)*dL0_fcc_AlSi_dT

   
   dGm_fcc_dSi = (-1)*(_0G_fcc_Al-_0G_fcc_Al)+(_0G_fcc_Si-_0G_dia_Si)+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+(-2*X_Si+1)*L0_fcc_AlSi
   d2Gm_fcc_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_fcc_AlSi
   dGm_fcc_dAl   = -dGm_fcc_dSi
   d2Gm_fcc_dAl2 = d2Gm_fcc_dSi2
   d2Gm_fcc_dSiAl = -d2Gm_fcc_dSi2

   # molar enthalpy of fcc
   Hm_fcc       = Gm_fcc-T*dGm_fcc_dT
   d_Hm_fcc_dSi = dGm_fcc_dSi-T*d2Gm_fcc_dTSi

   #
   # free energy for diamond phase
   Gm_dia = X_Al*(_0G_dia_Al-_0G_fcc_Al)+X_Si*(_0G_dia_Si-_0G_dia_Si)+R*T*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*L0_dia_AlSi    # ok
   # derivative with respect to T
   dGm_dia_dT = X_Al*(d_0G_dia_Al_dT-d_0G_fcc_Al_dT)+X_Si*(d_0G_dia_Si_dT-d_0G_dia_Si_dT)+R*(X_Al*np.log(X_Al)+X_Si*np.log(X_Si))+\
     X_Al*X_Si*dL0_dia_AlSi_dT    # ok

   # its 1st and 2nd derivative with respect to Si
   dGm_dia_dSi  = (-1)*(_0G_dia_Al-_0G_fcc_Al)+(_0G_dia_Si-_0G_dia_Si)+R*T*(-1-np.log(1-X_Si)+1+np.log(X_Si))+\
   (-2*X_Si+1)*dL0_dia_AlSi_dT    # ok
   
   # molar enthalpy of diam

   Hm_dia = Gm_dia-T*dGm_dia_dT


   d2Gm_dia_dSi2 =  R*T*(-1+1/(1-X_Si)+1+1/(X_Si))+(-2)*L0_dia_AlSi
   dGm_dia_dAl   = -dGm_dia_dSi
   d2Gm_dia_dAl2 = d2Gm_dia_dSi2
   d2Gm_dia_dSiAl = -d2Gm_dia_dSi2
   #dGm_dia_dAl = _0G_dia_Al+(-1)*_0G_dia_Si+R*T*(1+np.log(X_Al)-1-np.log(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #d2Gm_dia_dAl2 =  R*T*(1/X_Al+1/(1-X_Al))+\
   #  +(-2*X_Al+1)*L0_dia_AlSi
   #
   #
   mu_dia_si =  dGm_dia_dSi*(1-X_Si)+Gm_dia                            #  ok
   mu_dia_Al =  dGm_dia_dAl*(1-X_Al)+Gm_dia                            # ok
   # chemical potential  
   mu_fcc_Si = dGm_fcc_dSi*(1-X_Si)+Gm_fcc                             # ok
   dmu_fcc_SidSi =  d2Gm_fcc_dSi2*(1-X_Si) # ok
   dmu_fcc_SidAl =  d2Gm_fcc_dSiAl*(1-X_Si)#+dGm_fcc_dSi+dGm_fcc_dAl
   #dmu_fcc_SidAl =  -dmu_fcc_SidSi 

   mu_fcc_Al = dGm_fcc_dAl*(1-X_Al)+Gm_fcc                              
   #
   #dmu_fcc_AldSi = d2Gm_fcc_dSiAl+dGm_fcc_dAl
   #dmu_fcc_AldAl = d2Gm_fcc_dAl2-(dGm_fcc_dAl+X_Al*d2Gm_fcc_dAl2)+dGm_fcc_dAl
   dmu_fcc_AldAl = d2Gm_fcc_dAl2*(1-X_Al)
   dmu_fcc_AldSi =  -dmu_fcc_AldAl # ok
   
   #
   # derivatives of chemical potential
   #dmu_fcc_SidSi =  d2Gm_fcc_dSi2-(dGm_fcc_dSi+X_Si*d2Gm_fcc_dSi2)+dGm_fcc_dSi # ok
   # derivatives of chemical potential
   dmu_dia_Si_dSi =  d2Gm_dia_dSi2*(1-X_Si)
   #dmu_dia_SidSi =  dGm_dia_dSi-(dGm_dia_dSi+X_Si*d2Gm_dia_dSi2)+dGm_dia_dSi 
   dmu_fdia_Si_dAl =  -dmu_dia_Si_dSi # ok

   if output=="Gm_fcc":
      out = Gm_fcc
   elif output=="Hm_fcc":
      out = Hm_fcc
   elif output=="d_Hm_fcc_dSi":
      out = d_Hm_fcc_dSi
   elif output=="dGm_fcc_dSi":
      out = dGm_fcc_dSi
   elif output=="d2Gm_fcc_dSi2":
      out = d2Gm_fcc_dSi2
   elif output=="dGm_fcc_dAl":
      out = dGm_fcc_dAl
   elif output=="d2Gm_fcc_dAl2":
      out = d2Gm_fcc_dAl2
   elif output=="Gm_dia":
      out = Gm_dia
   elif output=="Hm_dia":
      out = Hm_dia
   elif output=="dGm_dia_dSi":
      out = dGm_dia_dSi
   elif output=="d2Gm_fcc_dSiAl":
      out = d2Gm_fcc_dSiAl
   elif output=="mu_fcc_Si":
      out = mu_fcc_Si
   elif output=="dmu_fcc_SidSi":
      out = dmu_fcc_SidSi
   elif output=="mu_fcc_Al":
      out = mu_fcc_Al
   elif output=="dmu_fcc_AldSi":
      out = dmu_fcc_AldSi
   elif output=="mu_dia_si":
      out = mu_dia_si
   elif output=="mu_dia_Al":
      out = mu_dia_Al
   elif output=="dmu_fcc_SidAl":
      out = dmu_fcc_SidAl
   elif output=="dmu_fcc_AldAl":
      out = dmu_fcc_AldAl
   return out

def gibbs_energy_ref_mur(T,X_Si,output):
   R     = 8.314472 
   _0G_fcc_al_mur = -10792+11.56*T
   _0G_fcc_si_mur = 12.12*T
   A_fcc = -200-7.594*T
   
   _0G_dia_si_mur = -50600+30.*T
   _0G_dia_al_mur = 30.*T
   A_dia = 89138-31.455*T

   G_fcc_mur = (_0G_fcc_al_mur-_0G_fcc_al_mur)*(1-X_Si)+(_0G_fcc_si_mur-_0G_dia_si_mur)*X_Si\
             +R*T*(X_Si*np.log(X_Si)+(1-X_Si)*np.log(1-X_Si))\
             +X_Si*(1-X_Si)*A_fcc
   G_dia_mur = (_0G_dia_al_mur-_0G_dia_si_mur)*(1-X_Si)+(_0G_dia_si_mur-_0G_dia_si_mur)*X_Si\
             +R*T*(X_Si*np.log(X_Si)+(1-X_Si)*np.log(1-X_Si))\
             +X_Si*(1-X_Si)*A_dia

   if output=="Gm_fcc_mur":
      out = G_fcc_mur
   elif output=="Gm_dia_mur":
      out = G_dia_mur
   return out
