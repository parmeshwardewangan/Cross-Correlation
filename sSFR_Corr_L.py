#!/usr/bin/env python
# coding: utf-8

# In[21]:


# import frogress
import pandas as pd
import numpy as np
import sys
from subprocess import call
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import glob
from astropy.io import fits
import healpy as hp
import numpy as np;
import astropy.io.fits as pyfits;
from pylab import *
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
def fArea(N_rand, Z):
    r = cosmo.comoving_distance(Z).value
    Area = (N_rand*(r**2))/1181810286.0042278

    return Area

# ra,dec 2 xyz
def get_xyz(ra,dec):
    theta = (90-dec)*np.pi/180
    phi = ra*np.pi/180
    z = np.cos(theta)
    y = np.sin(phi)*np.sin(theta)
    x = np.cos(phi)*np.sin(theta)
    return x,y,z


# getting mirror points
@np.vectorize
def get_mirr(ra0,dec0,gra,gdec):
    x0,y0,z0  = get_xyz(ra0,dec0)
    xg,yg,zg  = get_xyz(gra,gdec)
    cos_theta = x0*xg + y0*yg + z0*zg 
    alpha   = y0*zg - yg*z0  
    beta    = x0*zg - xg*z0  
    gamma   = yg*x0 - xg*y0  
    
    num_zm = (2*cos_theta**2 - 1)*(x0*beta + y0*alpha) - cos_theta*(xg*beta + yg*alpha)
    den_zm = yg*(gamma*x0 - alpha*z0) - xg*(beta*z0 + gamma*y0) + zg*(beta*x0 + alpha*y0)
    
    zm = num_zm *1.0/den_zm
    ym = (alpha*cos_theta - zm*(z0*alpha - x0*gamma))*1.0/(x0*beta + y0*alpha)
    xm = (beta*cos_theta - zm*(y0*gamma + z0*beta))*1.0/(x0*beta + y0*alpha)
    r = (xm**2 + ym**2 + zm**2)**0.5
        
    theta = np.arccos(np.clip(zm*1.0/r,-1,1))
    phi = np.arctan(ym*1.0/xm)
    if xm>0 and ym>0:
        phi = phi
    if xm<0 and ym>0:
        phi = np.pi - abs(phi)
    if xm<0 and ym<0:
        phi = np.pi + abs(phi)
    if xm>0 and ym<0:
        phi = 2*np.pi - abs(phi)
        
    ra = phi*180/np.pi
    dec = (np.pi/2 - theta)*180.0/np.pi
    
    return ra,dec

def sep2(mra, mdec, pra, pdec, pzred):
    c1 = SkyCoord(ra=mra*u.degree, dec=mdec*u.degree)
    c2 = SkyCoord(ra=pra*u.degree, dec=pdec*u.degree)
    sep = c1.separation(c2)
    sep.arcsecond

    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

    d_A = cosmo.comoving_distance(pzred)
    # print(d_A) 

    theta = sep.arcsecond*u.arcsec
    distance_Mpc = (theta * d_A).to(u.Mpc, u.dimensionless_angles())/(1+pzred) # unit is Mpc only now
    
    return distance_Mpc.value


# getting jackknife region
def getregions(ra,dec):
    regionID=np.zeros(len(ra)).astype(int)
    RAMIN,RAMAX,DECMIN,DECMAX=np.loadtxt("/mnt/home/project/cparmeshwar/Minor_project/Annular_radius_0.3_0.6/jackregion_SM.txt", delimiter=" ",unpack=True)
    for ni in range(len(ra)):
        idx= ((RAMIN<ra[ni]) & (RAMAX>ra[ni]) & (DECMIN<dec[ni]) & (DECMAX>dec[ni]))
        regionID[ni]=int(np.where(idx)[0])
    return regionID

def lens_select(Rmin, Rmax, mirror=False, HighSM = False):
    
    cdat = fits.open('/mnt/home/project/cparmeshwar/Minor_project/GAMA/grpGal_smass_sSFR.fits')[1].data
    
    idx = (cdat['Nfof'] >= 15) 
    sSFR = np.log10(cdat['sSFR_0_1Gyr_best_fit'][idx])
    sm = cdat['logmstar'][idx]
    pid = cdat['GroupID_1'][idx]
    pra = cdat['IterCenRA'][idx]
    pdec = cdat['IterCenDec'][idx]
    pzred = cdat['Zfof'][idx]
    
    mra   = cdat['RA'][idx]
    mdec  = cdat['Dec'][idx]
    mrsep = sep2(mra, mdec, pra, pdec, pzred)
    mpid  = cdat['CATAID_1_1'][idx]
    mzred = pzred
    
#     plt.hist(mrsep)
    idx =  (mrsep>Rmin) & (mrsep<Rmax) # &(sm>10.2) & (sm<11.5) & (mzred>0.04) & (mzred<0.34)
    sSFR = sSFR[idx]
    sm = sm[idx]
    mra   = mra[idx]
    mdec  = mdec[idx]
    mzred = mzred[idx]
    mpra  = pra[idx]
    mpdec = pdec[idx]
    #     mprid = mprid[idx]

    mirrra, mirrdec = get_mirr(mpra, mpdec, mra, mdec)

    msk = hp.read_map("/mnt/home/student/cdivya/github/weaklens_pipeline_s16a/DataStore/data/S16A_mask_w_holes.fits")
    galpix = hp.ang2pix(int(np.sqrt(msk.size/12)), mra, mdec, lonlat=1)
    mirrpix = hp.ang2pix(int(np.sqrt(msk.size/12)), mirrra, mirrdec, lonlat=1)

    sel = (msk[galpix]==1.0) & (msk[mirrpix]==1.0)  # Masking is on both sat and mirr point 
    print(len(sel),'len_sel')
    if mirror==False:
        ra1    = mra[sel].astype('float')
        dec1   = mdec[sel].astype('float')
        cra    = mpra[sel].astype('float')
        cdec   = mpdec[sel].astype('float')
        sm1    = sm[sel].astype('float')
        sSFR1 = sSFR[sel].astype('float')

    else:
        ra1    = mirrra[sel].astype('float')
        dec1   = mirrdec[sel].astype('float')
        cra    = mpra[sel].astype('float')
        cdec   = mpdec[sel].astype('float')
        sm1    = sm[sel].astype('float')
        sSFR1 = sSFR[sel].astype('float')

        print("using mirror points")

    print('len(ra) after masking',len(ra1))
    print('lencra',len(cra))
    print(len(sm1))
    print('median of sm', np.median(sm1))
    
#     plt.scatter(cra, cdec)


    lid1 = getregions(cra,cdec)



    zred1  = mzred[sel].astype('float')
    
    print('min(zred)',np.min(zred1))
    print('midian(zred)', np.median(zred1))
    wt1    = ra1*1.0/ra1

    sys.stdout.write("Selecting %d samples \n" % (ra1.size))
    
    
    values = sSFR1
    sfrmid, binedges, binnumber = stats.binned_statistic(sm1, values, 'median', bins=75)
    
    SFR = np.array([])
    sm = np.array([])
    ra = np.array([])
    dec = np.array([])
    zred = np.array([])
    wt = np.array([])
    lid = np.array([])
    for i in range(len(binedges)-1):
        if HighSM == True:
            ind = (sm1>binedges[i]) & (sm1<binedges[i+1]) & (sSFR1 > sfrmid[i])
            SFR = np.concatenate((SFR, sSFR1[ind]))
            sm = np.concatenate((sm, sm1[ind]))
            ra = np.concatenate((ra, ra1[ind]))
            dec = np.concatenate((dec, dec1[ind]))
            zred = np.concatenate((zred, zred1[ind]))
            wt = np.concatenate((wt, wt1[ind]))
            lid = np.concatenate((lid, lid1[ind]))
            
        else: 
            ind = (sm1>binedges[i]) & (sm1<binedges[i+1]) & (sSFR1 < sfrmid[i])
            SFR = np.concatenate((SFR, sSFR1[ind]))
            sm = np.concatenate((sm, sm1[ind]))
            ra = np.concatenate((ra, ra1[ind]))
            dec = np.concatenate((dec, dec1[ind]))
            zred = np.concatenate((zred, zred1[ind]))
            wt = np.concatenate((wt, wt1[ind]))
            lid = np.concatenate((lid, lid1[ind]))
        
    if HighSM == False:
        sfr = SFR
        plt.scatter(binedges[1:101], sfrmid)
        plt.scatter(sm, sfr, c='r')
        plt.ylabel('log10(sSFR)')
        plt.xlabel('log10(StellarMass)')
    else:
        sfr = SFR
        plt.scatter(binedges[1:101], sfrmid)
        plt.scatter(sm, SFR, c = 'b')
        plt.ylabel('log10(sSFR)')
        plt.xlabel('log10(StellarMass)')
        
    print('umber of final sample', len(sfr))
    
    return ra, dec, zred, wt, lid


# In[22]:




# plt.figure(figsize=(8, 6), dpi=300)
# # ys = np.linspace(10.2, 11.5, 1000)
# # xs = np.linspace(0.04, 0.34, 1000)
# # plt.plot(xs, [10.2]*1000, c = 'k')
# # plt.plot([0.04]*1000, ys, c = 'k')
# # plt.plot(xs, [11.5]*1000, c = 'k')
# # plt.plot([0.34]*1000, ys, c = 'k')
# plt.ylim(min(sm1), max(sm1))
# plt.scatter(lred, sm1, s = 3)
# plt.ylim(0,1e-10)
# plt.xlabel('Redshift')
# plt.ylabel('log(stellar mass)')
# plt.show()


# In[88]:



# read sources
def read_sources(ifil):
    # various columns in sources 
    # ragal, decgal
    data = fits.open(ifil)[1].data
    sra = data['ira']
    sdec = data['idec']
    smag = data['imag_kron']
    mat = np.transpose([sra,sdec,smag])
    return mat
def read_sources1(ifil):
    # various columns in sources 
    # ragal, decgal
    data = fits.open(ifil)[1].data
    sra = data['ra']
    sdec = data['dec']
    mat = np.transpose([sra,sdec])
    return mat

def MAG(mag, red, maxred):
    Omegam=0.315
    cc = FlatLambdaCDM(H0=100, Om0=Omegam)
    D = cc.comoving_distance(maxred).value
    d = cc.comoving_distance(red).value
    Mabs = 24.5 - 5*np.log10(D) - 5
    
    mabs = mag - 5*np.log10(d) - 5
    
    if mabs < Mabs:
        return 1
    else:
        return 0
    
    
# counts given radial bins and lens positions
def run_pipe(Omegam=0.315, rmin=0.03, rmax=1.0, nbins=10, randomfile = 'RandomSFRLM', outputfile = 'GalaxySFRLM', Rmin=0.1, Rmax=1.0, mirror=False, HighSM=False):
    #set the cosmology with omegaM parameter
    cc = FlatLambdaCDM(H0=100, Om0=Omegam) # fixing H0=100 to set units in Mpc h-1
    # set the projected radial binning
    rmin  =  rmin
    rmax  =  rmax
    nbins = nbins #10 radial bins for our case
    rbins  = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    rdiff  = np.log10(rbins[1]*1.0/rbins[0])
    # initializing arrays for signal compuations
    sumwls = np.zeros(len(rbins[:-1]))
    sumwlsr = np.zeros(len(rbins[:-1]))
    

    # getting the lenses data
    lra, ldec, lred, lwgt, lid = lens_select(Rmin=Rmin, Rmax=Rmax, mirror=mirror, HighSM = HighSM)
    
    maxred = np.max(lred)
    

    # convert lense ra and dec into x,y,z cartesian coordinates
    lx, ly, lz = get_xyz(lra, ldec)
#     lx1, ly1, lz1 = get_xyz(rnra, rndec)
    
    
    # putting kd tree around the lenses
    lens_tree = cKDTree(np.array([lx, ly, lz]).T)
#     lens_tree1 = cKDTree(np.array([lx1,ly1,lz1]).T)

    print('lenses tree is ready\n')

    # setting maximum search radius
    dcommin = cc.comoving_distance(0.05).value   # Lowest redshift was 0.006 so the whole code was taking time
    dismax  = (rmax*1.0/(dcommin))

    # lets first catch the file list for source
    sflist = ['WIDE12H', 'GAMA15H', 'GAMA09H']
    
    if mirror:
        outputfile = outputfile + '_mirror'
    fout = open(outputfile, "w")
    fout.write("# 0:rbin 1:regionid 2:redshift 3:RA\n")
    
    if mirror:
        randomfile = randomfile + '_mirror'
    rout = open(randomfile, "w")
    rout.write("# 0:rbin 1:regionid 2:redshift 3:RA\n")

    # Ready to pounce on the source data
    for ifil in sflist:
        fpin ='/mnt/home/faculty/csurhud/github/weaklens_pipeline/DataStore/S16A_v2.0/%s.fits'%ifil
        rpin ='/mnt/home/faculty/csurhud/github/weaklens_pipeline/DataStore/S16A_v2.0/%s_rand.fits'%ifil
        
        
        # catching the source data matrix
        # please have a check for the columns names
        datagal = read_sources(fpin)
        datarnd = read_sources1(rpin)

        
        Ngal = len(datagal[:,0])  # total number of galaxies in the source file
        Nrnd = len(datarnd[:,0])
        
        # first two entries are ra and dec for the sources
        allrard = datarnd[:,0]
        alldecrd = datarnd[:,1]
        allmag = datagal[:2]
        
        allragal = datagal[:,0]
        alldecgal = datagal[:,1]
        # ra and dec to x,y,z for sources
        allsx, allsy, allsz = get_xyz(allragal, alldecgal)
        allrx, allry, allrz = get_xyz(allrard, alldecrd)
        
        # query in a ball around individual sources and collect the lenses ids with a maximum radius
        slidx = lens_tree.query_ball_point(np.transpose([allsx, allsy, allsz]), dismax)
        rlidx = lens_tree.query_ball_point(np.transpose([allrx, allry, allrz]), dismax)
        # various columns in sources
        # ragal, decgal
#         looping over all the galaxies
        for igal in range(Ngal):
            ragal    = datagal[igal,0]
            decgal   = datagal[igal,1]
            maggal = datagal[igal,2]
            
            if math.isnan(maggal):
                continue
            wgal     = ragal*1.0/ragal

            # array of lenses indices
            lidx = np.array(slidx[igal])
            # removing sources which doesn't have any lenses around them
            if len(lidx)==0:
                continue
            sra    = ragal
            sdec   = decgal
            smag  = maggal

            l_ra   = lra[lidx]
            l_dec  = ldec[lidx]
            l_zred = lred[lidx]
            l_wgt  = lwgt[lidx]
            l_id  = lid[lidx]
            
            

            sx, sy, sz = get_xyz(sra,sdec) # This is source which have lense in Rmax radii
            lx, ly, lz = get_xyz(l_ra,l_dec) # This are lenses which are within 2 radii of a particular source

            # getting the radial separations for a lense source pair
            sl_sep = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - sz)**2)
            sl_sep = sl_sep * cc.comoving_distance(l_zred).value
            
            
            for ll,sep in enumerate(sl_sep):
                
                if (sep<rmin or sep>rmax or MAG(smag, l_zred[ll], maxred) == 0):
                    continue
                rb = int(np.log10(sep*1.0/rmin)*1/rdiff)

                fout.write("%le\t%le\t%le\t%le\n"%(rbins[rb], l_id[ll], l_zred[ll], l_ra[ll]))
                
                # following equations given in the surhud's lectures
#                 w_ls    = l_wgt[ll] * wgal 
                # separate numerator and denominator computation
#                 sumwls[rb]  += w_ls
                
        print(ifil)
        
        for igal in range(Nrnd):
            rarnd    = datarnd[igal,0]
            decrnd   = datarnd[igal,1]
            wgal     = rarnd*1.0/rarnd

            # array of lenses indices
            lidx = np.array(rlidx[igal])
            # removing sources which doesn't have any lenses around them
            if len(lidx)==0:
                continue
            sra    = rarnd
            sdec   = decrnd

            l_ra   = lra[lidx]
            l_dec  = ldec[lidx]
            l_zred = lred[lidx]
            l_wgt  = lwgt[lidx]
            l_id  = lid[lidx]

            sx, sy, sz = get_xyz(sra,sdec) # This is source which have lense in Rmax radii
            lx, ly, lz = get_xyz(l_ra,l_dec) # This are lenses which are within 2 radii of a particular source

            # getting the radial separations for a lense source pair
            sl_sep = np.sqrt((lx - sx)**2 + (ly - sy)**2 + (lz - sz)**2)
            sl_sep = sl_sep * cc.comoving_distance(l_zred).value
            
            for ll,sep in enumerate(sl_sep):
                
                if (sep<rmin or sep>rmax):
                    continue
                rb = int(np.log10(sep*1.0/rmin)*1/rdiff)

                rout.write("%le\t%le\t%le\t%le\n"%(rbins[rb], l_id[ll], fArea(1, l_zred[ll]), l_ra[ll]))
                
                # following equations given in the surhud's lectures
#                 w_ls    = l_wgt[ll] * wgal 
                # separate numerator and denominator computation
#                 sumwls[rb]  += w_ls
                
        print(ifil)


#     if mirror:
#         outputfile = outputfile + '_mirror'
#     fout = open(outputfile, "w")
#     fout.write("# 0:rmin/2+rmax/2 1:paircounts\n")
#     for i in range(len(rbins[:-1])):
#         rrmin = rbins[i]
#         rrmax = rbins[i+1]
#         #Resp = sumwls_resp[i]*1.0/sumwls[i]

#         fout.write("%le\t%le\n"%(rrmin/2.0+rrmax/2.0, sumwls[i]))
#     fout.write("#OK")
#     fout.close()

    return 0


if __name__ == "__main__":
    
    run_pipe(mirror=False, HighSM = False)
    run_pipe(mirror=True, HighSM = False)


