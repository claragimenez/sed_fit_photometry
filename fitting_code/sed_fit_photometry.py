"""
SED-Fitting tool for spatially-resolved photometric observations.

NAME:
      SED_FIT_PHOTOMETRY
AUTHOR:
      Clara Giménez Arteaga, University of Copenhagen, clara.arteaga_at_nbi.ku.dk
      Gabriel Brammer, University of Copenhagen, gabriel.brammer_at_nbi.ku.dk
PURPOSE:
      SED-Fitting for spatially-resolved photometry. Can particularly be useful 
      to extract line fluxes from narrow-band imaging, by robustly fitting and 
      subtracting the continuum. Outputs physical properties such as the stellar 
      mass, star formation rate and visual extinction Av.
EXPLANATION:
      Further information on FIT_PHOTOMETRY algorithm can be found in
      Giménez-Arteaga et al. in prep.
CALLING SEQUENCE:
      master_table = fit_photometry(target,master_table,z,x0,y0,size,
                    mw_ebv,ha_line_filter,pab_line_filter,plot=True)
INPUTS:
          target: The name of the target, to be used in the output files.
    master_table: Catalog file.
        REDSHIFT: Redshift of the source.
      x0,y0,size: x and y coordinates, and size of the output cutouts.
          mw_ebv: Milky Way extinction E(B-V) from the source.
  ha_line_filter: Narrow-band filter that targets H-alpha.
 pab_line_filter: Narrow-band filter that targets Paschen-beta.
KEYWORDS:
            PLOT: Set this keyword to produce multiple plots of the
                  output parameters and cutouts of the target.
OUTPUTS:
              mt: Catalog file updated with output parameters from the fit.

PROCEDURES USED:
          FIT_PHOTOMETRY     -- Main program to perform the SED-fit and subtract
                                the continuum to obtain line fluxes.
          RGB_IMAGE          -- Obtain RGB image of the target.
     _INTEGRATE_TEMPFILT     -- Integrate template through a filter. From eazy-py.
"""


import matplotlib.pyplot as plt
import eazy
import numpy as np
from grizli import utils,prep
import astropy.io.fits as pyfits
from astropy.table import Table
import os
import copy
from scipy.optimize import nnls
import glob
from matplotlib.colors import LogNorm
from astropy.visualization import lupton_rgb
from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
import pysynphot as S
import eazy.filters
import eazy.templates
from grizli import utils

#----------------------------------------------------------------------------

def rgb_image(target='galaxy',file_name='galaxy*.fits',blue_f='f435w',green_f='f814w',red_f='f110w',plot=False):
      
    """
    Function that creates an RGB image of an input target.
    
    INPUTS:
        target: The name of the target, to be used in the output files.
     file_name: General form of the file names that contain the RGB filter images.
        blue_f: Filter in the blue band.
       green_f: Filter in the green band.
         red_f: Filter in the red band.
                 
    KEYWORDS:
          PLOT: Set this keyword to produce a plot of the two-dimensional
                RGB image.   
    OUTPUTS:
          Fits file with the RGB image.
    """
       
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.io.fits as pyfits
    import glob
    from grizli import utils
    from astropy.visualization import lupton_rgb
    
    files = glob.glob(file_name)
    files.sort()
    images = {}
    headers = {}
    bandpasses = {}

    for file in files:
        im = pyfits.open(file)
        filt = utils.get_hst_filter(im[0].header)
        for ext in [0,1]:
            if 'IM2FLAM' in im[ext].header:
                photflam = im[ext].header['IM2FLAM']
                headers[filt.lower()] = im[ext].header
                break
        images[filt.lower()] = im['SCI'].data
    
    blue = images[blue_f]*headers[blue_f]['IM2FLAM']/1.e-19
    green = images[green_f]*headers[green_f]['IM2FLAM']/1.e-19
    red = images[red_f]*headers[red_f]['IM2FLAM']/1.e-19
    rgb = lupton_rgb.make_lupton_rgb(red, green, blue, minimum=-0.1, stretch=1, Q=8)

    if plot:
        fig = plt.figure(figsize=[6,6])
        ax = fig.add_subplot(111)
        imsh = ax.imshow(rgb, origin='lower')
      
    return rgb

 #----------------------------------------------------------------------------

def _integrate_tempfilt(itemp, templ, self, REDSHIFT):
      
    """
    Function for integrating templates through filters. From eazy-py/photoz.
    
    INPUTS:
         itemp: Index of the template.
         templ: Template to integrate.
          self: Self object from eazy-py (eazy.photoz.PhotoZ object).
      REDSHIFT: Redshift of the source.
                 
    OUTPUTS:
          Synthetic photometry of the template integrated through the filter.
    """

    import eazy.utils as eazyutils
    
    NFILT = len(self.filters)
    igm = 1.

    galactic_ebv = self.param.params['MW_EBV']
    f99 = eazyutils.GalacticExtinction(EBV=galactic_ebv, Rv=3.1)
    Eb = self.param.params['SCALE_2175_BUMP']
    # Add bump with Drude profile in template rest frame
    width = 350
    l0 = 2175
    tw = templ.wave
    Abump = Eb/4.05*(tw*width)**2/((tw**2-l0**2)**2+(tw*width)**2)
    Fbump = 10**(-0.4*Abump)

    tempfilt = np.zeros(NFILT)
    lz = templ.wave*(1+REDSHIFT)
    igmz = 1.

    # Galactic Redenning        
    red = (lz > 910.) & (lz < 6.e4)
    A_MW = templ.wave*0.        
    A_MW[red] = f99(lz[red])

    F_MW = 10**(-0.4*A_MW)

    for ifilt in range(NFILT):
        fnu = templ.integrate_filter(self.filters[ifilt], 
                                     scale=igmz*F_MW*Fbump, 
                                     z=REDSHIFT, 
                                     include_igm=False)

        tempfilt[ifilt] = fnu

    return tempfilt

#----------------------------------------------------------------------------

def fit_photometry(target,master_table,REDSHIFT,x0,y0,size,mw_ebv,ha_line_filter,pab_line_filter,plot=True):
      
 """
      Function fitting spatially-resolved photometry with a set of SED templates. 
      Can extract line fluxes from narrow-band imaging, by robustly fitting and 
      subtracting the continuum. Outputs physical properties such as the stellar 
      mass, star formation rate and visual extinction Av.
    
      INPUTS:
          target: The name of the target, to be used in the output files.
    master_table: Catalog file.
        REDSHIFT: Redshift of the source.
      x0,y0,size: x and y coordinates, and size of the output cutouts.
          mw_ebv: Milky Way extinction E(B-V) from the source.
  ha_line_filter: Narrow-band filter that targets H-alpha.
 pab_line_filter: Narrow-band filter that targets Paschen-beta.
      KEYWORDS:
            PLOT: Set this keyword to produce multiple plots of the
                  output parameters and cutouts of the target.
      OUTPUTS:
              mt: Catalog file updated with output parameters from the fit.
 """

    slx = slice(x0-size, x0+size)
    sly = slice(y0-size, y0+size)
   # file_name='{0}binned_{1}_f*image.fits'.format(path,target)
   # master_table = '{0}binned_{1}_master_table.fits'.format(path,target)
    mt = Table.read(master_table) 
    
    template_files = """templates/spline_templates_v3/fsps_alpha_bin0_Av0.0.fits
    templates/spline_templates_v3/fsps_alpha_bin1_Av0.0.fits
    templates/spline_templates_v3/spline_age0.31_av0.0.fits
    templates/spline_templates_v3/spline_age0.62_av0.0.fits
    templates/spline_templates_v3/spline_age1.76_av0.0.fits
    templates/spline_templates_v3/spline_age7.18_av0.0.fits
    templates/spline_templates_v3/spline_age0.91_av0.0.fits""".split()

    tables = []
    for file in template_files:
        tab = utils.read_catalog(file)
        print(file, tab.colnames, tab['flux'].shape)
        tables.append(tab)

    cont_templ = []
    for tab, file in zip(tables, template_files):
        wclip = (tab['wave'] > 0.092*1e4) & (tab['wave'] < 3e4) 
        templ = eazy.templates.Template(arrays=(tab['wave'][wclip], tab['continuum'][wclip,0]), 
                                        name=os.path.basename(file), redfunc=eazy.templates.Redden(model='f99'))
        cont_templ.append(templ)
        
        # Line template from first in the list
    tab = tables[0]
    wclip = (tab['wave'] > 0.092*1e4) & (tab['wave'] < 3e4) 
    line_only = tab['flux'] - tab['continuum']
    line_templ = eazy.templates.Template(arrays=(tab['wave'][wclip], line_only[wclip,0]), name='Em.Lines', 
                                redfunc=eazy.templates.Redden(model='f99'))
    
        # Import photometric fluxes, using eazy self.fnu object
    zsp=REDSHIFT
    params = {}
    params['CATALOG_FILE'] = master_table
    params['MAIN_OUTPUT_FILE'] = target+'.eazypy'
    params['CAT_HAS_EXTCORR'] = 'n'
    params['MW_EBV'] = mw_ebv
    params['SYS_ERR'] = 0.05
    params['Z_STEP'] = 0.0002
    params['Z_MIN'] = np.maximum(zsp - 10*params['Z_STEP']*(1+zsp), 0)
    params['Z_MAX'] = zsp + 10*params['Z_STEP']*(1+zsp)
    params['PRIOR_ABZP'] = 23.9 
    params['PRIOR_FILTER'] = 241 # K
    params['PRIOR_FILE'] = 'templates/prior_K_TAO.dat'
    #  params['TEMPLATES_FILE'] = 'templates/spline_templates_v3/spline.param'
    params['FIX_ZSPEC'] = True

    translate_file = os.path.join(os.getenv('EAZYCODE'), 'inputs/zphot.translate.hst')
    self = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None, 
                        params=params, load_prior=True, load_products=False)

    #self.efnu = self.efnu_orig*1
    self.efnu = np.sqrt(self.efnu_orig**2+(self.param['SYS_ERR']*self.fnu)**2)
    
    all_templates = cont_templ
    all_templates += [line_templ]

    tab_temp = utils.GTable()
    tab_temp['temp'] = np.arange(1,len(all_templates)+1,dtype=int)
    sfr = []
    mass = []
    for i in tables:
        sfr.append(i.meta['SFR100'])
        mass.append(i.meta['stellar_mass'])
    for i in tables:
        sfr.append(0)
        mass.append(0)
        break
    tab_temp['SFR'] = np.array(sfr)
    tab_temp['mass'] = np.array(mass)
    
    Av_grid = np.arange(0.0,10.1,0.1)
    N_Av = len(Av_grid)
    N_Av2 = 16
    chi2_grid = np.zeros((N_Av,N_Av2,self.NOBJ))        
    Av_full_grid = np.zeros((N_Av,N_Av2,self.NOBJ,len(all_templates)))
    coeffs_grid = np.zeros((N_Av,N_Av2,self.NOBJ,len(all_templates)))
    templ_fluxes_grid = np.zeros((N_Av,N_Av2,self.NFILT,len(all_templates)))
    model_grid = np.zeros((N_Av,N_Av2,self.NFILT,self.NOBJ))
    
    vband = self.RES[155]    
    templ_vband = np.array([t.integrate_filter(vband, z=0) for t in all_templates])

    flux_intratio = np.array(_integrate_tempfilt(-1,line_templ,self,REDSHIFT)).T
        
    for j,Av in enumerate(Av_grid):
        print('Av=',Av)
        for temp in all_templates:
            temp.redfunc.Av = Av
        
        Av_i_list = [Av for temp in all_templates]
        Av_i = np.array(Av_i_list[:-1])
        
        min_av = Av
        #max_av = Av+1.27*Av
        max_av = 6*Av
        Av_line_grid = np.linspace(min_av,max_av,N_Av2)
        
        for k,Av_line in enumerate(Av_line_grid):
            all_templates[-1].redfunc.Av = Av_line 

            _tempfilt =  np.array([_integrate_tempfilt(i, templ, self, REDSHIFT) for i,templ in enumerate(all_templates)]).T
            templ_fluxes_grid[j,k,:,:] = _tempfilt

            for i in range(self.NOBJ):
                flux = self.fnu[i]*self.ext_redden
                err = self.efnu[i]*self.ext_redden
                sys_err = np.sqrt(err**2+(self.TEF(REDSHIFT)*flux)**2)

                # Fit it 
                _A = (_tempfilt.T/sys_err).T
                _y = flux/sys_err

                # Fit for the normalization coefficients
                _coeffs = nnls(_A, _y)
                coeffs_grid[j,k,i,:]=_coeffs[0]

                # Check the fit
                model = _tempfilt.dot(_coeffs[0])
                model_grid[j,k,:,i] = model
                chisq = ((model - flux)**2/sys_err**2).sum()
                chi2_grid[j,k,i]=chisq
                Av_full = np.append(Av_i,Av_line)
                Av_full_grid[j,k,i,:]=Av_full
                  
                
                _A = (_tempfilt.T/sys_err).T
                _y = flux/sys_err

                BOUNDED_DEFAULTS = {'method': 'bvls', 'tol': 1.e-8, 'verbose': 0}
                LINE_BOUNDS = [-6.e3, 6.e5] #uJy 
                bounded_kwargs = BOUNDED_DEFAULTS

                NTEMP = len(all_templates)

                lower_bound = np.zeros(NTEMP)
                upper_bound = np.ones(NTEMP)*np.inf

                lower_bound[:-1] = 0.
                lower_bound[-1] = -1.e+18 #LINE_BOUNDS[0]
                upper_bound[:-1] = 1.e+30
                upper_bound[-1] = 1.e+30 #LINE_BOUNDS[1]

                  # Bounded Least Squares
                func = scipy.optimize.lsq_linear
                bounds = (lower_bound, upper_bound)
                lsq_out = func(_A, _y, bounds=bounds, **bounded_kwargs)
                _coeffs = lsq_out.x
                coeffs_grid[j,k,i,:]=_coeffs
    
    # Get best fit coefficients
    chi2_bestindex_cont = np.argmin(chi2_grid,axis=0)
    chi2_best_cont = chi2_grid.min(axis=0)
    Av_best = Av_grid[chi2_bestindex_cont][0,:]
    coeffs_best_cont = np.zeros((N_Av2,self.NOBJ,len(all_templates)))
    templ_fluxes_best_cont = np.zeros((N_Av2,self.NOBJ,self.NFILT,len(all_templates)))
    Av_cont_best_cont = np.zeros((N_Av2,self.NOBJ,len(all_templates)))
    model_best_cont = np.zeros((N_Av2,self.NFILT,self.NOBJ))

    for k,_ in enumerate(coeffs_best_cont[:,0,0]):
        for i,_ in enumerate(coeffs_best_cont[0,:,0]):
            coeffs_best_cont[k,i,:]=coeffs_grid[chi2_bestindex_cont[k,i],k,i,:]
            templ_fluxes_best_cont[k,i,:,:] = templ_fluxes_grid[chi2_bestindex_cont[k,i],k,:,:]
            Av_cont_best_cont[k,i,:] = Av_full_grid[chi2_bestindex_cont[k,i],k,i,:]
            model_best_cont[k,:,i] = model_grid[chi2_bestindex_cont[k,i],k,:,i]

    chi2_bestindex = np.argmin(chi2_best_cont,axis=0)
    chi2_best = chi2_best_cont.min(axis=0)
    coeffs_best = np.zeros((self.NOBJ,len(all_templates)))
    templ_fluxes_best = np.zeros((self.NOBJ,self.NFILT,len(all_templates)))
    Av_cont_best = np.zeros((self.NOBJ,len(all_templates)))
    model_best = np.zeros((self.NFILT,self.NOBJ))

    for i,_ in enumerate(coeffs_best[:,0]):
        coeffs_best[i,:]= coeffs_best_cont[chi2_bestindex[i],i,:]
        templ_fluxes_best[i,:,:] = templ_fluxes_best_cont[chi2_bestindex[i],i,:,:]
        Av_cont_best[i,:] = Av_cont_best_cont[chi2_bestindex[i],i,:]
        model_best[:,i] = model_best_cont[chi2_bestindex[i],:,i]

    var_cont = np.zeros((self.NOBJ, 2))
    var_line = np.zeros((self.NOBJ, 2))
    flux_cont = np.zeros((self.NOBJ, self.NFILT))
    flux_line = np.zeros((self.NOBJ, self.NFILT))

    ha_ix = self.flux_columns.index(ha_line_filter)
    pab_ix = self.flux_columns.index(pab_line_filter)
    
    Av_stars = np.zeros(self.NOBJ)
    mass = np.zeros(self.NOBJ)
    SFR = np.zeros(self.NOBJ)
    
    dL = cosmo.luminosity_distance(REDSHIFT).to(u.cm)
    
    uJy_to_cgs = u.microJansky.to(u.erg/u.s/u.cm**2/u.Hz)
    fnu_units = u.erg/u.s/u.cm**2/u.Hz
    fnu_scl = 10**(-0.4*(self.param.params['PRIOR_ABZP']-23.9))*uJy_to_cgs
    template_fnu_units = (1*u.solLum / u.Hz)
    to_physical = fnu_scl*fnu_units*4*np.pi*dL**2/(1+REDSHIFT)
    to_physical /= (1*template_fnu_units).to(u.erg/u.second/u.Hz)
        
    for i in range(self.NOBJ): 
        coeffs = coeffs_best[i,:]
        Av_i = Av_cont_best[i,:][:-1]
        
        if (coeffs[-1]==0)|(coeffs.sum()==0):
            var_cont[i,:] = 0
            var_line[i,:] = 0
            continue
        else:
            T_nonzero = templ_fluxes_best[i,:,:][:,coeffs!=0]
            coeffs_nonzero = coeffs[coeffs != 0]
            self.efnu[self.efnu ==0] = -99
            T_pos = T_nonzero.T/self.efnu[i,:]

            flux_cont[i,:] = (T_nonzero[:,:-1].dot(coeffs_nonzero[:-1]))
            flux_line[i,:] = (T_nonzero[:,-1:].dot(coeffs_nonzero[-1:]))

            covm =np.matrix(np.dot(T_pos, T_pos.T)).I.A
            cont_mask = np.ones(coeffs_nonzero.shape[0])
            cont_mask[-1] = 0
            line_mask = cont_mask == 0

            draws = np.random.multivariate_normal(coeffs_nonzero, covm, size=2048)

            cont_model = T_nonzero.dot((draws*cont_mask).T).T
            line_model = T_nonzero.dot((draws*line_mask).T).T

            cont_ha = cont_model[:,ha_ix]
            line_ha = line_model[:,ha_ix]
            cont_pab = cont_model[:,pab_ix]
            line_pab = line_model[:,pab_ix]

            var_cont[i,:] = [np.std(cont_ha),np.std(cont_pab)]
            var_line[i,:] = [np.std(line_ha),np.std(line_pab)]
        
        # calculate Av stars
        c_i = (coeffs*templ_vband)[:-1]
        Av_tau = 0.4*np.log(10)
        temp_par = np.exp(Av_tau*Av_i)
        par_value = c_i.dot(temp_par)
        par_denom = c_i.dot(temp_par*0+1)
        Av_stars[i] = np.log(par_value/par_denom) / Av_tau
        
        # SFR and stellar mass
        coeffs_rest = ((coeffs.T*to_physical).T)
        coeffs_rest = np.array(coeffs_rest)
        mass[i] = coeffs_rest.dot(tab_temp['mass'])
        SFR[i] = coeffs_rest.dot(tab_temp['SFR'])

    # Subtract continuum
    ha_flux_fnu = (self.fnu[:,ha_ix]-flux_cont[:,ha_ix]) 
    ha_err_fnu = var_line[:,0]
    pab_flux_fnu = self.fnu[:,pab_ix]-flux_cont[:,pab_ix]
    pab_err_fnu = var_line[:,-1]
    
    # Convert to line fluxes
    
    # H-alpha
    # emission line "template"
    w0 = 0.65628e4*(1+REDSHIFT)
    s0 = (30./3.e5*w0) # sigma width 30 km/s
    # Gaussian line 
    line_flux = 1.e-17 # erg/s/cm2

    # Pysynphot objects
    flat = S.FlatSpectrum(1, fluxunits='microJy')
    sp = S.GaussianSource(line_flux, w0, s0)
    bp = S.ObsBandpass('wfc3,uvis1,{0}'.format(f[1]))

    # Integrate with pysynphot
    # Get image value conversion with flat spectrum and convert the line flux
    flat_eps = S.Observation(flat, bp).countrate()*u.electron/u.second/u.microJansky
    line_eps = S.Observation(sp, bp).countrate()*u.electron/u.second
    pysyn_fnu = line_eps / flat_eps
    
    ha_flux = ha_flux_fnu*(line_flux/pysyn_fnu)
    ha_err = ha_err_fnu*(line_flux/pysyn_fnu)
    int_ratio_ha = flux_intratio[ha_ix]*(line_flux/pysyn_fnu)
    
    # Paschen-beta
    # emission line "template"
    w0 = 1.28e4*(1+REDSHIFT)
    s0 = (30./3.e5*w0) # sigma width 30 km/s
    # Gaussian line 
    line_flux = 1.e-17 # erg/s/cm2

    # Pysynphot objects
    flat = S.FlatSpectrum(1, fluxunits='microJy')
    sp = S.GaussianSource(line_flux, w0, s0)
    bp = S.ObsBandpass('wfc3,ir,{0}'.format(f[2]))

    # Integrate with pysynphot
    # Get image value conversion with flat spectrum and convert the line flux
    flat_eps = S.Observation(flat, bp).countrate()*u.electron/u.second/u.microJansky
    line_eps = S.Observation(sp, bp).countrate()*u.electron/u.second
    pysyn_fnu = line_eps / flat_eps
    
    pab_flux = pab_flux_fnu*(line_flux/pysyn_fnu)
    pab_err = pab_err_fnu*(line_flux/pysyn_fnu)
    int_ratio_pab = flux_intratio[pab_ix]*(line_flux/pysyn_fnu)
    
    mt['ha_flux'] = ha_flux
    mt['pab_flux'] = pab_flux
    mt['ha_err'] = ha_err
    mt['pab_err'] = pab_err
    
    im = '{0}binned_{1}_{2}_image.fits'.format(path,target,f[2])    
    sci = np.cast[np.float32](pyfits.open(im)['SCI'].data)
    seg = np.cast[np.float32](pyfits.open('{0}binned_{1}_seg.fits'.format(path,target))[0].data)        
    ha_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['ha_flux']/mt['npix'])
    pab_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['pab_flux']/mt['npix'])
    ha_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['ha_err']/mt['npix'])
    pab_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['pab_err']/mt['npix'])

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=ha_im,name='SCI')
    err_extn = pyfits.ImageHDU(data=ha_im_err,name='ERR')
    hdul = pyfits.HDUList([primary_extn,sci_extn,err_extn])
    hdul.writeto('fit_output/{0}_ha.fits'.format(target),output_verify='fix',overwrite=True)
    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=pab_im,name='SCI')
    err_extn = pyfits.ImageHDU(data=pab_im_err,name='ERR')
    hdul = pyfits.HDUList([primary_extn,sci_extn,err_extn])
    hdul.writeto('fit_output/{0}_pab.fits'.format(target),output_verify='fix',overwrite=True)

    chi2_im = prep.get_seg_iso_flux(sci, seg, mt, fill=chi2_best)
    av_im_gas = prep.get_seg_iso_flux(sci, seg, mt, fill=Av_cont_best[:,-1])
    av_im_cont = prep.get_seg_iso_flux(sci, seg, mt, fill=Av_cont_best[:,0])
    av_im_stars = prep.get_seg_iso_flux(sci, seg, mt, fill=Av_stars)
    sfr_im = prep.get_seg_iso_flux(sci, seg, mt, fill=SFR/mt['npix'])
    mass_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mass/mt['npix'])

    mt['chi2'] = chi2_best/self.NFILT
    mt['SFR'] = SFR
    mt['stellar_mass'] = mass
    mt['Av_line_templ'] = Av_cont_best[:,-1]
    mt['Av_stars'] = Av_stars
    
    ext = ha_im/pab_im
    ext_mt = ha_flux/pab_flux
    ext_err = np.sqrt((ha_im_err/pab_im)**2+(pab_im_err*(ha_im/(pab_im**2)))**2)
    int_ratio =  int_ratio_ha/int_ratio_pab
    print('Intrinsic Ratio = ',int_ratio)

    #k = [1.27,3.33] # calzetti+00
    k = [0.76538229,2.356559] # fitzpatrick+99
    R_v = 3.1
    ebv = (2.5/(k[0]-k[1]))*np.log10(ext/int_ratio)
    av_lines = R_v*ebv
    av_lines_err = abs(R_v*(2.5/((int_ratio*np.log(10))*(k[0]-k[1])))*(1/ext)*ext_err)
    
    ebv_mt = (2.5/(k[0]-k[1]))*np.log10(ext_mt/int_ratio)
    av_lines_mt = R_v*ebv_mt
    mt['Av_decrement'] = av_lines_mt
    
   # primary_extn = pyfits.PrimaryHDU()
   # sci_extn = pyfits.ImageHDU(data=av_lines,name='SCI')
   # err_extn = pyfits.ImageHDU(data=av_lines_err,name='ERR')
   # hdul = pyfits.HDUList([primary_extn,sci_extn,err_extn])
   # hdul.writeto('fit_output/{0}_av_decrement.fits'.format(target),output_verify='fix',overwrite=True)

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=av_im_gas,name='SCI')
    hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto('fit_output/{0}_av_line_templ.fits'.format(target),output_verify='fix',overwrite=True)

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=av_im_stars,name='SCI')
    hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto('fit_output/{0}_av_stars.fits'.format(target),output_verify='fix',overwrite=True)

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=chi2_im/self.NFILT,name='SCI')
    hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto('fit_output/{0}_chi2.fits'.format(target),output_verify='fix',overwrite=True)

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=sfr_im,name='SCI')
    hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto('fit_output/{0}_sfr.fits'.format(target),output_verify='fix',overwrite=True)

    primary_extn = pyfits.PrimaryHDU()
    sci_extn = pyfits.ImageHDU(data=mass_im,name='SCI')
    hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto('fit_output/{0}_mass.fits'.format(target),output_verify='fix',overwrite=True)

    mt.write('fit_output/{0}_master_table.fits'.format(target), overwrite=True)

        ### PLOT RESULTS
    if plot:
        rgb = rgb_image(target,file_name,blue_f,green_f,red_f)     
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(16,11),gridspec_kw={'hspace': 0.01, 'wspace': 0.01}, frameon=False)
        ax1.imshow(rgb[sly, slx],origin='lower')
        ax1.text(30, 30, r'{0}/{1}/{2}'.format(blue_f,green_f,red_f), fontsize=15, weight=20, color='black', bbox={'facecolor':'white', 'alpha':0.8,'pad': 8})
        ax1.text(30, 2*size-40, r'{0}'.format(target), fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.8, 'pad': 8})
        ax1.axis('off')
        ax2.imshow(ha_im[sly, slx],norm=LogNorm(),origin='lower')
        ax2.text(30, 30, r'H$\alpha$', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.8,'pad': 8})
        ax2.axis('off')
        ax3.imshow(pab_im[sly, slx],norm=LogNorm(),origin='lower')
        ax3.text(30, 30, r'Pa$\beta$', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.8,'pad': 8})
        ax3.axis('off')
        ax4.imshow(av_lines[sly, slx],vmin=0,vmax=6,cmap='Spectral_r',origin='lower')
        ax4.text(30, 30, r'$A_V$ Ha/PaB', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad': 8})
        ax4.axis('off')
        ax5.imshow(av_im_gas[sly, slx],vmin=0,vmax=6,cmap='Spectral_r',origin='lower')
        ax5.text(30, 30, r'$A_V$ line templates', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad': 8})
        ax5.axis('off')
        ax6.imshow(av_im_stars[sly, slx], vmin=0,vmax=6,cmap='Spectral_r',origin='lower')
        ax6.text(30, 30, r'$A_V$ stars', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad':10})
        ax6.axis('off')
        fig.tight_layout(pad=0)
        plt.show()
        
        SFR_tot = np.nansum(sfr_im[sly,slx].flatten())
        mass_tot = np.nansum(mass_im[sly,slx].flatten())
        print('SFR = %.1f solMass/yr'%(SFR_tot))
        print('Stellar Mass = %.2e solMass'%(mass_tot))
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(22,10))
        im1 = ax1.imshow(sfr_im[sly,slx],norm=LogNorm(vmin=1e-6,vmax=1e-1),cmap='Spectral',origin='lower')
        ax1.axis('off')
        cbar1=plt.colorbar(im1, cax = fig.add_axes([0.485, 0.17, 0.01, 0.67]))
        cbar1.set_label(r'SFR [solMass/yr]',fontsize=20)
        cbar1.ax.tick_params(labelsize=20)
        im2 = ax2.imshow(mass_im[sly,slx],norm=LogNorm(),cmap='inferno',origin='lower')
        ax2.axis('off')
        cbar2=plt.colorbar(im2, cax = fig.add_axes([0.91, 0.17, 0.01, 0.67]))
        cbar2.set_label(r'Stellar Mass [solMass]',fontsize=20)
        cbar2.ax.tick_params(labelsize=20)
        plt.show()
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(24,10))
        im1 = ax1.imshow(chi2_im[sly,slx]/self.NFILT,vmin=-0.1,vmax=10,cmap='Spectral_r',origin='lower')
        ax1.axis('off')
        cbar1=plt.colorbar(im1, cax = fig.add_axes([0.46, 0.17, 0.01, 0.67]))
        cbar1.set_label(r'$\chi^2$',fontsize=20)
        cbar1.ax.tick_params(labelsize=20)
        mass_lin = np.linspace(7.5,12.5,1000)
        z = 0.018126
        a = 0.70-0.13*z
        b = 0.38 + 1.14*z-0.19*z**2
        ms = a*(mass_lin-10.5)+b
        ax2.scatter(np.log10(mass_tot),np.log10(SFR_tot),s=200,c='darkturquoise',zorder=99,alpha=1)
        ax2.set_ylabel('log($SFR$) [$M_{\odot}/yr$]',fontsize=20)
        ax2.set_xlabel('log($M$) [$M_{\odot}$]',fontsize=20)
        ax2.plot(mass_lin,ms,'k-',lw=4,label='Whitaker+12 (0.0001<z<0.035)')
        ax2.plot(mass_lin,np.log10((10**ms)*3),'k--',lw=3)
        ax2.plot(mass_lin,np.log10((10**ms)/3),'k--',lw=3)
        ax2.legend(fontsize=20)
        plt.show()
        
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(22,10))
        ax1.scatter(av_lines[sly,slx].flatten(),av_im_gas[sly,slx].flatten(),marker='o',s=20,c='r',alpha=0.5,label='F99, Rv = 3.1')
        ax1.plot([0,6],[0,6],'--',c='k',zorder=-99)
        ax1.set_xlabel('Av Ha/PaB',fontsize=25)
        ax1.set_ylabel('Av Templates',fontsize=25)
        ax1.set_ylim(0,6)
        ax1.set_xlim(0,6)
        ax1.legend(fontsize=25)
        ax2.scatter(av_lines[sly,slx].flatten(),av_im_stars[sly,slx].flatten(),marker='o',s=15,c='r',alpha=0.5)
        y = np.linspace(0,10,100)
        ax2.plot(y,y,'k-',label='$A_{extra}$ = 0',lw=2)
        ax2.plot(y*1.27+y,y,'k--',label='$A_{extra}$ = 1.27 $A_{cont}$',lw=2)
        ax2.plot(y+0.9*y-0.15*y**2,y,'k-.',label='$A_{extra}$ = 0.9 $A_{cont}$ - 0.15 $A_{cont}^2$',lw=2)
        ax2.set_xlabel('$A_V$ gas',fontsize=25)
        ax2.set_ylabel('$A_V$ stars',fontsize=25)
        ax2.set_ylim(0,10)
        ax2.set_xlim(0,10)
        ax2.legend(fontsize=25)
        fig.tight_layout()
        plt.show()

        primary_extn = pyfits.PrimaryHDU()
        sci_extn = pyfits.ImageHDU(data=rgb,name='SCI')
        hdul = pyfits.HDUList([primary_extn,sci_extn])
        hdul.writeto('fit_output/{0}_rgb.fits'.format(target),output_verify='fix',overwrite=True)

    return mt
  
  #----------------------------------------------------------------------------
