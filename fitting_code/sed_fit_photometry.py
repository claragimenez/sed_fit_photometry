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
      Giménez-Arteaga et al. 2022
CALLING SEQUENCE:
     output_catalog = fit_catalog(target,catalog,template_files,
            template_params_file, z, mw_ebv,lower_bounds, upper_bounds, 
            intrinsic_ratio, nii_to_ha, reddening_func,reddening_func_lines,
            R_v, ha_sfr_conv_factor,niter,seg_im,blue_f,green_f,red_f,
            x0,y0,size,file_name,output_path,save_output,plot)       
             
OUTPUTS:
              mt: Catalog file updated with output parameters from the fit.

PROCEDURES USED:
          FIT_CATALOG     -- Main function to perform the SED-fit and subtract
                                the continuum to obtain line fluxes and
                                infer physical properties.
            RGB_IMAGE     -- Obtain RGB image of the target.
  _INTEGRATE_TEMPFILT     -- Integrate template through a filter. From eazy-py.
 INIT_CONTINUUM_TEMPL     -- Function to fetch / initiate the continuum templates.
    SAVE_OUTPUT_IMAGE     -- Function to save results in output file.
      GET_PERCENTILES     -- Function to calculate the 16th, 50th and 84th percentiles of
                             an array given a PDF.
  
"""

import numpy as np
import matplotlib.pyplot as plt
import eazy
import os
import glob
import copy
import scipy
from grizli import utils,prep
import astropy.io.fits as pyfits
from astropy.table import Table, vstack
from matplotlib.colors import LogNorm
import astropy.units as u
import eazy.templates
from scipy.stats import norm
import scipy.ndimage as nd
from astropy.cosmology import WMAP9 as cosmo
print('EAZYCODE = '+os.getenv('EAZYCODE'))

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

def init_continuum_templ(template_files=None, reddening_func='c00'):
      
 """
      Function to fetch / initiate the continuum templates.
    
      INPUTS:
  template_files: Continuum templates that can be input by the user

      KEYWORDS:
  reddening_func: Attenuation function to use when reddening the continuum
                  templates.
      OUTPUTS:
      cont_templ: Eazy template object, with the reddening function specified.
  template_files: Continuum template files.
  
 """
    
    if template_files is None:
        template_files = """templates/hr_sfhz_13_bin0_av0.01.fits
           templates/hr_sfhz_13_bin1_av0.01.fits
           templates/hr_sfhz_13_bin2_av0.01.fits
           templates/hr_sfhz_13_bin3_av0.01.fits""".split()

    tables = []
    for file in template_files:
        tab = utils.read_catalog(file)
        print(file, tab.colnames, tab['flux'].shape)
        tables.append(tab)

    cont_templ = []
    for tab, file in zip(tables, template_files):
        wclip = (tab['wave'] > 0.092*1e4) & (tab['wave'] < 3e4) 
        templ = eazy.templates.Template(arrays=(tab['wave'][wclip], tab['continuum'][wclip,0]),
                        name=os.path.basename(file), redfunc=eazy.templates.Redden(model=reddening_func, Rv=4.05))
        cont_templ.append(templ)
    
    return cont_templ, template_files


#----------------------------------------------------------------------------

def get_percentiles(pdf, data, so_data=None, perc_levels=[0.16,0.5,0.84]):
      
"""
      Function to calculate the 16th, 50th and 84th percentiles of
      an array given a PDF.
    
      INPUTS:
            pdf: Probability density function.
           data: Array for which to calculate the percentiles given a PDF.

      KEYWORDS:
        so_data: Sorted data.
       
      OUTPUTS:
      data_perc: 16th, 50th and 84th percentiles of the input data.
       data_unc: Uncertainty of the given data.
  
 """
    
    NOBJ = data.shape[-1]
    if so_data is None:
        so_data = np.argsort(data, axis=0)
    data_perc = np.zeros([NOBJ, 3])
    for iobj in range(NOBJ):
        so_i = so_data[:,iobj]
        data_perc[iobj,:] = np.interp(perc_levels,np.cumsum(pdf[:,iobj][so_i]),data[:,iobj][so_i])
    
    data_unc = (data_perc[:,2] - data_perc[:,0])/2.

    return data_perc, data_unc


#----------------------------------------------------------------------------

def save_output_image(im, sci, err, target, output_name, output_path, comment):   
            
"""
      Function to save results in output file.
    
      INPUTS:
            im: Original image from which we will retrieve the WCS information.
           sci: Science image that we want to save.
           err: Uncertainty image that we save on the error extension.

      KEYWORDS:
        target: Name of the object
   output_name: Name of the output file.
   output_path: Location of the output file.
       
      OUTPUTS:
          hdul: HDU object with all input information saved.
  
 """
    
    import grizli
    import astropy.wcs as pywcs

    wcs = pywcs.WCS(im[1].header)
    header_wcs = grizli.utils.to_header(wcs)
    header_wcs['COMMENT'] = comment

    primary_extn = pyfits.PrimaryHDU(header = header_wcs)
    sci_extn = pyfits.ImageHDU(data=sci,name='SCI', header=header_wcs)
    if err is not None:
        err_extn = pyfits.ImageHDU(data=err,name='ERR')
        hdul = pyfits.HDUList([primary_extn,sci_extn,err_extn])
    else:
        hdul = pyfits.HDUList([primary_extn,sci_extn])
    hdul.writeto(f'{output_path}/{target}_{output_name}.fits',output_verify='fix',overwrite=True)

    return hdul   


#----------------------------------------------------------------------------

def fit_catalog(target='galaxy',catalog='catalog.fits',template_files=None, template_params_file=None, z=0., mw_ebv=0.,lower_bounds = [0., -1.e+18], 
    upper_bounds = [1.e+30, 1.e+30], intrinsic_ratio = 17.56, nii_to_ha=0.55, reddening_func ='c00', reddening_func_lines = 'mw', R_v = 3.1, ha_sfr_conv_factor=7.9e-42 / 1.8,
    niter = 5,seg_im = 'seg.fits',blue_f='f435w',green_f='f814w', red_f = 'f110w',x0=100,y0=100,size=50,file_name='galaxy*.fits',
    output_path='output',save_output=True,plot=False):     

  """
      Function fitting spatially-resolved photometry with a set of SED templates. 
      Can extract line fluxes from narrow-band imaging, by robustly fitting and 
      subtracting the continuum. Outputs physical properties such as the stellar 
      mass, star formation rate and visual extinction Av.
    
      INPUTS:
          target: The name of the target, to be used in the output files.
         catalog: Catalog file.
  template_files: Continuum template files.
  template_params_file: Parameters file of the continuum templates.
               z: Redshift of the source.
          mw_ebv: Milky Way extinction E(B-V) from the source.
    lower_bounds: Lower bounds for the line fluxes in the bounded least
                  squared optimization method.
    upper_bounds: Upper bounds for the line fluxes in the bounded least
                  squared optimization method.
 intrinsic_ratio: Intrinsic Ha/PaB ratio, as tabulated by e.g. Osterbrock+89
       nii_to_ha: [NII] to H-alpha ratio, to apply as a correction to the
                  H-alpha flux due to contamination from [NII] line to 
                  the narrowband images.
  reddening_func: Reddening function to apply to the continuum templates.
  reddening_func_lines: Reddening function to apply to the line emission templates.
             R_v: Conversion factor between hydrogen recombination lines ratios
                  and visual extinction (A_V).
  ha_sfr_conv_factor: Conversion factor from H-alpha luminosity to SFR.
           niter: Number of iterations to calculate percentiles.
          seg_im: Segmentation image.
          blue_f: Filter in the blue band.
         green_f: Filter in the green band.
           red_f: Filter in the red band.
      x0,y0,size: x and y coordinates, and size of the output cutouts.
       file_name: General form of the file names that contain the RGB filter images.
     output_path: Location of the output files.

      KEYWORDS:
     save_output: Set this keyword to produce and save the output files.
            plot: Set this keyword to produce multiple plots of the
                  output parameters and cutouts of the target.
      OUTPUTS:
              mt: Catalog file updated with output parameters from the fit.
 """
   
    mt = Table.read(catalog) 
    BOUNDED_DEFAULTS = {'method': 'bvls', 'tol': 1.e-8, 'verbose': 0}

    # Initialise continuum templates
    cont_templ, template_files = init_continuum_templ(template_files, reddening_func)

    # Manually create H-alpha and Paschen-beta line templates
    lwave = np.arange(920, 3.e4, 0.5)
    # Ha+[NII]
    w0_ha = 6564.61
    w0_nii1 = 6549.86
    w0_nii2 = 6585.27
    s0_ha = (100./3.e5*w0_ha) # sigma width 100 km/s
    s0_nii1 = (100./3.e5*w0_nii1)
    s0_nii2 = (100./3.e5*w0_nii2)
    lflux_ha = 1/np.sqrt(2*np.pi*s0_ha**2)*np.exp(-(lwave-w0_ha)**2/2/s0_ha**2)
    lflux_nii1 = 1/np.sqrt(2*np.pi*s0_nii1**2)*np.exp(-(lwave-w0_nii1)**2/2/s0_nii1**2)
    lflux_nii2 = 1/np.sqrt(2*np.pi*s0_nii2**2)*np.exp(-(lwave-w0_nii2)**2/2/s0_nii2**2)
    ha_nii_ratio = 1/nii_to_ha
    nii_ratio = 2.95
    lflux = (ha_nii_ratio/(ha_nii_ratio+1))*lflux_ha 
    lflux += (1/(1+ha_nii_ratio))*((1/(1+nii_ratio))*lflux_nii1+(nii_ratio/(1+nii_ratio))*lflux_nii2)
    line_templ_ha = eazy.templates.Template(arrays=(lwave, lflux), name='Ha')
    # PaB
    w0 = 1.28e4 
    s0 = (100./3.e5*w0)  
    lflux = 1/np.sqrt(2*np.pi*s0**2)*np.exp(-(lwave-w0)**2/2/s0**2)
    line_templ_pab = eazy.templates.Template(arrays=(lwave, lflux), name='PaB')

    all_templates = cont_templ + [line_templ_ha, line_templ_pab]

    # Import photometric fluxes, using eazy self.fnu object
    params = {}
    params['CATALOG_FILE'] = catalog
    params['CAT_HAS_EXTCORR'] = 'n'
    params['MW_EBV'] = mw_ebv
    params['SYS_ERR'] = 0.05
    params['Z_STEP'] = 0.0002
    params['Z_MIN'] = np.maximum(z - 10*params['Z_STEP']*(1+z), 0)
    params['Z_MAX'] = z + 10*params['Z_STEP']*(1+z)
    params['PRIOR_ABZP'] = 23.9 
    params['PRIOR_FILTER'] = 241 # K
    params['PRIOR_FILE'] = 'templates/prior_K_TAO.dat'
    params['TEMPLATES_FILE'] = 'templates/hr_sfhz_13_c3k.param'
    params['FIX_ZSPEC'] = True

    translate_file = os.path.join(os.getenv('EAZYCODE'), 'inputs/zphot.translate.hst')
    self = eazy.photoz.PhotoZ(param_file=None, translate_file=translate_file, zeropoint_file=None, 
                        params=params, load_prior=True, load_products=False)

    self.efnu = np.sqrt(self.efnu_orig**2+(self.param['SYS_ERR']*self.fnu)**2)

    if template_params_file is None:
        templ_params_file = 'templates/hr_sfhz_13_c3k.param.fits'
    else:
        templ_params_file = template_params_file

    tab_temp2 = Table.read(templ_params_file)

    tab_temp = utils.GTable()
    tab_temp['temp'] = np.arange(1,len(all_templates)+1,dtype=int)
    sfr = []
    mass = []
    for i in tab_temp2['file']:
        print(i)
        k = glob.glob(f'*/*{i}*.fits')[0]
        if k in template_files:
            ix = np.where(tab_temp2['file']==i)
            sfr.append(tab_temp2[ix]['sfr'][0][0])
            mass.append(tab_temp2[ix]['mass'][0][0])
    for j in range(2):
        sfr.append(0)
        mass.append(0)

    tab_temp['SFR'] = np.array(sfr)
    tab_temp['mass'] = np.array(mass)

    # Initialise grids
    Av_grid = np.arange(0.0,5.05,0.05)
    N_Av = len(Av_grid)
    chi2_grid = np.zeros((N_Av,self.NOBJ))       
    Av_full_grid = np.zeros((N_Av,self.NOBJ))  
    coeffs_grid = np.zeros((N_Av,self.NOBJ,len(all_templates)))
    templ_fluxes_grid = np.zeros((N_Av,self.NFILT,len(all_templates)))

    # Calculate V-band normalisation
    vband = self.RES[155]    
    templ_vband = np.array([t.integrate_filter(vband, z=0) for t in all_templates[:-2]])

    # Define some fitting parameters & Bounded Least Squares function
    bounded_kwargs = BOUNDED_DEFAULTS
    NTEMP = len(all_templates)
    lower_bound = np.zeros(NTEMP)
    upper_bound = np.ones(NTEMP)*np.inf
    lower_bound[:-2] = lower_bounds[0]
    lower_bound[-2:] = lower_bounds[-1]
    upper_bound[:-2] = upper_bounds[0]
    upper_bound[-2:] = upper_bounds[-1]
    bounds = (lower_bound, upper_bound)
    func = scipy.optimize.lsq_linear 

    for j,Av in enumerate(Av_grid):
        print('Av = %.2f'%Av)
        # Redden continuum templates
        for temp in all_templates[:-2]:
            temp.redfunc.Av = Av        
        
        # Integrate templates through the filters
        _tempfilt =  np.array([_integrate_tempfilt(i, templ, self, z) for i,templ in enumerate(all_templates)]).T
        templ_fluxes_grid[j,:,:] = _tempfilt
        
        # SED fit
        for i in range(self.NOBJ):
            flux = self.fnu[i]*self.ext_redden
            err = self.efnu[i]*self.ext_redden
            sys_err = np.sqrt(err**2+(self.TEF(z)*np.maximum(flux,0))**2)  # clip to positive fluxes

            _A = (_tempfilt.T/sys_err).T
            _y = flux/sys_err

            lsq_out = func(_A, _y, bounds=bounds, **bounded_kwargs)
            _coeffs = lsq_out.x
            coeffs_grid[j,i,:]=_coeffs

            # Check the fit
            model = _tempfilt.dot(_coeffs)
            chisq = ((model - flux)**2/sys_err**2).sum()
            chi2_grid[j,i]=chisq
            Av_full_grid[j,i]=Av

    # Covariance Matrix
    n_draws = 500
    print('Calculating covariance matrix')
    draws_grid = np.zeros((N_Av, n_draws, self.NOBJ, len(all_templates)))
    chisq_grid = np.zeros((N_Av, n_draws, self.NOBJ))
    Av_full_grid_draws = np.zeros((N_Av, n_draws, self.NOBJ))
    model_grid = np.zeros((N_Av, n_draws, self.NFILT,self.NOBJ))

    for i in range(self.NOBJ): 
        flux = self.fnu[i]*self.ext_redden
        err = self.efnu[i]*self.ext_redden
        sys_err = np.sqrt(err**2+(self.TEF(z)*flux)**2)
        sys_err[self.efnu[i] <= 0] = 1e10

        for j in range(N_Av):
            coeffs = coeffs_grid[j,i,:]   
            if (coeffs[-1]==0)|(coeffs.sum()==0):
                continue
            else:
                T_nonzero = templ_fluxes_grid[j,:,:][:,coeffs!=0] 
                coeffs_nonzero = coeffs[coeffs != 0]
                T_pos = T_nonzero.T/sys_err
                try:
                    covm = eazy.utils.safe_invert(np.dot(T_pos, T_pos.T))
                except:
                    covm = np.matrix(np.dot(T_pos, T_pos.T)).I.A
                draws = np.random.multivariate_normal(coeffs_nonzero, covm, size=n_draws)
                draws_grid[j,:,i,coeffs!=0] = draws.T
                model = T_nonzero.dot(draws.T).T
                model_grid[j,:,:,i] = model
                chisq = ((model - flux)**2/sys_err**2).sum(axis=1)
                chisq_grid[j,:,i]=chisq
                Av_full_grid_draws[j,:,i] = Av_full_grid[j,i]

    reshape = N_Av*n_draws
    draws_grid = np.reshape(draws_grid,newshape=(reshape,self.NOBJ, len(all_templates)))
    Av_full_grid_draws = np.reshape(Av_full_grid_draws,newshape=(reshape,self.NOBJ))
    chisq_grid = np.reshape(chisq_grid,newshape=(reshape,self.NOBJ))
    model_grid = np.reshape(model_grid,newshape=(reshape,self.NFILT,self.NOBJ))
    
    # Unit conversions
    dL = cosmo.luminosity_distance(z).to(u.cm)
    uJy_to_cgs = u.microJansky.to(u.erg/u.s/u.cm**2/u.Hz)
    fnu_units = u.erg/u.s/u.cm**2/u.Hz
    fnu_scl = 10**(-0.4*(self.param.params['PRIOR_ABZP']-23.9))*uJy_to_cgs
    template_fnu_units = (1*u.solLum / u.Hz)
    to_physical = fnu_scl*fnu_units*4*np.pi*dL**2/(1+z)
    to_physical /= (1*template_fnu_units).to(u.erg/u.second/u.Hz)

    # Normed PDF calculation
    pdf = np.exp(-0.5*(chisq_grid - np.nanmin(chisq_grid,axis=0)))
    normed_pdf = pdf/np.nansum(pdf,axis=0)

    # Line fluxes Ha and PaB  - For this to be correct, the line template has to be normalised (integral = 1)
    ha_flux = (ha_nii_ratio/(ha_nii_ratio+1))*(draws_grid[:,:,-2]*fnu_scl)
    pab_flux = draws_grid[:,:,-1]*fnu_scl

    redfunc = eazy.templates.Redden(model=reddening_func_lines, Av=1.)
    klam = -2.5*np.log10(redfunc([w0_ha, w0]))/redfunc.Av*R_v
    # Av from the Ha/PaB decrement
    ext = ha_flux/pab_flux
    ebv = (2.5/(klam[1]-klam[0]))*np.log10(ext/intrinsic_ratio)
    av_lines = R_v*ebv
    # SFR from PaB
    f_pab = pab_flux*u.erg/u.second/u.cm**2  # PaB flux in erg/s/cm**2
    L_pab = f_pab*4*np.pi*(dL**2)            # PaB luminosity in erg/s
    sfr_pab = ha_sfr_conv_factor*L_pab.value*intrinsic_ratio 
    # Dust corrected PaB SFR
    E = 2.5*np.log10(ext/intrinsic_ratio)
    A_PaB = (E/(klam[1]-klam[0]))*klam[1]
    L_pab_corr = L_pab*10**(0.4*A_PaB)
    sfr_pab_corr = np.log10(ha_sfr_conv_factor*L_pab_corr.value*intrinsic_ratio)

    # Calculate stellar mass smoothness prior
    print('Calculating prior')
    coeffs_rest = ((draws_grid.T*to_physical).T)
    coeffs_rest = np.array(coeffs_rest)
    mass_r = coeffs_rest.dot(tab_temp['mass'])
    SFR_r = coeffs_rest.dot(tab_temp['SFR'])
    logsfr = np.log10(SFR_r)
    logmass = np.log10(mass_r)

    so_mass = np.argsort(logmass,axis=0)
    mass_perc, mass_unc = get_percentiles(normed_pdf,logmass,so_mass)  

    im = glob.glob(file_name)[0]
    sci = np.cast[np.float32](pyfits.open(im)['SCI'].data)
    seg = np.cast[np.float32](pyfits.open(seg_im)[0].data)
    mass_image = prep.get_seg_iso_flux(sci, seg, mt, fill=10**(mass_perc[:,1])/mt['npix'])
    first_mass_image = mass_image*1.
    mass_iters = [mass_image]

    for i in range(niter):
        print('Iteration step ',i+1)
        smoothed_mass = nd.median_filter(np.log10(mass_image),5)
        smooth_mass_tab, smooth_mass_err, mass_area = prep.get_seg_iso_flux(10**smoothed_mass, seg, mt)
        full_weight = np.zeros((N_Av*n_draws, self.NOBJ))
        for iobj in range(self.NOBJ):  
            target_mass_i = np.log10(smooth_mass_tab)[iobj]
            masses = logmass[:,iobj]
            prior = norm(loc=target_mass_i, scale=0.1)
            prior_lnp = prior.logpdf(masses)
            full_lnp = np.log(normed_pdf[:,iobj]) + prior_lnp
            full_weight[:,iobj] = np.exp(full_lnp - np.nanmax(full_lnp))
            full_weight[:,iobj] /= np.nansum(full_weight[:,iobj]) 

        full_weight[~np.isfinite(full_weight)] = 0
        normed_pdf2 = full_weight
        
        mass_perc2,mass_unc = get_percentiles(normed_pdf2,logmass,so_mass)
        mass_image = prep.get_seg_iso_flux(sci, seg, mt, fill=10**(mass_perc2[:,1])/mt['npix'])
        mass_iters.append(mass_image)

    # Uncertainties on the line fluxes
    print("Calculating uncertainties on the line fluxes")
    ha_perc, ha_unc = get_percentiles(normed_pdf2,ha_flux)
    pab_perc, pab_unc = get_percentiles(normed_pdf2,pab_flux)

    # Uncertainties on SFR, SM and Av
    print('Calculating uncertainties on physical properties')
    sfr_perc, sfr_unc = get_percentiles(normed_pdf2, logsfr)
    av_perc, av_unc = get_percentiles(normed_pdf2, Av_full_grid_draws)

    # Uncertainties on SFR(PaB) and Av(Ha/PaB)
    sfr_pab_perc, sfr_pab_unc = get_percentiles(normed_pdf2, sfr_pab_corr)
    av_lines_perc, av_lines_unc = get_percentiles(normed_pdf2, av_lines)

    # Get chi2 estimate at maximum posterior distribution
    map_index = np.argmax(normed_pdf2, axis=0)
    coeffs_best = np.array([draws_grid[j,i,:] for i,j in enumerate(map_index)])
    chi2_best = np.array([chisq_grid[j,i] for i,j in enumerate(map_index)])
    model_best = np.array([model_grid[j,:,i] for i,j in enumerate(map_index)])

    # Write output on catalog
    mt['ha_flux'] = ha_perc[:,1]
    mt['pab_flux'] = pab_perc[:,1]
    mt['ha_err'] = ha_unc
    mt['pab_err'] = pab_unc
    mt['chi2'] = chi2_best #/self.NFILT
    mt['logSFR'] = sfr_perc[:,1]
    mt['logstellar_mass'] = mass_perc2[:,1]
    mt['Av_continuum'] = av_perc[:,1]
    mt['logSFR_err'] = sfr_unc
    mt['logstellar_mass_err'] = mass_unc
    mt['Av_continuum_err'] = av_unc
    mt['Av_lines'] = av_lines_perc[:,1]
    mt['Av_lines_err'] = av_lines_unc
    mt['logSFR_pab'] = sfr_pab_perc[:,1]
    mt['logSFR_pab_err'] = sfr_pab_unc
     
    # Create images from table values
    ha_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['ha_flux']/mt['npix'])
    pab_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['pab_flux']/mt['npix'])
    ha_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['ha_err']/mt['npix'])
    pab_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['pab_err']/mt['npix'])
    chi2_im = prep.get_seg_iso_flux(sci, seg, mt, fill=chi2_best) # Nfilt?
    av_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['Av_continuum'])
    av_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['Av_continuum_err'])
    sfr_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logSFR']-np.log10(mt['npix'])) 
    sfr_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logSFR_err']-np.log10(mt['npix']))
    mass_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logstellar_mass']-np.log10(mt['npix']))
    mass_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logstellar_mass_err']-np.log10(mt['npix']))
    av_lines_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['Av_lines'])
    av_lines_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['Av_lines_err'])
    sfr_pab_im = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logSFR_pab']-np.log10(mt['npix']))
    sfr_pab_im_err = prep.get_seg_iso_flux(sci, seg, mt, fill=mt['logSFR_pab_err']-np.log10(mt['npix']))
    
    mt['ha_flux'].unit = u.erg/u.s/u.cm**2
    mt['pab_flux'].unit = u.erg/u.s/u.cm**2
    mt['ha_err'].unit = u.erg/u.s/u.cm**2
    mt['pab_err'].unit = u.erg/u.s/u.cm**2
    mt['logSFR'].unit = u.solMass/u.yr
    mt['logstellar_mass'].unit = u.solMass
    mt['Av_continuum'].unit = u.mag
    mt['logSFR_err'].unit = u.dex
    mt['logstellar_mass_err'].unit = u.dex
    mt['Av_continuum_err'].unit = u.mag
    mt['Av_lines'].unit = u.mag
    mt['Av_lines_err'].unit = u.mag
    mt['logSFR_pab'].unit = u.solMass/u.yr
    mt['logSFR_pab_err'].unit = u.solMass/u.yr

    mt.write(f'{output_path}/{target}_master_table.fits', overwrite=True)

    # Save output
    if save_output:
        print('Saving output')
        im = pyfits.open(glob.glob(file_name)[0])
        save_output_image(im, ha_im, ha_im_err, target, 'ha_flux', output_path, 'H-alpha flux in erg/s/cm2')
        save_output_image(im, pab_im, pab_im_err, target, 'pab_flux', output_path,'Pa-beta flux in erg/s/cm2')
        save_output_image(im, av_im, av_im_err, target, 'av_continuum', output_path, 'Av from stellar continuum')
        save_output_image(im, av_lines_im, av_lines_im_err, target, 'av_lines', output_path, 'Av from H-alpha/Pa-beta')
        save_output_image(im, sfr_im, sfr_im_err, target, 'sfr', output_path, 'SFR from SED-fit')
        save_output_image(im, mass_im, mass_im_err, target, 'mass', output_path, 'Stellar Mass from SED-fit')
        save_output_image(im, sfr_pab_im, sfr_pab_im_err, target, 'sfr_pab', output_path, 'SFR from Pa-beta')
        save_output_image(im, chi2_im, None, target, 'chi2', output_path, 'Chi2 map at maximum posterior distribution')

        primary_extn = pyfits.PrimaryHDU() 
        name_list = ['chi2_grid', 'draws_grid', 'model_grid', 'normed_pdf', 'smoothed_pdf', 'ha_flux',
                      'pab_flux', 'av_continuum', 'av_lines', 'logsfr','logmass','logsfr_pab',
                      'coeffs_best', 'model_best','chi2_best']
        data_list = [chisq_grid, draws_grid, model_grid, normed_pdf,normed_pdf2,ha_flux, pab_flux, 
                      Av_full_grid_draws,av_lines, logsfr, logmass, sfr_pab_corr, 
                      coeffs_best,model_best, chi2_best]
        extn_list = [primary_extn]
        for i,_ in enumerate(name_list):
            extn = pyfits.ImageHDU(data=data_list[i].astype(np.float32),name=f'{_}')
            extn_list.append(extn)
        hdul = pyfits.HDUList(extn_list)
        hdul.writeto(f'{output_path}/{target}_pdf_grids.fits', output_verify='fix', overwrite=True)

    # Plot results
    if plot:
        rgb = rgb_image(target,file_name,blue_f,green_f,red_f)     
        slx = slice(x0-size, x0+size)
        sly = slice(y0-size, y0+size)
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
        ax4.imshow(av_lines_im[sly, slx],vmin=0,vmax=4,cmap='Spectral_r',origin='lower')
        ax4.text(30, 30, r'$A_V$(Ha/PaB)', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad': 8})
        ax4.axis('off')
        ax5.imshow(av_im[sly, slx],vmin=0,vmax=4,cmap='Spectral_r',origin='lower')
        ax5.text(30, 30, r'$A_V$ stars', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad': 8})
        ax5.axis('off')
        ax6.imshow((ha_im/ha_im_err)[sly, slx], vmin=-1,vmax=20,origin='lower')
        ax6.text(30, 30, r'H$\alpha$ S/N', fontsize=15, color='black', bbox={'facecolor':'white', 'alpha':0.9, 'pad':10})
        ax6.axis('off')
        fig.tight_layout(pad=0)
        plt.show()

        SFR_tot = np.nansum(10**mt['logSFR'])
        mass_tot = np.nansum(10**mt['logstellar_mass'])
        print('SFR = %.1f solMass/yr'%(SFR_tot))
        print('Stellar Mass = %.2e solMass'%(mass_tot))
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16,6))
        im1 = ax1.imshow(sfr_im[sly,slx],cmap='Spectral',origin='lower')
        ax1.axis('off')
        cbar1=plt.colorbar(im1, cax = fig.add_axes([0.485, 0.17, 0.01, 0.67]))
        cbar1.set_label(r'log(SFR [$M_{\odot}$/yr])',fontsize=20)
        cbar1.ax.tick_params(labelsize=20)
        im2 = ax2.imshow(mass_im[sly,slx],cmap='inferno',origin='lower')
        ax2.axis('off')
        cbar2=plt.colorbar(im2, cax = fig.add_axes([0.91, 0.17, 0.01, 0.67]))
        cbar2.set_label(r'log(Stellar Mass [$M_{\odot}$])',fontsize=20)
        cbar2.ax.tick_params(labelsize=20)
        plt.show()

    return mt

  #----------------------------------------------------------------------------
