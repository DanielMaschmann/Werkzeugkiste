"""
This script gathers classes with methods that will be used in multiple applications
"""

import os
from mailbox import FormatError
from pathlib import Path, PosixPath
import warnings
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.io import ascii, fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astropy import constants as const
speed_of_light_kmps = const.c.to('km/s').value
from scipy.constants import c as speed_of_light_mps
from scipy.spatial import ConvexHull
from scipy import odr
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from pandas import read_csv
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from reproject import reproject_interp
# own packages
from werkzeugkiste import phys_params


class CoordTools:
    """
    Class to gather helper functions for coordinates and distances
    """

    @staticmethod
    def calc_coord_separation(ra_1, dec_1, ra_2, dec_2):
        pos_1 = SkyCoord(ra=ra_1*u.deg, dec=dec_1*u.deg)
        pos_2 = SkyCoord(ra=ra_2*u.deg, dec=dec_2*u.deg)

        return pos_1.separation(pos_2)

    @staticmethod
    def arcsec2kpc(diameter_arcsec, target_dist_mpc):
        """
        convert a length of arcsecond into a length in kpc by taking the distance of the object into account

        Parameters
        ----------
        diameter_arcsec : float or array-like
            diameter in arcseconds
        target_dist_mpc : float
            target distance in Mpc

        Returns
        -------
        diameter_kpc : float or array-like
        """
        # convert arcseconds into radian
        diameter_radian = diameter_arcsec / 3600 * np.pi / 180
        # return target_dist_mpc * diameter_radian * 1000
        return 2 * target_dist_mpc * np.tan(diameter_radian/2) * 1000

    @staticmethod
    def kpc2arcsec(diameter_kpc, target_dist_mpc):
        """
        convert a length of arcsecond into a length in kpc by taking the distance of the object into account

        Parameters
        ----------
        diameter_kpc : float or array-like
            diameter in kpc
        target_dist_mpc : float
            target distance in Mpc

        Returns
        -------
        diameter_arcsec : float or array-like
        """

        kpc_per_arcsec = CoordTools.arcsec2kpc(diameter_arcsec=1, target_dist_mpc=target_dist_mpc)
        return diameter_kpc / kpc_per_arcsec

    @staticmethod
    def get_target_central_simbad_coords(target_name, target_dist_mpc=None):
        """
        Function to find central target coordinates from SIMBAD with astroquery
        Parameters
        ----------

        Returns
        -------
        central_target_coords : ``astropy.coordinates.SkyCoord``
        """
        # get the center of the target
        simbad_table = Simbad.query_object(target_name)

        if target_dist_mpc is None:
            return SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                            unit=(u.hourangle, u.deg))
        else:
            return SkyCoord('%s %s' % (simbad_table['RA'].value[0], simbad_table['DEC'].value[0]),
                            unit=(u.hourangle, u.deg), distance=target_dist_mpc * u.Mpc)

    @staticmethod
    def construct_wcs(ra_min, ra_max, dec_min, dec_max, img_shape, quadratic_image=True, ctype=None):
        """Function to generate a WCS from scratch by only using a box of coordinates and pixel sizes.
        Parameters
        ----------
        ra_min, ra_max, dec_min, dec_max,  : float
            outer coordinates of the new frame.
        img_shape : tuple
            number of pixels
        quadratic_image : bool
            flag whether the resulting WCS is quadratic or not

        Returns
        -------
        wcs : astropy.wcs.WCS()
            new WCS system centered on the coordinates
        """
        # get length of image
        pos_coord_lower_left = SkyCoord(ra=ra_min * u.deg, dec=dec_min * u.deg)
        pos_coord_lower_right = SkyCoord(ra=ra_max * u.deg, dec=dec_min * u.deg)
        pos_coord_upper_left = SkyCoord(ra=ra_min * u.deg, dec=dec_max * u.deg)
        # now get the size of the image
        ra_width = (pos_coord_lower_left.separation(pos_coord_lower_right)).degree
        dec_width = (pos_coord_lower_left.separation(pos_coord_upper_left)).degree

        # if we want to have a quadratic image we use the largest width
        if quadratic_image:
            ra_image_width = np.max([ra_width, dec_width])
            dec_image_width = np.max([ra_width, dec_width])
        else:
            ra_image_width = ra_width
            dec_image_width = dec_width

        # get central coordinates
        ra_center = (ra_min + ra_max) / 2
        dec_center = (dec_min + dec_max) / 2

        # now create a WCS for this histogram
        new_wcs = WCS(naxis=2)
        # what is the center pixel of the XY grid.
        new_wcs.wcs.crpix = [img_shape[1] / 2, img_shape[0] / 2]
        # what is the galactic coordinate of that pixel.
        new_wcs.wcs.crval = [ra_center, dec_center]
        # what is the pixel scale in lon, lat.
        new_wcs.wcs.cdelt = np.array([-ra_image_width / img_shape[1], dec_image_width / img_shape[0]])
        # you would have to determine if this is in fact a tangential projection.
        if ctype is None:
            new_wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        else:
            new_wcs.wcs.ctype = ctype
        return new_wcs

    @staticmethod
    def reproject_image(data, wcs, new_wcs, new_shape):
        """function to reproject an image with na existing WCS to a new WCS
        Parameters
        ----------
        data : ndarray
        wcs : astropy.wcs.WCS()
        new_wcs : astropy.wcs.WCS()
        new_shape : tuple

        Returns
        -------
        new_data : ndarray
            new data reprojected to the new wcs
        """
        hdu = fits.PrimaryHDU(data=data, header=wcs.to_header())
        return reproject_interp(hdu, new_wcs, shape_out=new_shape, return_footprint=False)

    @staticmethod
    def get_img_cutout(img, wcs, coord, cutout_size):
        """function to cut out a region of a larger image with an WCS.
        Parameters
        ----------
        img : ndarray
            (Ny, Nx) image
        wcs : astropy.wcs.WCS()
            astropy world coordinate system object describing the parameter image
        coord : astropy.coordinates.SkyCoord
            astropy coordinate object to point to the selected area which to cutout
        cutout_size : float or tuple
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.

        Returns
        -------
        cutout : astropy.nddata.Cutout2D object
            cutout object of the initial image
        """
        if isinstance(cutout_size, tuple):
            size = cutout_size * u.arcsec
        elif isinstance(cutout_size, float) | isinstance(cutout_size, int):
            size = (cutout_size, cutout_size) * u.arcsec
        else:
            raise KeyError('cutout_size must be float or tuple')

        # check if cutout is inside the image
        pix_pos = wcs.world_to_pixel(coord)
        if (pix_pos[0] > 0) & (pix_pos[0] < img.shape[1]) & (pix_pos[1] > 0) & (pix_pos[1] < img.shape[0]):
            if np.ndim(img) == 2:
                return Cutout2D(data=img, position=coord, size=size, wcs=wcs)
            elif np.ndim(img) == 3:
                img_data1 = Cutout2D(data=img[:, :, 0], position=coord, size=size, wcs=wcs)
                img_data2 = Cutout2D(data=img[:, :, 1], position=coord, size=size, wcs=wcs)
                img_data3 = Cutout2D(data=img[:, :, 2], position=coord, size=size, wcs=wcs)
                cutout_data = np.zeros((*img_data1.shape, 3))
                cutout_data[:,:, 0] = img_data1.data
                cutout_data[:,:, 1] = img_data2.data
                cutout_data[:,:, 2] = img_data3.data
                cut_out = type('', (), {})()
                cut_out.data = cutout_data
                cut_out.wcs = img_data1.wcs
                return cut_out
            else:
                raise FormatError('The cutout must have 2 or 3 dimensions. The latter would be a RGB image')
        else:
            warnings.warn("The selected cutout is outside the original dataset. The data and WCS will be None",
                          DeprecationWarning)
            cut_out = type('', (), {})()
            cut_out.data = None
            cut_out.wcs = None
            return cut_out

    @staticmethod
    def transform_world2pix_scale(length_in_arcsec, wcs, dim=0):
        """ Function to get the pixel length of a length in arcseconds
        Parameters
        ----------
        length_in_arcsec : float
            length
        wcs : ``astropy.wcs.WCS``
            astropy world coordinate system object describing the parameter image
        dim : int, 0 or 1
            specifys the dimension 0 for ra and 1 for dec. This should be however always the same values...

        Returns
        -------
        length_in_pixel : float
            length in pixel along the axis
        """

        return (length_in_arcsec * u.arcsec).to(u.deg) / wcs.proj_plane_pixel_scales()[dim]

    @staticmethod
    def transform_pix2world_scale(length_in_pix, wcs, dim=0):
        """ Function to get the pixel length of a length in arcseconds
        Parameters
        ----------
        length_in_pix : float
            length
        wcs : ``astropy.wcs.WCS``
            astropy world coordinate system object describing the parameter image
        dim : int, 0 or 1
            specifys the dimension 0 for ra and 1 for dec. This should be however always the same values...

        Returns
        -------
        length_in_pixel : float
            length in arcsec along the axis
        """

        return length_in_pix * wcs.proj_plane_pixel_scales()[dim].to(u.arcsec).value

    @staticmethod
    def mask_2d_region_in_cube(cube, wcs_2d, ra, dec, cutoutsize):
        print('you lazy fucker should code this!')

    @staticmethod
    def find_cross_match(ra_obj1, dec_obj1, ra_obj2, dec_obj2, cross_match_rad_arcsec, nth_neighbor=1):
        coords_obj1 = SkyCoord(ra=ra_obj1*u.deg, dec=dec_obj1*u.deg)
        coords_obj2 = SkyCoord(ra=ra_obj2*u.deg, dec=dec_obj2*u.deg)

        print(coords_obj1)
        print(coords_obj2)


        cross_match_idx_obj1_with_obj2, cross_match_d2d_obj1_with_obj2, _ = \
            coords_obj2.match_to_catalog_sky(coords_obj1, nthneighbor=nth_neighbor)

        mask_ob_obj1_with_obj2 = cross_match_d2d_obj1_with_obj2 < cross_match_rad_arcsec * u.arcsec

        return mask_ob_obj1_with_obj2, cross_match_d2d_obj1_with_obj2, cross_match_idx_obj1_with_obj2


class UnitTools:
    """
    Class to gather all tools for unit conversions
    """

    @staticmethod
    def get_flux_unit_conv_fact(old_unit, new_unit, pixel_size=None, band_wave=None):
        assert old_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']
        assert new_unit in ['mJy', 'Jy', 'MJy/sr', 'erg A-1 cm-2 s-1']

        conversion_factor = 1
        if old_unit != new_unit:
            # now first change the conversion factor to Jy
            if old_unit == 'mJy':
                conversion_factor *= 1e-3
            elif old_unit == 'MJy/sr':
                conversion_factor *= (1e6 * pixel_size)
            elif old_unit == 'erg A-1 cm-2 s-1':
                # The conversion from erg A-1 cm-2 s-1 is well described in
                # https://www.physicsforums.com/threads/unit-conversion-flux-densities.742561/
                # se also
                # https://www.physicsforums.com/threads/unit-conversion-of-flux-jansky-to-erg-s-cm-a-simplified-guide.927166/
                # we use fv dv = fλ dλ
                # fλ = fv dv/dλ
                # and because v = c/λ...
                # fλ = fv*c / λ^2
                # thus the conversion factor is:
                conversion_factor = 1e23 * 1e-8 * (band_wave ** 2) / (speed_of_light_mps * 1e2)
                # the speed of light is in m/s the factor 1-e2 changes it to cm/s
                # the factor 1e8 changes Angstrom to cm (the Angstrom was in the nominator therefore it is 1/1e-8)

            # now convert to new unit
            if new_unit == 'mJy':
                conversion_factor *= 1e3
            elif new_unit == 'MJy/sr':
                conversion_factor *= 1e-6 / pixel_size
            elif new_unit == 'erg A-1 cm-2 s-1':
                conversion_factor *= 1e-23 * 1e8 * (speed_of_light_mps * 1e2) / (band_wave ** 2)

        return conversion_factor

    @staticmethod
    def get_hst_img_conv_fct(img_header, img_wcs, flux_unit='Jy', no_header_conv=False):
        """
        get unit conversion factor to go from electron counts to mJy of HST images
        Parameters
        ----------
        img_header : ``astropy.io.fits.header.Header``
        img_wcs : ``astropy.wcs.WCS``
        flux_unit : str
        no_header_conv : bool
            this keyword is to flagg hst products which have no header information and are provided in Jy
        Returns
        -------
        conversion_factor : float

        """
        # convert the flux unit
        if not no_header_conv:
            if 'PHOTFNU' in img_header:
                conversion_factor = img_header['PHOTFNU']
            elif 'PHOTFLAM' in img_header:
                # wavelength in angstrom
                pivot_wavelength = img_header['PHOTPLAM']
                # inverse sensitivity, ergs/cm2/Ang/electron
                sensitivity = img_header['PHOTFLAM']
                # speed of light in Angstrom/s
                c = speed_of_light_mps * 1e10
                # change the conversion facto to get erg s−1 cm−2 Hz−1
                f_nu = sensitivity * pivot_wavelength ** 2 / c
                # change to get Jy
                conversion_factor = f_nu * 1e23
            else:
                raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')
        else:
            conversion_factor = 1

        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = conversion_factor
        elif flux_unit == 'mJy':
            # rescale to mJy
            conversion_factor *= 1e3
        elif flux_unit == 'MJy/sr':
            # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later
            # change to MJy/sr
            conversion_factor /= (pixel_area_size_sr * 1e6)
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand!')

        return conversion_factor

    @staticmethod
    def get_hst_ir_img_conv_fct(band, img_header, img_wcs, flux_unit='Jy'):
        """
        get unit conversion factor to go from electron counts to mJy of HST images
        Parameters
        ----------
        band: str
        img_header : ``astropy.io.fits.header.Header``
        img_wcs : ``astropy.wcs.WCS``
        flux_unit : str
        no_header_conv : bool
            this keyword is to flagg hst products which have no header information and are provided in Jy
        Returns
        -------
        conversion_factor : float

        """
        # convert the flux unit

        os.environ['PYSYN_CDBS'] = ('/home/benutzer/software/python_packages/phangs_data_access/meta_data/'
                                    'wfc3_calibration_files/'
                                    'hlsp_reference-atlases_hst_multi_everything_multi_v11_sed/grp/redcat/trds/')

        import stsynphot as stsyn

        vega_url = 'https://ssb.stsci.edu/trds/calspec/alpha_lyr_stis_010.fits'
        stsyn.Vega = stsyn.spectrum.SourceSpectrum.from_file(vega_url)

        mjd = img_header['ROUTTIME']

        # aper = '0.385'
        aper = '6.0'

        # obsmode = f'wfc3, {detector}, {filt}, mjd#{mjd}, aper#{aper}'
        obsmode = f'wfc3, ir, {band.lower()}, mjd#{mjd}, aper#{aper}'

        bp = stsyn.band(obsmode)

        photflam = bp.unit_response(stsyn.conf.area)  # inverse sensitivity in flam

        photplam = bp.pivot() # pivot wavelength in angstroms

        # wavelength in angstrom
        pivot_wavelength = photplam.value
        # inverse sensitivity, ergs/cm2/Ang/electron
        sensitivity = photflam.value
        # speed of light in Angstrom/s
        c = speed_of_light_mps * 1e10
        # change the conversion facto to get erg s−1 cm−2 Hz−1
        f_nu = sensitivity * pivot_wavelength ** 2 / c
        # change to get Jy
        conversion_factor = f_nu * 1e23

        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = conversion_factor
        elif flux_unit == 'mJy':
            # rescale to mJy
            conversion_factor *= 1e3
        elif flux_unit == 'MJy/sr':
            # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later
            # change to MJy/sr
            conversion_factor /= (pixel_area_size_sr * 1e6)
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand!')

        return conversion_factor

    @staticmethod
    def get_jwst_conv_fact(img_wcs, flux_unit='Jy'):
        """
        get unit conversion factor for JWST image observations
        ----------
        img_wcs : ``astropy.wcs.WCS``
        flux_unit : str

        Returns
        -------
        conversion_factor : float

        """
        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e6

        elif flux_unit == 'mJy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e9
        elif flux_unit == 'MJy/sr':
            conversion_factor = 1
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')
        return conversion_factor

    @staticmethod
    def get_astrosat_conv_fact(img_wcs, wave_angstrom, flux_unit='Jy'):
        """
        get unit conversion factor for ASTROSAT image observations
        ----------
        img_wcs : ``astropy.wcs.WCS``
        flux_unit : str

        Returns
        -------
        conversion_factor : float

        """
        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * phys_params.sr_per_square_deg

        # rescale data image
        if flux_unit == 'erg A-1 cm-2 s-1':
            conversion_factor = 1
        elif flux_unit == 'Jy':
            conversion_factor = 1e23 * 1e-2 * 1e-8 * (wave_angstrom ** 2) / speed_of_light_mps
        elif flux_unit == 'mJy':
            conversion_factor = 1e3 * 1e23 * 1e-2 * 1e-8 * (wave_angstrom ** 2) / speed_of_light_mps
        elif flux_unit == 'MJy/sr':
            conversion_factor = (1e-6 * 1e23 * 1e-2 * 1e-8 * (wave_angstrom ** 2) /
                                 (speed_of_light_mps * pixel_area_size_sr))
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')
        return conversion_factor

    @staticmethod
    def conv_mag2abs_mag(mag, dist):
        """
        conversion following https://en.wikipedia.org/wiki/Absolute_magnitude
        M = m - 5*log10(d_pc) + 5
        M = m - 5*log10(d_Mpc * 10^6) + 5
        M = m - 5*log10(d_Mpc) -5*log10(10^6) + 5
        M = m - 5*log10(d_Mpc) -25

        Parameters
        ----------
        mag : float or array-like
            magnitude
        dist : float or array-like
            distance in Mpc

        Returns
        -------
        float or array
            the absolute magnitude

         """
        return mag - 25 - 5 * np.log10(dist)

    @staticmethod
    def conv_abs_mag2mag(abs_mag, dist):
        """
        conversion following https://en.wikipedia.org/wiki/Absolute_magnitude
        M = m - 5*log10(d_pc) + 5
        M = m - 5*log10(d_Mpc * 10^6) + 5
        M = m - 5*log10(d_Mpc) -5*log10(10^6) + 5
        M = m - 5*log10(d_Mpc) -25

        Parameters
        ----------
        abs_mag : float or array-like
            absolute magnitude
        dist : float or array-like
            distance in Mpc

        Returns
        -------
        float or array
            the observed magnitude

         """
        return abs_mag + 25 + 5 * np.log10(dist)

    @staticmethod
    def angstrom2unit(wave, unit='mu'):
        """
        Returns wavelength at needed wavelength
        Parameters
        ----------
        wave : float
        unit : str

        Returns
        -------
        wavelength : float
        """
        if unit == 'angstrom':
            return wave
        if unit == 'nano':
            return wave * 1e-1
        elif unit == 'mu':
            return wave * 1e-4
        else:
            raise KeyError('return unit not understand')

    @staticmethod
    def nanometers2unit(wave, unit='mu'):
        """
        Returns wavelength at needed wavelength
        Parameters
        ----------
        wave : float
        unit : str

        Returns
        -------
        wavelength : float
        """
        if unit == 'angstrom':
            return wave * 1e1
        if unit == 'nano':
            return wave
        elif unit == 'mu':
            return wave * 1e-3
        else:
            raise KeyError('return unit not understand')

    @staticmethod
    def conv_mjy2ab_mag(flux):
        """
        conversion of mJy to AB mag.
        See definition on Wikipedia : https://en.wikipedia.org/wiki/AB_magnitude
        Parameters
        ----------
        flux : float or ``np.ndarray``
        """

        return -2.5 * np.log10(flux * 1e-3) + 8.90

    @staticmethod
    def conv_ab_mag2mjy(mag):
        """
        conversion of AB mag to mJy.
        See definition on Wikipedia : https://en.wikipedia.org/wiki/AB_magnitude
        Parameters
        ----------
        mag : float or ``np.ndarray``
        """

        return 1e3 * 10 ** ((8.5 - mag) / 2.5)

    @staticmethod
    def get_hst_vega_zp(instrument, band):
        """
        Function to get HST Vega zero point flux in Jy
        Parameters
        ----------
        instrument : str
        band : str
        Return
        ------
        zp_vega_flux : float
            Zero-point flux in Jy
        """
        if instrument == 'acs':
            return phys_params.hst_acs_wfc1_bands_wave[band]["zp_vega"]
        elif instrument in ['uvis', 'uvis1']:
            return phys_params.hst_wfc3_uvis1_bands_wave[band]["zp_vega"]
        elif instrument == 'uvis2':
            return phys_params.hst_wfc3_uvis2_bands_wave[band]["zp_vega"]
        else:
            raise KeyError('instrument must be acs, uvis, uvis1 or uvis2')

    @staticmethod
    def get_jwst_vega_zp(instrument, band):
        """
        Function to get JWST Vega zero point flux in Jy
        Parameters
        ----------
        instrument : str
        band : str
        Return
        ------
        zp_vega_flux : float
            Zero-point flux in Jy
        """
        if instrument == 'nircam':
            return phys_params.nircam_bands_wave[band]["zp_vega"]
        elif instrument == 'miri':
            return phys_params.miri_bands_wave[band]["zp_vega"]
        else:
            raise KeyError('instrument must be nircam or miri')

    @staticmethod
    def get_astrosat_vega_zp(band):
        """
        Function to get ASTROSAT Vega zero point flux in Jy
        Parameters
        ----------
        band : str
        Return
        ------
        zp_vega_flux : float
            Zero-point flux in Jy
        """
        return phys_params.astrosat_bands_wave[band]["zp_vega"]

    @staticmethod
    def get_roman_vega_zp(band):
        """
        Function to get ROMAN Vega zero point flux in Jy
        Parameters
        ----------
        band : str
        Return
        ------
        zp_vega_flux : float
            Zero-point flux in Jy
        """
        return phys_params.roman_bands_wave[band]["zp_vega"]

    @staticmethod
    def conv_mjy2vega(flux, telescope, instrument, band):
        """
        This function converts
        """
        if telescope == 'hst':
            zp_vega_flux = UnitTools.get_hst_vega_zp(instrument=instrument, band=band)
        elif telescope == 'jwst':
            zp_vega_flux = UnitTools.get_jwst_vega_zp(instrument=instrument, band=band)
        elif telescope == 'astrosat':
            zp_vega_flux = UnitTools.get_astrosat_vega_zp(band=band)
        elif telescope == 'roman':
            zp_vega_flux = UnitTools.get_roman_vega_zp(band=band)
        else:
            raise KeyError('telescope musst be hst, jwst or astrosat')

        # here we must be careful because the flux of the filter is in mJy and the zero point flux is in Jy
        return -2.5 * np.log10(flux*1e-3 / zp_vega_flux)

    @staticmethod
    def conv_mjy_err2vega_err(flux, flux_err):


        return 2.5 * flux_err / (flux * np.log(10))

    @staticmethod
    def conv_vega2mjy(vega_mag, telescope, instrument, band):
        """
        This function converts
        """
        if telescope == 'hst':
            zp_vega_flux = UnitTools.get_hst_vega_zp(instrument=instrument, band=band)
        elif telescope == 'jwst':
            zp_vega_flux = UnitTools.get_jwst_vega_zp(instrument=instrument, band=band)
        elif telescope == 'astrosat':
            zp_vega_flux = UnitTools.get_astrosat_vega_zp(band=band)
        else:
            raise KeyError('telescope musst be hst, jwst or astrosat')

        # here we must be careful because the flux of the filter is in mJy and the zero point flux is in Jy


        return (10 ** (vega_mag/(-2.5))) * zp_vega_flux * 1e3

    @staticmethod
    def conv_flux2lum(flux, dist_mpc):
        dist_m = (dist_mpc * u.Mpc).to(u.cm).value
        return flux * (4 * np.pi) * (dist_m ** 2)

    @staticmethod
    def conv_flux2lum_uncertainty(flux_err, dist_mpc):
        dist_cm = (dist_mpc * u.Mpc).to(u.cm).value

        return flux_err * (4 * np.pi) * (dist_cm ** 2)


class TransTools:
    """
    Tools to transform quantities
    """
    @staticmethod
    def gauss_sig2fwhm(sig):
        """
        see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        """
        return sig*(2 * np.sqrt(2*np.log(2)))

    @staticmethod
    def gauss_fwhm2sig(fwhm):
        """
        see https://en.wikipedia.org/wiki/Full_width_at_half_maximum
        """
        return fwhm/(2 * np.sqrt(2*np.log(2)))

    @staticmethod
    def gauss_integral1d(amp, sig):
        return amp * sig * np.sqrt(2 * np.pi)

    @staticmethod
    def gauss_integral2d(amp, sig):
        return amp * 2 * np.pi * (sig ** 2)


class FileTools:
    """
    Tool to organize data paths, file names and local structures and str compositions
    """

    @staticmethod
    def target_name_no_directions(target):
        """
        removes letters at the end of the target name.

        Parameters
        ----------
        target :  str

        Returns
        -------
        target_name : str
        """
        if target[-1] in ['n', 's', 'w', 'e', 'c', 'N', 'S', 'W', 'E', 'C']:
            return target[:-1]
        else:
            return target

    @staticmethod
    def target_names_no_zeros(target):
        """
        removes zeros from target name.

        Parameters
        ----------
        target :  str

        Returns
        -------
        target_name : str
        """
        if (target[0:3] == 'ngc') & (target[3] == '0'):
            return target[0:3] + target[4:]
        else:
            return target

    @staticmethod
    def target_names_no_zeros_no_directions(target):
        """
        removes zeros from target name.

        Parameters
        ----------
        target :  str

        Returns
        -------
        target_name : str
        """
        return FileTools.target_name_no_directions(target=FileTools.target_names_no_zeros(target=target))

    @staticmethod
    def get_sample_table_target_name(target):
        """
        get the corresponding target name to access data in the phangs sample table

        Parameters
        ----------
        target :  str

        Returns
        -------
        target_name : str
        """
        # the target name needs to have no directional letters
        target = FileTools.target_name_no_directions(target=target)
        # NGC1510 is not in the PHANGS sample but is the minor companion of NGC1512
        if target == 'ngc1510':
            target = 'ngc1512'
        return target

    @staticmethod
    def download_file(file_path, url, unpack=False, reload=False):
        """

        Parameters
        ----------
        file_path : str or ``pathlib.Path``
        url : str
        unpack : bool
            In case the downloaded file is zipped, this function can unpack it and remove the downloaded file,
            leaving only the extracted file
        reload : bool
            If the file is corrupted, this removes the file and reloads it

        Returns
        -------

        """
        if reload:
            # if reload file the file will be removed to re download it
            os.remove(file_path)
        # check if file already exists
        if os.path.isfile(file_path):
            print(file_path, 'already exists')
            return True
        else:
            from urllib3 import PoolManager
            # download file
            http = PoolManager()
            r = http.request('GET', url, preload_content=False)

            if unpack:
                with open(file_path.with_suffix(".gz"), 'wb') as out:
                    while True:
                        data = r.read()
                        if not data:
                            break
                        out.write(data)
                r.release_conn()
                # uncompress file
                from gzip import GzipFile
                # read compressed file
                compressed_file = GzipFile(file_path.with_suffix(".gz"), 'rb')
                s = compressed_file.read()
                compressed_file.close()
                # save compressed file
                uncompressed_file = open(file_path, 'wb')
                uncompressed_file.write(s)
                uncompressed_file.close()
                # delete compressed file
                os.remove(file_path.with_suffix(".gz"))
            else:
                with open(file_path, 'wb') as out:
                    while True:
                        data = r.read()
                        if not data:
                            break
                        out.write(data)
                r.release_conn()

    @staticmethod
    def identify_file_in_folder(folder_path, str_in_file_name_1, str_in_file_name_2=None):
        """
        Identify a file inside a folder that contains a specific string.

        Parameters
        ----------
        folder_path : Path or str
        str_in_file_name_1 : str
        str_in_file_name_2 : str

        Returns
        -------
        file_name : Path
        """

        if str_in_file_name_2 is None:
            str_in_file_name_2 = str_in_file_name_1

        if isinstance(folder_path, str):
            folder_path = Path(folder_path)
        identified_files_1 = list(filter(lambda x: str_in_file_name_1 in x, os.listdir(folder_path)))

        identified_files_2 = list(filter(lambda x: str_in_file_name_2 in x, os.listdir(folder_path)))

        if not identified_files_1 and not identified_files_2:
            raise FileNotFoundError('The data file containing the string %s or %s does not exist.' %
                                    (str_in_file_name_1, str_in_file_name_2))
        elif len(identified_files_1) > 1:
            raise FileExistsError('There are more than one data files containing the string %s .' % str_in_file_name_1)
        elif len(identified_files_2) > 1:
            raise FileExistsError('There are more than one data files containing the string %s .' % str_in_file_name_2)
        else:
            if not identified_files_2:
                return folder_path / str(identified_files_1[0])
            if not identified_files_1:
                return folder_path / str(identified_files_2[0])
            if identified_files_1 and identified_files_2:
                return folder_path / str(identified_files_1[0])

    @staticmethod
    def load_img(file_name, hdu_number=0):
        """function to open hdu using astropy.

        Parameters
        ----------
        file_name : str or Path
            file name to open
        hdu_number : int or str
            hdu number which should be opened. can be also a string such as 'SCI' for JWST images

        Returns
        -------
        array-like,  ``astropy.io.fits.header.Header`` and ``astropy.wcs.WCS` and
        """
        # get hdu
        hdu = fits.open(file_name)
        # get header
        header = hdu[hdu_number].header
        # get WCS
        wcs = WCS(header)
        # update the header
        header.update(wcs.to_header())
        # reload the WCS and header
        header = hdu[hdu_number].header
        wcs = WCS(header)
        # load data
        data = hdu[hdu_number].data
        # close hdu again
        hdu.close()
        return data, header, wcs

    @staticmethod
    def load_alma_cube(file_name, hdu_number=0):
        """function to open hdu using astropy.

        Parameters
        ----------
        file_name : str or Path
            file name to open
        hdu_number : int or str
            hdu number which should be opened. can be also a string such as 'SCI' for JWST images

        Returns
        -------
        array-like,  ``astropy.io.fits.header.Header`` and ``astropy.wcs.WCS` and
        """
        # get hdu
        hdu = fits.open(file_name)
        # load the header
        header = hdu[hdu_number].header
        # get WCS in 3d and celestial
        wcs3d = WCS(header)
        wcs2d = wcs3d.celestial
        # load data
        data_cube = hdu[hdu_number].data
        # now get the 3rd axis of the cube
        x_axis_values = header['CRVAL3'] + np.arange(header['NAXIS3']) * header['CDELT3']
        # close hdu again
        hdu.close()
        return data_cube, x_axis_values, header, wcs3d, wcs2d



    @staticmethod
    def load_fits_table(file_name, hdu_number=0):
        """function to open hdu using astropy.

        Parameters
        ----------
        file_name : str or Path
            file name to open
        hdu_number : int or str
            hdu number which should be opened. can be also a string such as 'SCI' for JWST images

        Returns
        -------
        array-like and  ``astropy.io.fits.header.Header``
        """
        # get hdu
        hdu = fits.open(file_name)
        # get header
        header = hdu[hdu_number].header
        # load data
        data = hdu[hdu_number].data
        # close hdu again
        hdu.close()
        return data, header

    @staticmethod
    def load_ascii_table(file_name):
        """
        function to load ascii table with csv suffix using astropy.

        Parameters
        ----------
        file_name : str or Path
            file name to open
        Returns
        -------
        acsii_table : `astropy.io.ascii.BaseReader`
        """
        return ascii.read(file_name, format='csv')

    @staticmethod
    def load_ascii_table_from_txt(file_name):
        """
        function to open table from txt file with `#` as column name indicator

        Parameters
        ----------
        file_name : str or Path

        """
        ascii_tab = read_csv(file_name, delim_whitespace=True)
        ascii_tab.columns = ascii_tab.columns.str.replace('#', '')
        return ascii_tab

    @staticmethod
    def verify_suffix(file_name, suffix, change_suffix=False):
        """
        check if the wanted suffix is in place and if not, add it.
        If it is the wrong suffix, there is the possibility to change it
        Parameters
        ----------
        file_name : str or ``pathlib.Path``
        suffix : str
        change_suffix : bool

        Return
        ------
        file_name : ``pathlib.Path``
        """
        assert type(file_name) in [str, PosixPath]
        # make sure file_path is of pathlib type
        if isinstance(file_name, str):
            file_name = Path(file_name)
        # make sure there is a dot in front of the suffix
        if suffix[0] != '.':
            suffix = '.' + suffix
        # add suffix is needed
        if file_name.suffix == '':
            file_name = file_name.with_suffix(suffix)
        # change suffix if needed
        if change_suffix & (file_name.suffix != suffix):
            file_name = file_name.with_suffix(suffix)
        return file_name


class GeometryTools:
    """
    all functions related to compute hulls or check if objects are inside hulls or polygons
    """

    @staticmethod
    def contour2hull(data_array, level=0, contour_index=0, n_max_rejection_vertice=1000):
        """
        This function will compute the hull points of one contour line.
        It is important to notice that the contours can be patchy and therefore,
        this function will return multiple hulls.
        In order to avoid selecting smaller hulls around some outliers the keyword `n_max_rejection_vertice` limits
        the number of points that need to be in a hull. This can or course strongly vary from dataset to dataset and
        should not be used unsupervised.

        Parameters
        ----------
        data_array : ``numpy.ndarray``
        level : float
        contour_index : int
        n_max_rejection_vertice : int

        Returns
        -------
        hull_dict : dict
        """
        # estimate background
        # create a dummy figure and axis
        from matplotlib.pyplot import subplots, close
        dummy_fig, dummy_ax = subplots()
        # compute contours
        contours = dummy_ax.contour(data_array, levels=level, colors='red')
        # get the path collection of one specific contour level
        # print(contours.allsegs)
        # print(contours.allsegs[0])
        # print(contours.allsegs[0][0])
        # exit()
        contour_collection = contours.allsegs[contour_index]
        # get rid of the dummy figure
        close(dummy_fig)
        # loop over the contours and select valid paths
        hull_dict = {}
        for idx, contour in enumerate(contour_collection):
            if len(contour) > n_max_rejection_vertice:
                # get all points from contour
                x_cont = []
                y_cont = []
                for point in contour:
                    x_cont.append(point[0])
                    y_cont.append(point[1])
                x_cont = np.array(x_cont)
                y_cont = np.array(y_cont)

                # make the contour a closed loop (add the first point to the end of the array)
                x_convex_hull = np.concatenate([x_cont, np.array([x_cont[0]])])
                y_convex_hull = np.concatenate([y_cont, np.array([y_cont[0]])])

                hull_dict.update({idx: {
                    'x_convex_hull': x_convex_hull,
                    'y_convex_hull': y_convex_hull
                }})

        return hull_dict

    @staticmethod
    def check_points_in_2d_convex_hull(x_point, y_point, x_data_hull, y_data_hull, tol=1e-12):
        """
        Function to provide feedback whether a point lies inside a convex hull or not

        """
        hull = ConvexHull(np.array([x_data_hull, y_data_hull]).T)
        p = np.array([x_point, y_point]).T
        return np.all(hull.equations[:, :-1] @ p.T + np.repeat(hull.equations[:, -1][None, :], len(p), axis=0).T <= tol,
                      0)

    @staticmethod
    def check_points_in_polygon(x_point, y_point, x_data_hull, y_data_hull):
        """
        Function to check if a point is inside a polygon or not.
        This is not very fast however there have been many more attempts listed here:
        https://stackoverflow.com/questions/36399381/
        whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
        which are more computational optimized.

        Parameters
        ----------

        x_point : array-like or list
        y_point : array-like or list
        x_data_hull : array-like or list
        y_data_hull : array-like or list
        """
        polygon = Polygon(np.array([x_data_hull, y_data_hull]).T)

        mask_covere_points = np.zeros(len(x_point), dtype=bool)
        for idx in range(len(x_point)):
            point = Point(x_point[idx], y_point[idx])
            mask_covere_points[idx] = polygon.contains(point)
        return mask_covere_points

    @staticmethod
    def flag_close_points2ensemble(x_data, y_data, x_data_ensemble, y_data_ensemble, max_dist2ensemble):
        """
        This function flags all data points which are too close to the points of an ensemble.
        The ensemble can be for example a hull, borders of observations etc

        Parameters
        ----------

        x_data : array-like
        y_data : array-like
        x_data_ensemble : array-like
        y_data_ensemble : array-like
        max_dist2ensemble : float
        """

        min_dist2point_ensemble = np.zeros(len(x_data))
        for index in range(len(min_dist2point_ensemble)):
            # now get the distance to the arms
            dist2point_ensemble = np.sqrt((x_data_ensemble - x_data[index]) ** 2 + (y_data_ensemble - y_data[index]) ** 2)
            min_dist2point_ensemble[index] = min(dist2point_ensemble)
        return min_dist2point_ensemble > max_dist2ensemble

    @staticmethod
    def select_img_pix_along_line(data, x_pos, y_pos, angle):


        x_pixels = np.arange(data.shape[1])
        y_pixels = np.arange(data.shape[0])
        x_mesh, y_mesh = np.meshgrid(x_pixels, y_pixels)

        slope = np.tan(angle*np.pi/180)
        intercept = y_pos - slope * x_pos
         # check if slope is 0 or 90 degree
        if angle == 0:
            return (y_mesh == np.rint(y_pos - 0.5)) | (y_mesh == np.rint(y_pos + 0.5))
        elif angle == 90:
            return (x_mesh == np.rint(x_pos - 0.5)) | (x_mesh == np.rint(x_pos + 0.5))
        else:
            y_values_1 = x_mesh * slope + intercept - 0.5
            y_values_2 = x_mesh * slope + intercept + 0.5
            x_values_1 = (y_mesh - intercept - 0.5) / slope
            x_values_2 = (y_mesh - intercept + 0.5) / slope
            return (y_mesh == np.rint(y_values_1)) | (y_mesh == np.rint(y_values_2)) | (x_mesh == np.rint(x_values_1)) | (x_mesh == np.rint(x_values_2))

    @staticmethod
    def get_img_mask_pix_in_circ(data, x_pos, y_pos, rad):
        x_pixels = np.arange(data.shape[1])
        y_pixels = np.arange(data.shape[0])
        x_mesh, y_mesh = np.meshgrid(x_pixels, y_pixels)

        return np.sqrt((x_mesh - x_pos) ** 2 + (y_mesh - y_pos) ** 2) < rad

    @staticmethod
    def select_img_pix_in_circ(data, x_pos, y_pos, rad):
        return data[GeometryTools.get_img_mask_pix_in_circ(data=data, x_pos=x_pos, y_pos=y_pos, rad=rad)]

    @staticmethod
    def get_2d_array_value_from_pix_coords(array, x_pos, y_pos):
        # it is important to know that an array is organized in rows first and then columns.
        # Thus, when selecting a value of an 2D array from coordinates it is important to know that this appears
        # flipped.
        return array[int(np.rint(y_pos)), int(np.rint(x_pos))]


class SpecHelper:
    def __init__(self):
        pass

    @staticmethod
    def get_dap_data_identifier(res, ssp_model=None):
        if res == 'copt':
            data_identifier = res + '_' + ssp_model
        else:
            data_identifier = res
        return data_identifier


class FuncAndModels:
    @staticmethod
    def lin_func(p, x):
        gradient, intersect = p
        return gradient * x + intersect

    @staticmethod
    def gauss1d(x_data, amp, mu, sig):
        return amp * np.exp(-(x_data - mu) ** 2 / (2 * sig ** 2))

    @staticmethod
    def gauss2d_rot(x, y, amp, x0, y0, sig_x, sig_y, theta):
        sigx2 = sig_x ** 2
        sigy2 = sig_y ** 2
        a = np.cos(theta) ** 2 / (2 * sigx2) + np.sin(theta) ** 2 / (2 * sigy2)
        b = np.sin(theta) ** 2 / (2 * sigx2) + np.cos(theta) ** 2 / (2 * sigy2)
        c = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

        expo = -a * (x - x0) ** 2 - b * (y - y0) ** 2 - 2 * c * (x - x0) * (y - y0)

        return amp * np.exp(expo)

    @staticmethod
    def moffat1d(rad, beta, alpha):
        return ((beta-1) / (np.pi*(alpha**2))) * (1 + (rad**2 / (alpha**2)))**(-beta)

    @staticmethod
    def moffat2d(x, y, beta, alpha):
        return ((beta-1) / (np.pi*(alpha**2))) * (1 + ((x**2 + y**2) / (alpha**2)))**(-beta)

    @staticmethod
    def star_cluster_moffat1d(rad, mu_0, nu, fwhm):
        """
        This function follows the nomenclature of Thilker+2022 2022MNRAS.509.4094T
        """
        characteristic_rad = fwhm / 2 * (2 ** (1 / nu) - 1)

        return mu_0 * (1 + rad**2 / (characteristic_rad ** 2)) ** (-nu)

    @staticmethod
    def star_cluster_moffat2d(x, y, x0, y0, mu_0, nu, fwhm):
        """
        This function follows the nomenclature of Thilker+2022 2022MNRAS.509.4094T
        """
        rad = np.sqrt((x - x0)**2 + (y - y0)**2)

        characteristic_rad = fwhm / 2 * (2 ** (1 / nu) - 1)

        return mu_0 * (1 + rad**2 / (characteristic_rad ** 2)) ** (-nu)


class FitTools:
    @staticmethod
    def lin_func(p, x):
        gradient, intersect = p
        return gradient * x + intersect

    @staticmethod
    def gaussian_func(x_data, amp, mu, sig):
        return amp * np.exp(-(x_data - mu) ** 2 / (2 * sig ** 2))

    @staticmethod
    def super_pos_two_gaussian_func(x_data, amp_1, amp_2, mu, sig_1, sig_2):
        return amp_1 * np.exp(-(x_data - mu) ** 2 / (2 * sig_1 ** 2)) + amp_2 * np.exp(-(x_data - mu) ** 2 / (2 * sig_2 ** 2))

    @staticmethod
    def fit_line(x_data, y_data, x_data_err, y_data_err):
        # Create a model for fitting.
        lin_model = odr.Model(FitTools.lin_func)

        # Create a RealData object using our initiated data from above.
        data = odr.RealData(x_data, y_data, sx=x_data_err, sy=y_data_err)

        # Set up ODR with the model and data.
        odr_object = odr.ODR(data, lin_model, beta0=[0., 1.])

        # Run the regression.
        out = odr_object.run()

        # Use the in-built pprint method to give us results.
        # out.pprint()

        gradient, intersect = out.beta
        gradient_err, intersect_err = out.sd_beta

        # calculate sigma around fit
        sigma = np.std(y_data - FitTools.lin_func(p=(gradient, intersect), x=x_data))

        return {
            'gradient': gradient,
            'intersect': intersect,
            'gradient_err': gradient_err,
            'intersect_err': intersect_err,
            'sigma': sigma
        }

    @staticmethod
    def fit_gauss_old(x_data, y_data, x_data_err, y_data_err, amp_guess=1, mu_guess=0, sig_guess=1):
        # Create a model for fitting.
        gauss_model = odr.Model(FitTools.gaussian_func)

        # Create a RealData object using our initiated data from above.
        data = odr.RealData(x_data, y_data, sx=x_data_err, sy=y_data_err)

        # Set up ODR with the model and data.
        odr_object = odr.ODR(data, gauss_model, beta0=[amp_guess, mu_guess, sig_guess])

        # Run the regression.
        out = odr_object.run()
        amp, mu, sig = out.beta
        amp_err, mu_err, sig_err = out.sd_beta

        return {'amp': amp, 'mu': mu, 'sig': sig, 'amp_err': amp_err, 'mu_err': mu_err, 'sig_err': sig_err}

    @staticmethod
    def fit_gauss(x_data, y_data, y_data_err=None,
                  amp_guess=1, mu_guess=0, sig_guess=1,
                  lower_amp=np.inf*-1, upper_amp=np.inf,
                  lower_mu=np.inf*-1, upper_mu=np.inf,
                  lower_sigma=0, upper_sigma=np.inf
                  ):



        initial_guess = [amp_guess, mu_guess, sig_guess]
        bounds = ([lower_amp, lower_mu, lower_sigma], [upper_amp, upper_mu, upper_sigma])  # (lower bounds, upper bounds)
        popt, pcov = curve_fit(f=FitTools.gaussian_func, xdata=x_data, ydata=y_data, sigma=y_data_err,
                               p0=initial_guess, bounds=bounds, nan_policy='omit')

        perr = np.sqrt(np.diag(pcov))

        amp, mu, sig = popt
        amp_err, mu_err, sig_err = perr

        return {'amp': amp, 'mu': mu, 'sig': sig, 'amp_err': amp_err, 'mu_err': mu_err, 'sig_err': sig_err}

    @staticmethod
    def fit_super_pos_two_gaussian(x_data, y_data, y_data_err=None,
                                   amp_1_guess=1, amp_2_guess=1, mu_guess=0, sig_1_guess=1, sig_2_guess=1,
                                   lower_amp_1=np.inf*-1, upper_amp_1=np.inf,
                                   lower_amp_2=np.inf*-1, upper_amp_2=np.inf,
                                   lower_mu=np.inf*-1, upper_mu=np.inf,
                                   lower_sigma_1=0, upper_sigma_1=np.inf,
                                   lower_sigma_2=0, upper_sigma_2=np.inf):


        initial_guess = [amp_1_guess, amp_2_guess, mu_guess, sig_1_guess, sig_2_guess]
        bounds = ([lower_amp_1, lower_amp_2, lower_mu, lower_sigma_1, lower_sigma_2],
                  [upper_amp_1, upper_amp_2, upper_mu, upper_sigma_1, upper_sigma_2])  # (lower bounds, upper bounds)

        popt, pcov = curve_fit(f=FitTools.super_pos_two_gaussian_func, xdata=x_data, ydata=y_data, sigma=y_data_err,
                               p0=initial_guess, bounds=bounds, nan_policy='omit')

        perr = np.sqrt(np.diag(pcov))

        amp_1, amp_2, mu, sig_1, sig_2 = popt
        amp_1_err, amp_2_err, mu_err, sig_1_err, sig_2_err = perr

        return {
            'amp_1': amp_1, 'amp_2': amp_2, 'mu': mu, 'sig_1': sig_1, 'sig_2': sig_2,
            'amp_1_err': amp_1_err, 'amp_2_err': amp_2_err, 'mu_err': mu_err, 'sig_1_err': sig_1_err,
            'sig_2_err': sig_2_err,
        }


class InterpTools:
    """
    Mainly for 2d interpolations
    """

    @staticmethod
    def interp2dgrid(x_bins, y_bins, func_values, method='linear'):
        """
        This function will only return the Scipy function RegularGridInterpolator


        Parameters
        ----------
        x_bins : array_like
        y_bins : array_like
        func_values : array_like

        method : str, optional
        The method of interpolation to perform. Supported are "linear",
        "nearest", "slinear", "cubic", "quintic" and "pchip". This
        parameter will become the default for the object's ``__call__``
        method. Default is "linear".
        """

        return RegularGridInterpolator((x_bins, y_bins), func_values, method)

    @staticmethod
    def get2dinterp_value(interp_func, x_val, y_val):
        """
        This function will only return the Scipy function RegularGridInterpolator


        Parameters
        ----------
        interp_func : ``scipy.interpolate.RegularGridInterpolator``
        x_val : float
        y_val : float

        """

        return interp_func([x_val, y_val])


    @staticmethod
    def get2dinterp_fine_grid(interp_func, x_min, x_max, y_min, y_max, n_x_bins=100, n_y_bins=100):
        """
        This function will only return the Scipy function RegularGridInterpolator


        Parameters
        ----------
        interp_func : ``scipy.interpolate.RegularGridInterpolator``
        x_min : float
        x_max : float
        y_min : float
        y_max : float
        n_x_bins : int
        n_y_bins : int

        """
        # get the meshgrid
        x_fine = np.linspace(x_min, x_max, n_x_bins)
        y_fine = np.linspace(y_min, y_max, n_y_bins)
        x_fine, y_fine = np.meshgrid(x_fine, y_fine)

        # Evaluate the interpolator on the finer grid
        points_fine = np.array([x_fine.ravel(), y_fine.ravel()]).T
        return interp_func(points_fine).reshape(x_fine.shape)
