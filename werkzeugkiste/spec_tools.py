"""
Tools for spectrsocopic analysis
"""
from os import path
from urllib import request
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy import constants as const
from scipy.constants import c as speed_of_light_mps
from ppxf.ppxf import ppxf
import ppxf.sps_util as lib
import ppxf.ppxf_util as util
from TardisPipeline.readData.MUSE_WFM import get_MUSE_polyFWHM
from werkzeugkiste import helper_func, phys_params
from werkzeugkiste.fit_tools import FitModels


class SpecHelper:
    def __init__(self):
        pass

    @staticmethod
    def conv_vel2delta_wave(line, vel, line_ref='vac_wave'):
        """
        Function to calculate the line wavelength shift due to velocity

        Parameters
        ----------
        line : int or float
        vel : float or ``astropy.units.Quantity``
            important: if the velocity has no astopy quantitiy the unit km/s will be assumed
        line_ref : str


        Returns
        -------
        wave_shift : ``astropy.units.Quantity``
            the unit is Angstrom
        """
        if not isinstance(vel, u.Quantity):
            vel *= (u.km / u.s)
        return vel.to(u.km / u.s) / const.c.to(u.km / u.s) * phys_params.spec_line_dict[line][line_ref] * u.AA

    @staticmethod
    def conv_delta_wave2vel(line, delta_wave, line_ref='vac_wave'):
        """
        Function to calculate the line wavelength shift due to velocity

        Parameters
        ----------
        line : int or float
        delta_wave : float or ``astropy.units.Quantity``
            important: if delta_wave has no astopy quantitiy the unit Angstrom will be assumed
        line_ref : str


        Returns
        -------
        vel : ``astropy.units.Quantity``
            the unit is km/s
        """
        if not isinstance(delta_wave, u.Quantity):
            delta_wave *= u.AA
        return delta_wave.to(u.AA) * const.c.to(u.km / u.s) / (phys_params.spec_line_dict[line][line_ref] * u.AA)

    @staticmethod
    def instrument2wave_ref(instrument):
        """
        Function to get reference wavelength estimator from instrument

        Parameters
        ----------
        instrument : str

        Returns
        -------
        wave_ref : str
        """
        assert (instrument in ['muse', 'manga', 'sdss'])
        if instrument == 'muse':
            return 'vac_wave'
        elif (instrument == 'manga') | (instrument == 'sdss'):
            return 'vac_wave'

    @staticmethod
    def get_inst_broad_sig(line, instrument='muse', return_value='vel', wave_ref=None):
        """
        Function to get instrumental broadening
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!! To Do: add further instruments !!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        Parameters
        ----------
        line : int or str
        instrument : str
        return_value : str
        wave_ref : str
        Returns
        -------
        inst_broad: ``astropy.units.Quantity``
        """

        assert (instrument in ['muse', 'manga', 'sdss'])
        assert (return_value in ['vel', 'wave'])
        if wave_ref is None:
            wave_ref = SpecHelper.instrument2wave_ref(instrument=instrument)
        if instrument == 'muse':
            wave = phys_params.spec_line_dict[line][wave_ref]
            inst_broad_sig = (get_MUSE_polyFWHM(x=wave) / (2 * np.sqrt(2 * np.log(2)))) * u.AA
            if return_value == 'wave':
                return inst_broad_sig
            elif return_value in ['vel']:
                return SpecHelper.conv_delta_wave2vel(line=line, delta_wave=inst_broad_sig, line_ref=wave_ref)
        else:
            raise KeyError(instrument, ' not understand')

    @staticmethod
    def get_target_ned_redshift(target):
        """
        Function to get redshift from NED with astroquery
        Parameters
        ----------
        target : str

        Returns
        -------
        redshift : float
        """

        from astroquery.ipac.ned import Ned
        # get the center of the target
        ned_table = Ned.query_object(target)

        return ned_table['Redshift'][0]

    @staticmethod
    def get_target_sys_vel(target=None, redshift=None):
        """
        Function to get systemic velocity based on redshift or NED redshift
        the conversion is based on eq.(8) of Cappellari (2017) (2017MNRAS.466..798C)
        Parameters
        ----------
        target : str
        redshift : float

        Returns
        -------
        sys_vel : ``astropy.units.Quantity``
            unit is km/ s
        """
        if redshift is not None:
            return np.log(redshift + 1) * const.c.to(u.km / u.s)
        elif target is not None:
            redshift = SpecHelper.get_target_ned_redshift(target=target)
            return np.log(redshift + 1) * const.c.to(u.km / u.s)
        else:
            raise KeyError(' either target or redshift must be given!')

    @staticmethod
    def vel2redshift(vel):
        """
        Function to convert spectral velocity to redshift
        the conversion is based on eq.(8) of Cappellari (2017) (2017MNRAS.466..798C)

        Parameters
        ----------
        vel : float or ``astropy.units.Quantity``
            important: if the velocity has no astopy quantitiy the unit km/s will be assumed

        Returns
        -------
        redshift : float
        """
        if not isinstance(vel, u.Quantity):
            vel *= (u.km / u.s)
        return np.exp(vel.to(u.km / u.s) / const.c.to(u.km / u.s)) - 1

    @staticmethod
    def conv_rest_wave2obs_wave(rest_wave, vel):
        """
        Function to convert restframe wavelength to an observed wavlelength.

        Parameters
        ----------
        rest_wave : float or ``astropy.units.Quantity``
        vel : float or ``astropy.units.Quantity``
            important: if the velocity has no astopy quantitiy the unit km/s will be assumed

        Returns
        -------
        obs_wave :  float or ``astropy.units.Quantity``
        """
        if not isinstance(vel, u.Quantity):
            vel *= (u.km / u.s)
        if not isinstance(rest_wave, u.Quantity):
            rest_wave *= u.AA
        return rest_wave.to(u.AA) * (1 + vel.to(u.km / u.s) / const.c.to(u.km / u.s))

    @staticmethod
    def get_line_pos(line, vel=None, target=None, redshift=None, instrument='muse', wave_ref=None):
        """
        Function to get the position of a line based on redshift or velocity

        Parameters
        ----------
        line : int or str
        vel : float or ``astropy.units.Quantity``
            important: if the velocity has no astopy quantitiy the unit km/s will be assumed
        target : str
        redshift : float
        instrument : str
        wave_ref : str

        Returns
        -------
        obs_wave : ``astropy.units.Quantity``
            wavelength is units of angstrom
        """

        if vel is None:
            vel = SpecHelper.get_target_sys_vel(target=target, redshift=redshift)
        if not isinstance(vel, u.Quantity):
            vel *= (u.km / u.s)

        if wave_ref is None:
            wave_ref = SpecHelper.instrument2wave_ref(instrument=instrument)

        return SpecHelper.conv_rest_wave2obs_wave(rest_wave=phys_params.spec_line_dict[line][wave_ref],
                                                 vel=vel.to(u.km / u.s))

    @staticmethod
    def conv_helio_cen_vel2obs_line_wave(vel, line, line_ref='vac_wave'):
        """
        Function to get the position of a line based on redshift or velocity

        Parameters
        ----------
        line : int or str
        vel : float or ``astropy.units.Quantity``
            important: if the velocity has no astopy quantitiy the unit km/s will be assumed
        target : str
        redshift : float
        instrument : str
        wave_ref : str

        Returns
        -------
        obs_wave : ``astropy.units.Quantity``
            wavelength is units of angstrom
        """
        if not isinstance(vel, u.Quantity):
            vel *= (u.km / u.s)

        return phys_params.spec_line_dict[line][line_ref] + SpecHelper.conv_vel2delta_wave(
            line=line, vel=vel.to(u.km / u.s), line_ref=line_ref)

    @staticmethod
    def conv_obs_line_wave2helio_cen_vel(obs_line_wave, line, vel_unit='kmps', line_ref='vac_wave'):
        """
        Function to get the position of a line based on redshift or velocity

        Parameters
        ----------
        obs_line_wave : float or ``astropy.units.Quantity``
            important: assuming units of Angstrom if not specified
        line : int or str
        target : str
        redshift : float
        instrument : str
        wave_ref : str

        Returns
        -------
        vel : ``astropy.units.Quantity``
            wavelength is units of km / s
        """

        if not isinstance(obs_line_wave, u.Quantity):
            obs_line_wave *= (u.AA)

        line_offset = obs_line_wave.to(u.AA) - phys_params.spec_line_dict[line][line_ref]
        return SpecHelper.conv_delta_wave2vel(line=line, delta_wave=line_offset, line_ref=line_ref)

    @staticmethod
    def get_kcwi_lsf_sig(wave):
        """
        Adopted from Eq. 11 of van Dokkum+2019 2019ApJ...880...91V

        Parameters
        ----------
        wave : float or ``astropy.units.Quantity``
            in Units of angstrom
        Return
        ---------
        lsf : ``astropy.units.Quantity``
            in Units of angstrom

        """
        if not isinstance(wave, u.Quantity):
            wave *= u.AA
        return (0.377 - 5.79e-5 * (wave.to(u.AA).value - 5000) - 1.144e-7 * ((wave.to(u.AA).value - 5000)**2)) * u.AA

    @staticmethod
    def get_kcwi_lsf_fwhm(wave):
        """
        Parameters
        ----------
        wave : float or ``astropy.units.Quantity``
            in Units of angstrom
        Return
        ---------
        lsf : ``astropy.units.Quantity``
            in Units of angstrom

        """
        return SpecHelper.get_kcwi_lsf_sig(wave=wave)*2*np.sqrt(2 * np.ln(2))

    @staticmethod
    def get_muse_lsf_fwhm(wave):
        """
        we use equation 8 of Bacon et al 2017
        which is using the LSF calibration of the ultra deep field - 10 in the HUDF

        Parameters
        ----------
        wave : float or ``astropy.units.Quantity``
            in Units of angstrom
        Return
        ---------
        lsf : ``astropy.units.Quantity``
            in Units of angstrom

        """
        if not isinstance(wave, u.Quantity):
            wave *= u.AA
        return get_MUSE_polyFWHM(wave.to(u.AA).value, version="udf10") * u.AA

    @staticmethod
    def get_muse_lsf_sig(wave):
        """

        Parameters
        ----------
        wave : float or ``astropy.units.Quantity``
            in Units of angstrom
        Return
        ---------
        lsf : ``astropy.units.Quantity``
            in Units of angstrom

        """
        return SpecHelper.get_muse_lsf_sig(wave=wave) / (2*np.sqrt(2 * np.ln(2)))

    @staticmethod
    def log_rebin_spec_data(wave, spec_flx, spec_flx_err):
        """
        function to rescale the spectra to a natural logarithm withthe smallest spectral steps available.

        Parameters
        ----------
        wave : array-like or ``astropy.units.Quantity``
        spec_flx : array-like or ``astropy.units.Quantity``
        spec_flx_err : array-like or ``astropy.units.Quantity``

        Returns
        -------
        ln_wave : array-like
        wave : array-like
        ln_rebin_spec_flx : array-like
        ln_rebin_spec_flx_err : array-like
        ln_rebin_velscale_kmps_per_pix : float
        """

        if isinstance(wave, u.Quantity):
            wave = wave.value

        if isinstance(spec_flx, u.Quantity):
            spec_flx = spec_flx.value

        if isinstance(spec_flx_err, u.Quantity):
            spec_flx_err = spec_flx_err.value

        # get the smallest velocity step
        ln_rebin_velscale_kmps_per_pix = np.min(const.c.to(u.km / u.s).value * np.diff(np.log(wave)))
        # rescale fluxes
        ln_rebin_spec_flx, ln_rebin_ln_wave, ln_rebin_velscale_kmps_per_pix = util.log_rebin(
            lam=wave, spec=spec_flx, velscale=ln_rebin_velscale_kmps_per_pix)
        # rescale the uncertainties
        ln_rebin_spec_flx_err, _, _ = util.log_rebin(lam=wave, spec=spec_flx_err,
                                                     velscale=ln_rebin_velscale_kmps_per_pix)
        # get also wavelength in linear form
        ln_rebin_lin_wave = np.exp(ln_rebin_ln_wave)

        return (ln_rebin_ln_wave, ln_rebin_lin_wave, ln_rebin_spec_flx, ln_rebin_spec_flx_err,
                ln_rebin_velscale_kmps_per_pix)










    @staticmethod
    def compute_gauss(x_data, line, amp, mu_vel, sig_vel, vel_unit='kmps', line_ref='vac_wave'):
        pos_peak = SpecHelper.conv_helio_cen_vel2obs_line_wave(line_vel=mu_vel, line=line, vel_unit=vel_unit,
                                                              line_ref=line_ref)
        sig_obs_wave = SpecHelper.conv_vel2delta_wave(line=line, vel=sig_vel, vel_unit=vel_unit, line_ref=line_ref)
        return FitModels.gaussian(x_values=x_data, amp=amp, mu=pos_peak, sig=sig_obs_wave)

    @staticmethod
    def get_obs_gauss_from_fit_output(x_data, em_line_fit_dict, line, gauss_index, line_type='nl', vel_unit='kmps',
                                      instrument='muse'):

        amp = em_line_fit_dict['amp_%s_%i_gauss_%i' % (line_type, line, gauss_index)]
        mu_vel = em_line_fit_dict['mu_%s_gauss_%i' % (line_type, gauss_index)]
        sig_int_vel = em_line_fit_dict['sig_%s_gauss_%i' % (line_type, gauss_index)]

        # get instrumental broadening
        sig_inst_broad_vel = SpecHelper.get_inst_broad_sig(line=line, instrument=instrument, unit='kmps')
        sig_obs_vel = np.sqrt(sig_int_vel ** 2 + sig_inst_broad_vel ** 2)
        return SpecHelper.compute_gauss(x_data=x_data, line=line, amp=amp, mu_vel=mu_vel, sig_vel=sig_obs_vel,
                                       vel_unit=vel_unit, line_ref=SpecHelper.instrument2wave_ref(instrument=instrument))

    @staticmethod
    def wave_window2mask(wave, wave_window):
        if isinstance(wave_window, tuple):
            mask_wave = (wave > wave_window[0]) & (wave < wave_window[1])
        else:
            mask_wave = np.zeros(len(wave), dtype=bool)
            for window_idx in range(wave_window.shape[0]):
                mask_wave += (wave > wave_window[window_idx, 0]) & (wave < wave_window[window_idx, 1])

        return mask_wave

    @staticmethod
    def compute_ew(wave, flux, flux_err, line_window, continuum_window):

        # get masks
        # mask_line = (wave > line_window[0]) & (wave < line_window[1])
        mask_line = SpecHelper.wave_window2mask(wave=wave, wave_window=line_window)
        mask_continuum = SpecHelper.wave_window2mask(wave=wave, wave_window=continuum_window)
        # # There can be ,multiple continuum windows
        # if isinstance(continuum_window, tuple):
        #     mask_continuum = (wave > continuum_window[0]) & (wave < continuum_window[1])
        # else:
        #     mask_continuum = np.zeros(len(wave), dtype=bool)
        #     for window_idx in range(continuum_window.shape[0]):
        #         mask_continuum += (wave > continuum_window[window_idx, 0]) & (wave < continuum_window[window_idx, 1])

        # estimate continuum
        mean_continuum = np.nanmean(flux[mask_continuum])
        std_continuum = np.nanstd(flux[mask_continuum])

        # we need an estimation for the wavelength bins size now
        # since the wavelength size does not change much, we can take the mean value
        wave_comp = wave[mask_line]
        delta_lambda = np.mean((wave_comp[1:] - wave_comp[:-1]) / 2)

        ew = np.nansum(((mean_continuum - flux[mask_line]) / mean_continuum) * delta_lambda)

        ew_err = np.sqrt((delta_lambda * std_continuum * np.nansum(flux[mask_line]) / (mean_continuum ** 2)) ** 2 +
                         np.nansum((delta_lambda * flux_err[mask_line] / mean_continuum) ** 2))

        # print('ew, ew_err ', ew, ew_err)
        # print('mean_continuum ', mean_continuum)
        # print('std_continuum ', std_continuum)

        # # # plot it to take a look
        # min_wave = np.min(np.concatenate([wave[mask_continuum], wave[mask_line]])) - 10
        # max_wave = np.max(np.concatenate([wave[mask_continuum], wave[mask_line]])) + 10
        #
        # min_flux = np.min(np.concatenate([flux[mask_continuum], flux[mask_line]]))
        # max_flux = np.max(np.concatenate([flux[mask_continuum], flux[mask_line]]))
        #
        # plt.step(wave, flux, where='mid', color='k')
        # plt.fill_between(wave[mask_line], flux[mask_line], mean_continuum, color='tab:blue')
        # plt.plot([np.min(wave[mask_continuum]), np.max(wave[mask_continuum])], [mean_continuum, mean_continuum], color='tab:orange')
        # # plt.fill_between(wave[mask_continuum], flux[mask_continuum], color='tab:orange')
        # plt.xlim(min_wave, max_wave)
        # plt.ylim(min_flux, max_flux)
        # plt.show()

        return_dict = {'mask_continuum': mask_continuum, 'mask_line': mask_line, 'ew': ew, 'ew_err': ew_err,
                       'mean_continuum': mean_continuum, 'std_continuum': std_continuum
                       }

        return return_dict

    @staticmethod
    def get_line_mask(wave, line, vel_kmps, target, instrument='muse', blue_limit=30., red_limit=30.):
        if line in (6550, 6565, 6585):
            nii_6550_observed_line = SpecHelper.get_line_pos(line=6550, vel_kmps=vel_kmps, target=target,
                                                            instrument=instrument)
            nii_6585_observed_line = SpecHelper.get_line_pos(line=6585, vel_kmps=vel_kmps, target=target,
                                                            instrument=instrument)
            return (wave > (nii_6550_observed_line - blue_limit)) & \
                (wave < nii_6585_observed_line + red_limit)
        elif line in (6718, 6733):
            sii_6718_observed_line = SpecHelper.get_line_pos(line=6718, vel_kmps=vel_kmps, target=target,
                                                            instrument=instrument)
            sii_6733_observed_line = SpecHelper.get_line_pos(line=6733, vel_kmps=vel_kmps, target=target,
                                                            instrument=instrument)
            return (wave > (sii_6718_observed_line - blue_limit)) & \
                (wave < sii_6733_observed_line + red_limit)
        else:
            obs_line = SpecHelper.get_line_pos(line=line, vel_kmps=vel_kmps, target=target, instrument=instrument)
            return (wave > (obs_line - blue_limit)) & \
                (wave < obs_line + red_limit)

    @staticmethod
    def get_multiple_line_mask(wave, ln_list, vel_kmps, target, instrument='muse', blue_limit=30., red_limit=30.):

        multi_line_mask = np.zeros(len(wave), dtype=bool)
        if ln_list is None:
            ln_list = [4863, 4960, 5008, 6302, 6550, 6565, 6585, 6718, 6733]

        for line in ln_list:
            multi_line_mask += SpecHelper.get_line_mask(wave=wave, line=line, vel_kmps=vel_kmps, target=target,
                                                       instrument=instrument,
                                                       blue_limit=blue_limit, red_limit=red_limit)

        return multi_line_mask


class PpxfTools:
    """
    Class to gather tools to use Ppxf
    """

    @staticmethod
    def get_stellar_template(velscale_kmps_per_pix, lsf_dict, age_range, metal_range, sps_name='fsps',
                             norm_range=[5070, 5950]):
        """
        pPXF can be used with any set of SPS population templates.
        From pPXF some ready-to-use template files for four SPS are provided.

        If you use the ``fsps`` v3.2 SPS model templates,
        please also cite in your paper Conroy+2009 (2009ApJ...699..486C) and Conroy+2010 (2010ApJ...712..833C).

        If you use the ``galaxev`` v2020 SPS model templates, please also cite in your paper
        Bruzual & Charlot (2003) (2003MNRAS.344.1000B).

        If you use the ``emiles`` SPS model templates, please also cite in your paper
        Vazdekis+2016 (2016MNRAS.463.3409V).
        WARNING: The E-MILES models only include SPS with age > 63 Myr and are not recommended for highly
        star forming galaxies.

        If you use the ``xsl`` Spectral Library (XSL) SPS model templates,
        please also cite in your paper Verro+2022 (2022A&A...661A..50V).
        WARNING: The XSL models only include SPS with age > 50 Myr and are not recommended for
        highly star forming galaxies.

        Parameters
        ----------
        velscale_kmps_per_pix : float
        lsf_dict : dict
        age_range : tuple or list
        metal_range : tuple or list
        sps_name : str
            Must be ``fsps``, ``galaxev``, ``emiles`` and ``xsl``
        norm_range : tuple or list
        Returns
        -------
        stellar_template_dict : dict
        """
        # get
        ppxf_dir = path.dirname(path.realpath(lib.__file__))
        basename = f"spectra_{sps_name}_9.0.npz"
        filename = path.join(ppxf_dir, 'sps_models', basename)
        if not path.isfile(filename):
            url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
            request.urlretrieve(url, filename)

        sps = lib.sps_lib(filename=filename, velscale=velscale_kmps_per_pix, fwhm_gal=lsf_dict, norm_range=norm_range,
                          age_range=age_range, metal_range=metal_range)
        # reshape the template to be later stacked with gas templates
        stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
        # shape of (n_ages, n_metal)
        n_age_n_met_sps_temp = sps.templates.shape[1:]
        # get linear wavelength of template
        wave_sps_temp = sps.lam_temp
        # get the logarithmic wavelength of the template
        ln_wave_sps_temp = sps.ln_lam_temp

        stellar_template_dict = {
            'sps': sps,
            'stars_templates': stars_templates,
            'n_age_n_met_sps_temp': n_age_n_met_sps_temp,
            'ln_wave_sps_temp': ln_wave_sps_temp,
            'wave_sps_temp': wave_sps_temp,
        }

        return stellar_template_dict

    @staticmethod
    def prepare_ppxf_fit_comp_dict(
            spec_dict,
            sys_vel=None, target_name=None,
            # kinematic components
            n_star_comp=1,
            sps_name='fsps', age_range=None, metal_range=None, norm_range=[5070, 5950],
            vel_init_offset_star=0, vel_bound_lo_star=-500, vel_bound_hi_star=500,
            init_vel_sig_star=100, sigma_bound_lo_star=0, sigma_bound_hi_star=500,
            allow_asym_star=True,
            # gas components
            # all lines
            n_all_lines=1,
            vel_init_offset_all_lines=0, vel_bound_lo_all_lines=-500, vel_bound_hi_all_lines=500,
            init_vel_sig_all_lines=100, sigma_bound_lo_all_lines=0, sigma_bound_hi_all_lines=500,
            allow_asym_all_lines=True,
            # only allowed lines
            n_only_allowed=0,
            vel_init_offset_only_allowed=0, vel_bound_lo_only_allowed=-500, vel_bound_hi_only_allowed=500,
            init_vel_sig_only_allowed=100, sigma_bound_lo_only_allowed=0, sigma_bound_hi_only_allowed=500,
            allow_asym_only_allowed=True,
            # only forbidden lines
            n_only_forbidden=0,
            vel_init_offset_only_forbidden=0, vel_bound_lo_only_forbidden=-500, vel_bound_hi_only_forbidden=500,
            init_vel_sig_only_forbidden=100, sigma_bound_lo_only_forbidden=0, sigma_bound_hi_only_forbidden=500,
            allow_asym_only_forbidden=True,
    ):

        """

        Parameters
        ----------
        general parameters
        spec_dict : dict
        sys_vel : float
        target_name : str

        parameters for the stellar component
        age_range : tuple or list
        metal_range : tuple or list
        norm_range : tuple
        n_star_comp : int
        sps_name : str
            can be fsps, galaxev or emiles
        vel_init_offset_star : float or list
        vel_bound_lo_star : float or list
        vel_bound_hi_star : float or list
        init_vel_sig_star : float or list
        sigma_bound_lo_star : float or list
        sigma_bound_hi_star : float or list
        allow_asym_star : bool or list

        Returns
        -------
        dict
        """

        if (target_name is None) & (sys_vel is None):
            raise KeyError(
                'in order to estimate a redshift / systematic velocity either ``target_name`` or ``sys_vel`` '
                'has to be provided')

        if (target_name is not None) & (sys_vel is None):
            sys_vel = SpecHelper.get_target_sys_vel(target=target_name)
        # get redshift
        redshift = SpecHelper.vel2redshift(vel=sys_vel)

        # To Do: add the possibility of multiple stellar components #
        if n_star_comp > 1:
            raise NotImplementedError('Not yet implemented to have more than 1 stellar component!')

        # get rebinned wavelength
        # for further computation we also compute a rebinned spectrum for
        ln_rebin_ln_wave, ln_rebin_lin_wave, ln_rebin_spec_flx, ln_rebin_spec_flx_err, ln_rebin_velscale_kmps_per_pix = (
            SpecHelper.log_rebin_spec_data(wave=spec_dict['native_wave'], spec_flx=spec_dict['native_spec_flx'],
                                                     spec_flx_err=spec_dict['native_spec_flx_err']))
        ln_rebin_good_pixel_mask = np.invert(np.isnan(ln_rebin_spec_flx) + np.isinf(ln_rebin_spec_flx))
        ln_rebin_lin_wave_range = [np.nanmin(ln_rebin_lin_wave), np.nanmax(ln_rebin_lin_wave)]

        # get stellar template
        stellar_template_dict = PpxfTools.get_stellar_template(
            velscale_kmps_per_pix=ln_rebin_velscale_kmps_per_pix,
            lsf_dict={"lam": spec_dict['native_wave'].value, "fwhm": spec_dict['lsf_fwhm'].value}, age_range=age_range,
            metal_range=metal_range, sps_name=sps_name, norm_range=norm_range)

        n_star_temps = stellar_template_dict['stars_templates'].shape[1]
        # make a list of all components that will enter the ppxf fit
        component = [0] * n_star_temps

        # get kinematic start values and boundaries
        if allow_asym_star:
            start_star = [sys_vel.value + vel_init_offset_star, init_vel_sig_star, 0, 0]
            moments = [4]
            bounds_stars = [[sys_vel.value + vel_bound_lo_star, sys_vel.value + vel_bound_hi_star],
                            [sigma_bound_lo_star, sigma_bound_hi_star],
                            [-0.3, 0.3], [-0.3, 0.3]]

        else:
            start_star = [sys_vel.value + vel_init_offset_star, init_vel_sig_star]
            moments = [2]
            bounds_stars = [[sys_vel.value + vel_bound_lo_star, sys_vel.value + vel_bound_hi_star],
                            [sigma_bound_lo_star, sigma_bound_hi_star]]
        start = [start_star]
        bounds = [bounds_stars]

        # Now add gas component
        # get standard gas template
        # To Do: add a more customizable version of the emission line template
        standard_gas_templates, standard_gas_names, standard_line_wave = util.emission_lines(
            ln_lam_temp=stellar_template_dict['ln_wave_sps_temp'], lam_range_gal=ln_rebin_lin_wave_range,
            FWHM_gal={"lam": spec_dict['native_wave'].value, "fwhm": spec_dict['lsf_fwhm'].value}, limit_doublets=False,
            tie_balmer=True, )

        # Note: this should be 1 unless the stellar continuum is fitted with multiple components
        current_gas_comp = n_star_comp
        gas_templates = None
        gas_names = []
        line_wave = []

        # go through all individual components
        # go through all lines
        for gas_comp_idx in range(n_all_lines):
            # get starting values and specify number of moments
            if allow_asym_all_lines:
                start_gas = [sys_vel.value + vel_init_offset_all_lines, init_vel_sig_all_lines, 0, 0]
                moments_gas = 4
                bounds_gas = [[sys_vel.value + vel_bound_lo_all_lines, sys_vel.value + vel_bound_hi_all_lines],
                              [sigma_bound_lo_all_lines, sigma_bound_hi_all_lines],
                              [-0.3, 0.3], [-0.3, 0.3]]
            else:
                start_gas = [sys_vel.value + vel_init_offset_all_lines, init_vel_sig_all_lines]
                moments_gas = 2
                bounds_gas = [[sys_vel.value + vel_bound_lo_all_lines, sys_vel.value + vel_bound_hi_all_lines],
                              [sigma_bound_lo_all_lines, sigma_bound_hi_all_lines]]
            start.append(start_gas)
            moments.append(moments_gas)
            bounds.append(bounds_gas)

            # add each individual gas component
            for gas_line_idx in range(len(standard_gas_names)):
                if gas_templates is None:
                    gas_templates = standard_gas_templates[:, gas_line_idx]
                else:
                    gas_templates = np.vstack([gas_templates, standard_gas_templates[:, gas_line_idx]])
                gas_names.append(standard_gas_names[gas_line_idx] + '_(%i)' % current_gas_comp)
                line_wave.append(standard_line_wave[gas_line_idx])
                component.append(current_gas_comp)
            # for the next line component use an individual gas component
            current_gas_comp += 1

        # now loop over allowed only
        for gas_comp_idx in range(n_only_allowed):
            # get starting values and specify number of moments
            if allow_asym_only_allowed:
                start_gas = [sys_vel.value + vel_init_offset_only_allowed, init_vel_sig_only_allowed, 0, 0]
                moments_gas = 4
                bounds_gas = [[sys_vel.value + vel_bound_lo_only_allowed, sys_vel.value + vel_bound_hi_only_allowed],
                              [sigma_bound_lo_only_allowed, sigma_bound_hi_only_allowed],
                              [-0.3, 0.3], [-0.3, 0.3]]
            else:
                start_gas = [sys_vel.value + vel_init_offset_only_allowed, init_vel_sig_only_allowed]
                moments_gas = 2
                bounds_gas = [[sys_vel.value + vel_bound_lo_only_allowed, sys_vel.value + vel_bound_hi_only_allowed],
                              [sigma_bound_lo_only_allowed, sigma_bound_hi_only_allowed]]
            start.append(start_gas)
            moments.append(moments_gas)
            bounds.append(bounds_gas)

            # add each individual gas component
            for gas_line_idx in range(len(standard_gas_names)):
                # if standard_gas_names[gas_line_idx][0] == '[': continue
                if standard_gas_names[gas_line_idx][0] != 'B': continue
                if gas_templates is None:
                    gas_templates = standard_gas_templates[:, gas_line_idx]
                else:
                    gas_templates = np.vstack([gas_templates, standard_gas_templates[:, gas_line_idx]])
                gas_names.append(standard_gas_names[gas_line_idx] + '_(%i)' % current_gas_comp)
                line_wave.append(standard_line_wave[gas_line_idx])
                component.append(current_gas_comp)
            # for the next line component use an individual gas component
            current_gas_comp += 1

        # now loop over forbidden only
        for gas_comp_idx in range(n_only_forbidden):
            # get starting values and specify number of moments
            if allow_asym_only_forbidden:
                start_gas = [sys_vel.value + vel_init_offset_only_forbidden, init_vel_sig_only_forbidden, 0, 0]
                moments_gas = 4
                bounds_gas = [[sys_vel.value + vel_bound_lo_only_forbidden, sys_vel.value + vel_bound_hi_only_forbidden],
                              [sigma_bound_lo_only_forbidden, sigma_bound_hi_only_forbidden],
                              [-0.3, 0.3], [-0.3, 0.3]]
            else:
                start_gas = [sys_vel.value + vel_init_offset_only_forbidden, init_vel_sig_only_forbidden]
                moments_gas = 2
                bounds_gas = [[sys_vel.value + vel_bound_lo_only_forbidden, sys_vel.value + vel_bound_hi_only_forbidden],
                              [sigma_bound_lo_only_forbidden, sigma_bound_hi_only_forbidden]]
            start.append(start_gas)
            moments.append(moments_gas)
            bounds.append(bounds_gas)

            # add each individual gas component
            for gas_line_idx in range(len(standard_gas_names)):
                if standard_gas_names[gas_line_idx][0] != '[': continue
                if gas_templates is None:
                    gas_templates = standard_gas_templates[:, gas_line_idx]
                else:
                    gas_templates = np.vstack([gas_templates, standard_gas_templates[:, gas_line_idx]])
                gas_names.append(standard_gas_names[gas_line_idx] + '_(%i)' % current_gas_comp)
                line_wave.append(standard_line_wave[gas_line_idx])
                component.append(current_gas_comp)
            # for the next line component use an individual gas component
            current_gas_comp += 1

        # bring templates in correct form and combine them
        gas_templates = gas_templates.T
        templates = np.column_stack([stellar_template_dict['stars_templates'], gas_templates])
        # get mask of gas components
        mask_gas_component = np.array(component) > (n_star_comp - 1)

        ppxf_comp_dict = {
            # log-rebinned spectrum
            'ln_rebin_lin_wave_range': ln_rebin_lin_wave_range,
            'ln_rebin_ln_wave': ln_rebin_ln_wave,
            'ln_rebin_lin_wave': ln_rebin_lin_wave,
            'ln_rebin_spec_flx': ln_rebin_spec_flx,
            'ln_rebin_spec_flx_err': ln_rebin_spec_flx_err,
            'ln_rebin_velscale_kmps_per_pix': ln_rebin_velscale_kmps_per_pix,
            'ln_rebin_good_pixel_mask': ln_rebin_good_pixel_mask,
            # fit components
            'stellar_template_dict': stellar_template_dict,
            'templates': templates,
            'component': component,
            'moments': moments,
            'start': start,
            'bounds': bounds,
            'mask_gas_component': mask_gas_component,
            'gas_names': gas_names,
            'line_wave': line_wave,
            'sys_vel': sys_vel,
            'redshift': redshift,
        }

        return ppxf_comp_dict

    @staticmethod
    def fit_ppxf2spec(
            # general parameters
            spec_dict,
            ppxf_comp_dict=None,
            sys_vel=None,
            target_name=None,
            degree=4, mdegree=0,
            init_star_dust_av=0.5, border_star_dust_av_lo=0, border_start_dus_av_hi=4,
            init_gas_dust_av=0.5, border_gas_dust_av_lo=0, border_gast_dus_av_hi=8,
            # other params
            verbose_flag=True):
        """

        Parameters
        ----------
        spec_dict : dict

        degree: int, optional
            Degree of the *additive* Legendre polynomial used to correct the
            template continuum shape during the fit (default: 4). This uses the
            standard mathematical definition where e.g. ``degree=2`` is a
            quadratic polynomial. Set ``degree=-1`` not to include any additive
            polynomial.

        mdegree: int, optional
            Degree of the *multiplicative* Legendre polynomial (with a mean of 1)
            used to correct the continuum shape during the fit (default: 0). The
            zero degree multiplicative polynomial (i.e. constant) is always
            included in the fit as it corresponds to the multiplicative weights
            assigned to the templates. Note that the computation time is longer
            with multiplicative polynomials than with the same ``degree`` of
            additive polynomials.

        Returns
        -------
        dict
        """

        if ppxf_comp_dict is None:
            ppxf_comp_dict = PpxfTools.prepare_ppxf_fit_comp_dict(spec_dict=spec_dict, sys_vel=sys_vel,
                                                                  target_name=target_name, )

        # get bounds for the dust attenuation
        dust_gas_1 = {"start": [init_gas_dust_av],
                      "bounds": [[border_gas_dust_av_lo, border_gast_dus_av_hi]],
                      "component": np.array(ppxf_comp_dict['component']) == 1}

        dust_gas_2 = {"start": [init_gas_dust_av],
                      "bounds": [[border_gas_dust_av_lo, border_gast_dus_av_hi]],
                      "component": np.array(ppxf_comp_dict['component']) == 2}

        dust_stars = {"start": [init_star_dust_av],
                      "bounds": [[border_star_dust_av_lo, border_start_dus_av_hi]],
                      "component": ~ppxf_comp_dict['mask_gas_component']}
        dust = [dust_gas_1, dust_gas_2, dust_stars]

        pp = ppxf(
            templates=ppxf_comp_dict['templates'],
            galaxy=ppxf_comp_dict['ln_rebin_spec_flx'], noise=ppxf_comp_dict['ln_rebin_spec_flx_err'],
            mask=ppxf_comp_dict['ln_rebin_good_pixel_mask'],
            velscale=ppxf_comp_dict['ln_rebin_velscale_kmps_per_pix'],
            start=ppxf_comp_dict['start'], moments=ppxf_comp_dict['moments'], bounds=ppxf_comp_dict['bounds'],
            degree=degree, mdegree=mdegree,
            global_search=False,
            lam=ppxf_comp_dict['ln_rebin_lin_wave'], lam_temp=ppxf_comp_dict['stellar_template_dict']['wave_sps_temp'],
            reg_dim=ppxf_comp_dict['stellar_template_dict']['n_age_n_met_sps_temp'],
            component=ppxf_comp_dict['component'],
            gas_component=ppxf_comp_dict['mask_gas_component'],
            dust=dust,
            gas_names=ppxf_comp_dict['gas_names'],
        )

        # print(pp.dust)
        # print(pp.dust[0]['sol'][0], pp.dust[1]['sol'][0], pp.dust[2]['sol'][0])
        # pp.plot()
        # plt.show()
        #
        light_weights = pp.weights[~ppxf_comp_dict['mask_gas_component']]  # Exclude weights of the gas templates
        light_weights = light_weights.reshape(
            ppxf_comp_dict['stellar_template_dict']['n_age_n_met_sps_temp'])  # Reshape to (n_ages, n_metal)
        light_weights /= light_weights.sum()  # Normalize to light fractions

        # plt.figure(figsize=(9, 3))
        # ppxf_comp_dict['stellar_template_dict']['sps'].plot(light_weights)
        # plt.title("Light Weights Fractions")
        # plt.tight_layout()
        # plt.show()

        ages, met = ppxf_comp_dict['stellar_template_dict']['sps'].mean_age_metal(light_weights)
        mass2light = ppxf_comp_dict['stellar_template_dict']['sps'].mass_to_light(light_weights,
                                                                                  redshift=ppxf_comp_dict['redshift'])

        wave = pp.lam
        total_flux = pp.galaxy
        total_flux_err = pp.noise

        best_fit = pp.bestfit
        gas_best_fit = pp.gas_bestfit
        continuum_best_fit = best_fit - gas_best_fit

        # get velocity of balmer component
        sol_kin_comp = pp.sol[0]
        balmer_kin_comp = pp.sol[1]
        forbidden_kin_comp = pp.sol[2]

        ppxf_dict = {
            'wave': wave, 'total_flux': total_flux, 'total_flux_err': total_flux_err,
            'best_fit': best_fit, 'gas_best_fit': gas_best_fit, 'continuum_best_fit': continuum_best_fit,
            'ages': ages, 'met': met, 'mass2light': mass2light,
            'pp': pp,
            'star_red': pp.dust[0]['sol'][0], 'gas_red': pp.dust[1]['sol'][0],
            'sol_kin_comp': sol_kin_comp, 'balmer_kin_comp': balmer_kin_comp, 'forbidden_kin_comp': forbidden_kin_comp,
            'sys_vel': sys_vel, 'redshift': ppxf_comp_dict['redshift'], 'rad_arcsec': spec_dict['rad_arcsec']
        }

        return ppxf_dict


class LineSpecFit:
    """
    This class gathers all methods to provide a standalone emission line fitting and analysis method that works on
    continuum subtracted spectra

    """

    @staticmethod
    def fit_em_lines2spec(target, wave, em_flux, em_flux_err, sys_vel=None, ln_list=None, n_nl_gauss=1, n_nl_lorentz=0,
                          n_bl_gauss=0,
                          x_data_format='wave', instrument='muse', blue_limit=30., red_limit=30., search_outflow=True,
                          outflow_shift='redshift', outflow_mu_offset=400, outflow_sig=1200,
                          init_mu_nl_gauss=100, init_sig_nl_gauss=200):

        if ln_list is None:
            ln_list = [4863, 4960, 5008, 6550, 6565, 6585, 6718, 6733]
            # ln_list = [6718, 6733]

        if sys_vel is None:
            sys_vel = SpecHelper.get_target_sys_vel(target=target)
        print('sys_vel ', sys_vel)
        # get data
        ln_mask = SpecHelper.get_multiple_line_mask(wave=wave, ln_list=ln_list, vel_kmps=sys_vel, target=target,
                                                   instrument=instrument,
                                                   blue_limit=blue_limit, red_limit=red_limit)

        # plt.close()
        # plt.plot(wave[ln_mask], em_flux[ln_mask])
        # plt.show()
        # exit()

        # get systematic velocity
        dict_inst_broad = {}
        for line in ln_list:
            dict_inst_broad.update(
                {line: SpecHelper.get_inst_broad_sig(line=line, instrument=instrument, unit='kmps')})

        # initialize emission line fit
        fit_model = FitModels()
        fit_model.set_up_model(x_data=wave[ln_mask], flx=em_flux[ln_mask], flx_err=em_flux_err[ln_mask],
                               n_nl_gauss=n_nl_gauss, n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
                               ln_list=ln_list, dict_inst_broad=dict_inst_broad, x_data_format=x_data_format)

        fit_param_restrict_dict_nl_gauss, fit_param_restrict_dict_nl_lorentz, fit_param_restrict_dict_bl_gauss = \
            SpecHelper.get_fit_param_restrict_dict_outflow_search(target=target, n_nl_gauss=n_nl_gauss,
                                                                 n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
                                                                 balmer_ln=fit_model.balmer_ln, all_ln=fit_model.all_ln,
                                                                 wave=wave, em_flux=em_flux,
                                                                 sys_vel=sys_vel,
                                                                 instrument=instrument,
                                                                 search_outflow=search_outflow,
                                                                 outflow_shift=outflow_shift,
                                                                 outflow_mu_offset=outflow_mu_offset,
                                                                 outflow_sig=outflow_sig,
                                                                 init_mu_nl_gauss=init_mu_nl_gauss,
                                                                 init_sig_nl_gauss=init_sig_nl_gauss
                                                                 )
        print(fit_param_restrict_dict_nl_gauss)

        fit_param_dict = fit_model.run_fit(fit_param_restrict_dict_nl_gauss=fit_param_restrict_dict_nl_gauss,
                                           fit_param_restrict_dict_nl_lorentz=fit_param_restrict_dict_nl_lorentz,
                                           fit_param_restrict_dict_bl_gauss=fit_param_restrict_dict_bl_gauss)

        fit_param_dict.update({
            'sys_vel': sys_vel,
            'ln_list': ln_list,
            'wave': wave,
            'em_flux': em_flux,
            'em_flux_err': em_flux_err,
            'ln_mask': ln_mask,
            'dict_inst_broad': dict_inst_broad,
            'n_nl_gauss': n_nl_gauss,
            'n_nl_lorentz': n_nl_lorentz,
            'n_bl_gauss': n_bl_gauss})

        return fit_param_dict

    @staticmethod
    def estimate_line_amp(line, wave, em_flux, vel=None, target=None, redshift=None, instrument='muse', bin_rad=4):
        # get line position
        line_pos = SpecHelper.get_line_pos(line=line, vel_kmps=vel, target=target, redshift=redshift,
                                          instrument=instrument)
        # get wavelength steps
        # print(np.where(wave == np.wave - line_pos))
        closest_idx = (np.abs(wave - line_pos)).argmin()
        return np.nanmax(em_flux[closest_idx - bin_rad: closest_idx + bin_rad])

    @staticmethod
    def get_fit_param_restrict_dict_outflow_search(target,

                                                   n_nl_gauss, n_nl_lorentz, n_bl_gauss, balmer_ln, all_ln,
                                                   wave, em_flux,
                                                   sys_vel=None,
                                                   instrument='muse',
                                                   search_outflow=True,
                                                   outflow_shift='blueshift', outflow_mu_offset=0, outflow_sig=1200,
                                                   init_amp_nl_gauss_frac=1, lower_rel_amp_nl_gauss=0.0,
                                                   upper_rel_amp_nl_gauss=2,
                                                   amp_nl_gauss_floating=True,
                                                   init_mu_nl_gauss=200, lower_mu_nl_gauss=-500, upper_mu_nl_gauss=500,
                                                   mu_nl_gauss_floating=True,
                                                   init_sig_nl_gauss=100, lower_sig_nl_gauss=0, upper_sig_nl_gauss=700,
                                                   sig_nl_gauss_floating=True,

                                                   init_amp_nl_lorentz_frac=1, lower_rel_amp_nl_lorentz=0,
                                                   upper_rel_amp_nl_lorentz=2,
                                                   amp_nl_lorentz_floating=True,
                                                   init_x0_nl_lorentz=100, lower_x0_nl_lorentz=-1000,
                                                   upper_x0_nl_lorentz=1000,
                                                   x0_nl_lorentz_floating=True,
                                                   init_gam_nl_lorentz=100, lower_gam_nl_lorentz=0,
                                                   upper_gam_nl_lorentz=700,
                                                   gam_nl_lorentz_floating=True,

                                                   init_amp_bl_gauss_frac=0.1, lower_rel_amp_bl_gauss=0,
                                                   upper_rel_amp_bl_gauss=0.5,
                                                   amp_bl_gauss_floating=True,
                                                   init_mu_bl_gauss=100, lower_mu_bl_gauss=-1000,
                                                   upper_mu_bl_gauss=1000,
                                                   mu_bl_gauss_floating=True,
                                                   init_sig_bl_gauss=1000, lower_sig_bl_gauss=500,
                                                   upper_sig_bl_gauss=4000,
                                                   sig_bl_gauss_floating=True,

                                                   ):
        """

        Parameters
        ----------
        init_amp_nl_gauss_frac
        lower_rel_amp_nl_gauss
        upper_rel_amp_nl_gauss
        amp_nl_gauss_floating
        init_mu_nl_gauss
        lower_mu_nl_gauss
        upper_mu_nl_gauss
        mu_nl_gauss_floating
        init_sig_nl_gauss
        lower_sig_nl_gauss
        upper_sig_nl_gauss
        sig_nl_gauss_floating
        init_amp_nl_lorentz_frac
        lower_rel_amp_nl_lorentz
        upper_rel_amp_nl_lorentz
        amp_nl_lorentz_floating
        init_x0_nl_lorentz
        lower_x0_nl_lorentz
        upper_x0_nl_lorentz
        x0_nl_lorentz_floating
        init_gam_nl_lorentz
        lower_gam_nl_lorentz
        upper_gam_nl_lorentz
        gam_nl_lorentz_floating
        init_amp_bl_gauss_frac
        lower_rel_amp_bl_gauss
        upper_rel_amp_bl_gauss
        amp_bl_gauss_floating
        init_mu_bl_gauss
        lower_mu_bl_gauss
        upper_mu_bl_gauss
        mu_bl_gauss_floating
        init_sig_bl_gauss
        lower_sig_bl_gauss
        upper_sig_bl_gauss
        sig_bl_gauss_floating

        Returns
        -------

        """
        # get systematic velocity
        if sys_vel is None:
            sys_vel = SpecHelper.get_target_sys_vel(target=target)

        # create the empty parameter dict
        fit_param_restrict_dict_nl_gauss = {}
        fit_param_restrict_dict_nl_lorentz = {}
        fit_param_restrict_dict_bl_gauss = {}

        # make sure that all initial parameters are in list format
        # narrow line gauss
        # prepare parameters for mu
        if isinstance(init_mu_nl_gauss, list):
            init_mu_nl_gauss_pos = init_mu_nl_gauss
        else:
            # get equally distributed mu position inside +/- init_mu_nl_gauss
            init_mu_nl_gauss_pos = []
            for index in range(n_nl_gauss):
                if (index == 0) & search_outflow:
                    if outflow_shift == 'blueshift':
                        mu_offset = sys_vel - outflow_mu_offset
                    elif outflow_shift == 'redshift':
                        mu_offset = sys_vel + outflow_mu_offset
                    else:
                        raise KeyError('outflow_mu_offset must be redshift or bueshift')
                    init_mu_nl_gauss_pos.append(mu_offset)
                else:
                    init_mu_nl_gauss_pos.append(
                        sys_vel - init_mu_nl_gauss + ((2 * init_mu_nl_gauss / (n_nl_gauss + 1)) * (index + 1)))
        # boundaries
        if outflow_shift == 'blueshift':
            lower_mu_limit_outflow = - outflow_mu_offset - 500
            upper_mu_limit_outflow = - outflow_mu_offset + 500
        elif outflow_shift == 'redshift':
            lower_mu_limit_outflow = + outflow_mu_offset - 500
            upper_mu_limit_outflow = + outflow_mu_offset + 500
        else:
            raise KeyError('outflow_mu_offset must be redshift or bueshift')

        if not isinstance(lower_mu_nl_gauss, list):
            lower_mu_nl_gauss = [lower_mu_nl_gauss] * n_nl_gauss
            if search_outflow:
                lower_mu_nl_gauss[0] = lower_mu_limit_outflow
        if not isinstance(upper_mu_nl_gauss, list):
            upper_mu_nl_gauss = [upper_mu_nl_gauss] * n_nl_gauss
            if search_outflow:
                upper_mu_nl_gauss[0] = upper_mu_limit_outflow
        if not isinstance(mu_nl_gauss_floating, list):
            mu_nl_gauss_floating = [mu_nl_gauss_floating] * n_nl_gauss

        init_sig_outflow = outflow_sig
        lower_sig_limit_outflow = 300
        upper_sig_limit_outflow = 2000

        # prepare parameters for sigma
        if not isinstance(init_sig_nl_gauss, list):
            init_sig_nl_gauss = [init_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                init_sig_nl_gauss[0] = init_sig_outflow
        if not isinstance(lower_sig_nl_gauss, list):
            lower_sig_nl_gauss = [lower_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                lower_sig_nl_gauss[0] = lower_sig_limit_outflow
        if not isinstance(upper_sig_nl_gauss, list):
            upper_sig_nl_gauss = [upper_sig_nl_gauss] * n_nl_gauss
            if search_outflow:
                upper_sig_nl_gauss[0] = upper_sig_limit_outflow
        if not isinstance(sig_nl_gauss_floating, list):
            sig_nl_gauss_floating = [sig_nl_gauss_floating] * n_nl_gauss
        # prepare parameters for amplitudes
        if not isinstance(init_amp_nl_gauss_frac, list):
            init_amp_nl_gauss_frac = [init_amp_nl_gauss_frac] * n_nl_gauss
        if not isinstance(lower_rel_amp_nl_gauss, list):
            lower_rel_amp_nl_gauss = [lower_rel_amp_nl_gauss] * n_nl_gauss
        if not isinstance(upper_rel_amp_nl_gauss, list):
            upper_rel_amp_nl_gauss = [upper_rel_amp_nl_gauss] * n_nl_gauss
        if not isinstance(amp_nl_gauss_floating, list):
            amp_nl_gauss_floating = [amp_nl_gauss_floating] * n_nl_gauss

        # narrow line Lorentz
        # prepare parameters for mu
        if isinstance(init_x0_nl_lorentz, list):
            init_x0_nl_lorentz_pos = init_x0_nl_lorentz
        else:
            # get equally distributed x0 position inside +/- init_x0_nl_lorentz
            init_x0_nl_lorentz_pos = []
            for index in range(n_nl_lorentz):
                init_x0_nl_lorentz_pos.append(
                    sys_vel - init_x0_nl_lorentz + ((2 * init_x0_nl_lorentz / (n_nl_lorentz + 1)) * (index + 1)))
        if not isinstance(lower_x0_nl_lorentz, list):
            lower_x0_nl_lorentz = [lower_x0_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_x0_nl_lorentz, list):
            upper_x0_nl_lorentz = [upper_x0_nl_lorentz] * n_nl_lorentz
        if not isinstance(x0_nl_lorentz_floating, list):
            x0_nl_lorentz_floating = [x0_nl_lorentz_floating] * n_nl_lorentz
        # prepare parameters for sigma
        if not isinstance(init_gam_nl_lorentz, list):
            init_gam_nl_lorentz = [init_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(lower_gam_nl_lorentz, list):
            lower_gam_nl_lorentz = [lower_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_gam_nl_lorentz, list):
            upper_gam_nl_lorentz = [upper_gam_nl_lorentz] * n_nl_lorentz
        if not isinstance(gam_nl_lorentz_floating, list):
            gam_nl_lorentz_floating = [gam_nl_lorentz_floating] * n_nl_lorentz
        # prepare parameters for amplitudes
        if not isinstance(init_amp_nl_lorentz_frac, list):
            init_amp_nl_lorentz_frac = [init_amp_nl_lorentz_frac] * n_nl_lorentz
        if not isinstance(lower_rel_amp_nl_lorentz, list):
            lower_rel_amp_nl_lorentz = [lower_rel_amp_nl_lorentz] * n_nl_lorentz
        if not isinstance(upper_rel_amp_nl_lorentz, list):
            upper_rel_amp_nl_lorentz = [upper_rel_amp_nl_lorentz] * n_nl_lorentz
        if not isinstance(amp_nl_lorentz_floating, list):
            amp_nl_lorentz_floating = [amp_nl_lorentz_floating] * n_nl_lorentz

        # broad line gauss
        # prepare parameters for mu
        if isinstance(init_mu_bl_gauss, list):
            init_mu_bl_gauss_pos = init_mu_bl_gauss
        else:
            # get equally distributed mu position inside +/- init_mu_bl_gauss
            init_mu_bl_gauss_pos = []
            for index in range(n_bl_gauss):
                init_mu_bl_gauss_pos.append(
                    sys_vel - init_mu_bl_gauss + ((2 * init_mu_bl_gauss / (n_bl_gauss + 1)) * (index + 1)))
        if not isinstance(lower_mu_bl_gauss, list):
            lower_mu_bl_gauss = [lower_mu_bl_gauss] * n_bl_gauss
        if not isinstance(upper_mu_bl_gauss, list):
            upper_mu_bl_gauss = [upper_mu_bl_gauss] * n_bl_gauss
        if not isinstance(mu_bl_gauss_floating, list):
            mu_bl_gauss_floating = [mu_bl_gauss_floating] * n_bl_gauss
        # prepare parameters for sigma
        if not isinstance(init_sig_bl_gauss, list):
            init_sig_bl_gauss = [init_sig_bl_gauss] * n_bl_gauss
        if not isinstance(lower_sig_bl_gauss, list):
            lower_sig_bl_gauss = [lower_sig_bl_gauss] * n_bl_gauss
        if not isinstance(upper_sig_bl_gauss, list):
            upper_sig_bl_gauss = [upper_sig_bl_gauss] * n_bl_gauss
        if not isinstance(sig_bl_gauss_floating, list):
            sig_bl_gauss_floating = [sig_bl_gauss_floating] * n_bl_gauss
        # prepare parameters for amplitudes
        if not isinstance(init_amp_bl_gauss_frac, list):
            init_amp_bl_gauss_frac = [init_amp_bl_gauss_frac] * n_bl_gauss
        if not isinstance(lower_rel_amp_bl_gauss, list):
            lower_rel_amp_bl_gauss = [lower_rel_amp_bl_gauss] * n_bl_gauss
        if not isinstance(upper_rel_amp_bl_gauss, list):
            upper_rel_amp_bl_gauss = [upper_rel_amp_bl_gauss] * n_bl_gauss
        if not isinstance(amp_bl_gauss_floating, list):
            amp_bl_gauss_floating = [amp_bl_gauss_floating] * n_bl_gauss

        # fill all emission lines
        for gauss_index in range(n_nl_gauss):
            # add mu and sigma parameters
            fit_param_restrict_dict_nl_gauss.update(
                {'nl_gauss_%i' % gauss_index: {'mu': init_mu_nl_gauss_pos[gauss_index],
                                               'lower_mu': sys_vel + lower_mu_nl_gauss[gauss_index],
                                               'upper_mu': sys_vel + upper_mu_nl_gauss[gauss_index],
                                               'mu_floating': mu_nl_gauss_floating[gauss_index],
                                               'sig': init_sig_nl_gauss[gauss_index],
                                               'lower_sig': lower_sig_nl_gauss[gauss_index],
                                               'upper_sig': upper_sig_nl_gauss[gauss_index],
                                               'sig_floating': sig_nl_gauss_floating[gauss_index]}})
            # add amplitude paramaeters
            for line in all_ln:
                # if gauss_index == 0:
                #     if line == 5008:
                #         init_amp = SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) * 0.4
                #     else:
                #         init_amp = SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target, instrument=instrument, bin_rad=4) * 0.01
                # else:
                init_amp = SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                       instrument=instrument, bin_rad=4) * init_amp_nl_gauss_frac[
                               gauss_index]
                fit_param_restrict_dict_nl_gauss['nl_gauss_%i' % gauss_index].update({
                    'amp_%i' % line: init_amp,
                    'lower_amp_%i' % line: init_amp * lower_rel_amp_nl_gauss[gauss_index],
                    'upper_amp_%i' % line: init_amp * upper_rel_amp_nl_gauss[gauss_index],
                    'amp_floating_%i' % line: amp_nl_gauss_floating[gauss_index]
                })

        for lorentz_index in range(n_nl_lorentz):
            # add mu and sigma parameters
            fit_param_restrict_dict_nl_lorentz.update(
                {'nl_lorentz_%i' % lorentz_index: {'x0': init_x0_nl_lorentz_pos[lorentz_index],
                                                   'lower_x0': sys_vel + lower_x0_nl_lorentz[lorentz_index],
                                                   'upper_x0': sys_vel + upper_x0_nl_lorentz[lorentz_index],
                                                   'x0_floating': x0_nl_lorentz_floating[lorentz_index],
                                                   'gam': init_gam_nl_lorentz[lorentz_index],
                                                   'lower_gam': lower_gam_nl_lorentz[lorentz_index],
                                                   'upper_gam': upper_gam_nl_lorentz[lorentz_index],
                                                   'gam_floating': gam_nl_lorentz_floating[lorentz_index]}})
            # add amplitude paramaeters
            for line in all_ln:
                fit_param_restrict_dict_nl_lorentz['nl_lorentz_%i' % lorentz_index].update({
                    'amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        init_amp_nl_lorentz_frac[lorentz_index],
                    'lower_amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        lower_rel_amp_nl_lorentz[lorentz_index],
                    'upper_amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        upper_rel_amp_nl_lorentz[lorentz_index],
                    'amp_floating_%i' % line: amp_nl_lorentz_floating[lorentz_index]
                })

        for gauss_index in range(n_bl_gauss):
            # add mu and sigma parameters
            fit_param_restrict_dict_bl_gauss.update(
                {'bl_gauss_%i' % gauss_index: {'mu': init_mu_bl_gauss_pos[gauss_index],
                                               'lower_mu': sys_vel + lower_mu_bl_gauss[gauss_index],
                                               'upper_mu': sys_vel + upper_mu_bl_gauss[gauss_index],
                                               'mu_floating': mu_bl_gauss_floating[gauss_index],
                                               'sig': init_sig_bl_gauss[gauss_index],
                                               'lower_sig': lower_sig_bl_gauss[gauss_index],
                                               'upper_sig': upper_sig_bl_gauss[gauss_index],
                                               'sig_floating': sig_bl_gauss_floating[gauss_index]}})
            # add amplitude paramaeters
            for line in balmer_ln:
                fit_param_restrict_dict_bl_gauss['bl_gauss_%i' % gauss_index].update({
                    'amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        init_amp_bl_gauss_frac[gauss_index],
                    'lower_amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        lower_rel_amp_bl_gauss[gauss_index],
                    'upper_amp_%i' % line:
                        SpecHelper.estimate_line_amp(line=line, wave=wave, em_flux=em_flux, target=target,
                                                    instrument=instrument, bin_rad=4) *
                        upper_rel_amp_bl_gauss[gauss_index],
                    'amp_floating_%i' % line: amp_bl_gauss_floating[gauss_index]
                })

        return fit_param_restrict_dict_nl_gauss, fit_param_restrict_dict_nl_lorentz, fit_param_restrict_dict_bl_gauss

    @staticmethod
    def fit_nad_line(ppxf_fit_dict, em_line_fit_dict, target):

        line_1 = 5892
        line_2 = 5898

        # now get the absorption line position
        line_mask = SpecHelper.get_multiple_line_mask(wave=ppxf_fit_dict['wave'],
                                                     ln_list=[5892, 5898],
                                                     vel_kmps=ppxf_fit_dict['sol_kin_comp'][0],
                                                     target=target, instrument='muse', blue_limit=30., red_limit=50.)

        plt.plot(ppxf_fit_dict['wave'][line_mask], ppxf_fit_dict['total_flux'][line_mask])
        plt.show()

    @staticmethod
    def measure_caii_triplet(ppxf_fit_dict, plot_results=True):
        line_caii_1 = 8500
        line_caii_2 = 8544
        line_caii_3 = 8665

        line_pos_caii_1 = SpecHelper.get_line_pos(line=line_caii_1, vel_kmps=ppxf_fit_dict['sol_kin_comp'][0],
                                                 target=None, redshift=None, instrument='muse')
        line_pos_caii_2 = SpecHelper.get_line_pos(line=line_caii_2, vel_kmps=ppxf_fit_dict['sol_kin_comp'][0],
                                                 target=None, redshift=None, instrument='muse')
        line_pos_caii_3 = SpecHelper.get_line_pos(line=line_caii_3, vel_kmps=ppxf_fit_dict['sol_kin_comp'][0],
                                                 target=None, redshift=None, instrument='muse')
        # get the sizes of absorption lines
        sig_int_vel = ppxf_fit_dict['sol_kin_comp'][1]
        # we will calculate the sizes with the middle line since this will not change too much over this region
        # get instrumental broadening
        sig_inst_broad_vel = SpecHelper.get_inst_broad_sig(line=line_caii_2, instrument='muse', unit='kmps')
        sig_obs_vel = np.sqrt(sig_int_vel ** 2 + sig_inst_broad_vel ** 2)
        sig_obs_wave = SpecHelper.conv_vel2delta_wave(line=line_caii_2, vel=sig_obs_vel, vel_unit='kmps',
                                                     line_ref='vac_wave')

        mask_abs_caii_1 = ((ppxf_fit_dict['wave'] > line_pos_caii_1 - 3 * sig_obs_wave) &
                           (ppxf_fit_dict['wave'] < line_pos_caii_1 + 3 * sig_obs_wave))
        mask_abs_caii_2 = ((ppxf_fit_dict['wave'] > line_pos_caii_2 - 3 * sig_obs_wave) &
                           (ppxf_fit_dict['wave'] < line_pos_caii_2 + 3 * sig_obs_wave))
        mask_abs_caii_3 = ((ppxf_fit_dict['wave'] > line_pos_caii_3 - 3 * sig_obs_wave) &
                           (ppxf_fit_dict['wave'] < line_pos_caii_3 + 3 * sig_obs_wave))

        line_window_caii_1 = (line_pos_caii_1 - 3 * sig_obs_wave, line_pos_caii_1 + 3 * sig_obs_wave)
        line_window_caii_2 = (line_pos_caii_2 - 3 * sig_obs_wave, line_pos_caii_2 + 3 * sig_obs_wave)
        line_window_caii_3 = (line_pos_caii_3 - 3 * sig_obs_wave, line_pos_caii_3 + 3 * sig_obs_wave)

        continuum_window_caii = np.array([[line_pos_caii_1 - 5 * sig_obs_wave - 50, line_pos_caii_1 - 5 * sig_obs_wave],
                                          [line_pos_caii_1 + 5 * sig_obs_wave, line_pos_caii_2 - 5 * sig_obs_wave],
                                          [line_pos_caii_2 + 5 * sig_obs_wave, line_pos_caii_3 - 5 * sig_obs_wave],
                                          [line_pos_caii_3 + 5 * sig_obs_wave, line_pos_caii_3 + 5 * sig_obs_wave + 50]
                                          ])

        ew_dict_caii_1 = SpecHelper.compute_ew(wave=ppxf_fit_dict['wave'], flux=ppxf_fit_dict['total_flux'],
                                              flux_err=ppxf_fit_dict['total_flux_err'],
                                              line_window=line_window_caii_1,
                                              continuum_window=continuum_window_caii)

        ew_dict_caii_2 = SpecHelper.compute_ew(wave=ppxf_fit_dict['wave'], flux=ppxf_fit_dict['total_flux'],
                                              flux_err=ppxf_fit_dict['total_flux_err'],
                                              line_window=line_window_caii_2,
                                              continuum_window=continuum_window_caii)

        ew_dict_caii_3 = SpecHelper.compute_ew(wave=ppxf_fit_dict['wave'], flux=ppxf_fit_dict['total_flux'],
                                              flux_err=ppxf_fit_dict['total_flux_err'],
                                              line_window=line_window_caii_3,
                                              continuum_window=continuum_window_caii)

        # print('ew_caii_1, ew_err_caii_1 ', ew_dict_caii_1['ew'], ew_dict_caii_1['ew_err'])
        # print('ew_caii_2, ew_err_caii_2 ', ew_dict_caii_2['ew'], ew_dict_caii_2['ew_err'])
        # print('ew_caii_3, ew_err_caii_3 ', ew_dict_caii_3['ew'], ew_dict_caii_3['ew_err'])

        return_dict = {
            'ew_caii_1': ew_dict_caii_1['ew'],
            'ew_err_caii_1': ew_dict_caii_1['ew_err'],
            'ew_caii_2': ew_dict_caii_2['ew'],
            'ew_err_caii_2': ew_dict_caii_2['ew_err'],
            'ew_caii_3': ew_dict_caii_3['ew'],
            'ew_err_caii_3': ew_dict_caii_3['ew_err']
        }

        # # now get the absorption line position
        spec_part_mask = SpecHelper.get_multiple_line_mask(wave=ppxf_fit_dict['wave'],
                                                          ln_list=[line_caii_1, line_caii_2, line_caii_3],
                                                          vel_kmps=ppxf_fit_dict['sol_kin_comp'][0],
                                                          target=None, instrument='muse', blue_limit=70., red_limit=70.)

        mask_continuum = SpecHelper.wave_window2mask(wave=ppxf_fit_dict['wave'], wave_window=continuum_window_caii)

        # fit three gaussians to get the sizes

        if plot_results:
            fig, ax = plt.subplots(figsize=(20, 14))
            fontsize = 23

            flux_min = np.min(ppxf_fit_dict['total_flux'][spec_part_mask])
            flux_max = np.max(ppxf_fit_dict['total_flux'][spec_part_mask])

            ax.step(ppxf_fit_dict['wave'][spec_part_mask], ppxf_fit_dict['total_flux'][spec_part_mask], where='mid',
                    color='k')

            ax.plot([np.min(ppxf_fit_dict['wave'][spec_part_mask]), np.max(ppxf_fit_dict['wave'][spec_part_mask])],
                    [ew_dict_caii_1['mean_continuum'], ew_dict_caii_1['mean_continuum']], color='tab:blue')
            ax.plot([np.min(ppxf_fit_dict['wave'][spec_part_mask]), np.max(ppxf_fit_dict['wave'][spec_part_mask])],
                    [ew_dict_caii_2['mean_continuum'] - ew_dict_caii_2['std_continuum'],
                     ew_dict_caii_1['mean_continuum'] - ew_dict_caii_1['std_continuum']], color='tab:blue',
                    linestyle='--')
            ax.plot([np.min(ppxf_fit_dict['wave'][spec_part_mask]), np.max(ppxf_fit_dict['wave'][spec_part_mask])],
                    [ew_dict_caii_3['mean_continuum'] + ew_dict_caii_3['std_continuum'],
                     ew_dict_caii_1['mean_continuum'] + ew_dict_caii_1['std_continuum']], color='tab:blue',
                    linestyle='--')

            ax.scatter(ppxf_fit_dict['wave'][mask_continuum], ppxf_fit_dict['total_flux'][mask_continuum],
                       color='tab:red')

            ax.fill_between(ppxf_fit_dict['wave'][mask_abs_caii_1], ppxf_fit_dict['total_flux'][mask_abs_caii_1],
                            ew_dict_caii_1['mean_continuum'], color='gray', alpha=0.5)
            ax.fill_between(ppxf_fit_dict['wave'][mask_abs_caii_2], ppxf_fit_dict['total_flux'][mask_abs_caii_2],
                            ew_dict_caii_2['mean_continuum'], color='gray', alpha=0.5)
            ax.fill_between(ppxf_fit_dict['wave'][mask_abs_caii_3], ppxf_fit_dict['total_flux'][mask_abs_caii_3],
                            ew_dict_caii_3['mean_continuum'], color='gray', alpha=0.5)

            ax.set_ylim(flux_min - (flux_max - flux_min) * 0.2, flux_max + (flux_max - flux_min) * 0.1)

            ax.text(line_pos_caii_1,
                    np.min(ppxf_fit_dict['total_flux'][mask_abs_caii_1]) - (flux_max - flux_min) * 0.02,
                    phys_params.spec_line_dict[line_caii_1][
                        'plot_name'] + ' \n' + r'EW = %.2f $\pm$ %.2f ${\rm \AA}$' % (ew_dict_caii_1['ew'],
                                                                                      ew_dict_caii_1['ew_err']),
                    ha='center', va='top', fontsize=fontsize)

            ax.text(line_pos_caii_2,
                    np.min(ppxf_fit_dict['total_flux'][mask_abs_caii_2]) - (flux_max - flux_min) * 0.02,
                    phys_params.spec_line_dict[line_caii_2][
                        'plot_name'] + ' \n' + r'EW = %.2f $\pm$ %.2f ${\rm \AA}$' % (ew_dict_caii_2['ew'],
                                                                                      ew_dict_caii_2['ew_err']),
                    ha='center', va='top', fontsize=fontsize)

            ax.text(line_pos_caii_3,
                    np.min(ppxf_fit_dict['total_flux'][mask_abs_caii_3]) - (flux_max - flux_min) * 0.02,
                    phys_params.spec_line_dict[line_caii_3][
                        'plot_name'] + ' \n' + r'EW = %.2f $\pm$ %.2f ${\rm \AA}$' % (ew_dict_caii_3['ew'],
                                                                                      ew_dict_caii_3['ew_err']),
                    ha='center', va='top', fontsize=fontsize)

            ax.tick_params(axis='both', which='both', width=2, length=10, right=True, top=True, direction='in',
                           labelsize=fontsize)

            ax.set_xlabel(r'Wavelength [${\rm \AA}$]', fontsize=fontsize)
            ax.set_ylabel(r'$\phi$ [erg cm$^{-2}$ s$^{-1}$ ${\rm \AA^{-1}}$]', fontsize=fontsize)

            # plt.show()

            return return_dict, fig
        else:
            return return_dict, None

    @staticmethod
    def fit_complete_spec_old(spec_dict, target, sps_name='fsps', age_range=None, metal_range=None, ln_list=None,
                              n_nl_gauss=1, n_nl_lorentz=0, n_bl_gauss=0, search_outflow=False,
                              outflow_shift='redshift', outflow_mu_offset=400, outflow_sig=1200,
                              init_mu_nl_gauss=100, init_sig_nl_gauss=200):

        # first fit ppxf
        ppxf_dict = SpecHelper.fit_ppxf2spec(
            spec_dict=spec_dict, target=target, sps_name=sps_name, age_range=age_range, metal_range=metal_range,
            ln_list=ln_list, n_nl_gauss=n_nl_gauss, n_nl_lorentz=n_nl_lorentz, n_bl_gauss=n_bl_gauss,
            search_outflow=search_outflow, outflow_shift=outflow_shift, outflow_mu_offset=outflow_mu_offset,
            outflow_sig=outflow_sig, init_mu_nl_gauss=init_mu_nl_gauss, init_sig_nl_gauss=init_sig_nl_gauss)

        if ln_list is None:
            ln_list = [4863, 4960, 5008, 6302, 6550, 6565, 6585, 6718, 6733]
            # ln_list = [5008, 6550, 6565, 6585]

        # now fit the emission lines
        em_line_fit_dict = SpecHelper.fit_em_lines2spec(ln_list=ln_list, target=target, wave=wavelength,
                                                       em_flux=em_flux, em_flux_err=total_flux_err,
                                                       n_nl_gauss=n_nl_gauss, n_nl_lorentz=n_nl_lorentz,
                                                       n_bl_gauss=n_bl_gauss,
                                                       x_data_format='wave', instrument='muse', blue_limit=30.,
                                                       red_limit=30.,
                                                       search_outflow=search_outflow,
                                                       outflow_shift=outflow_shift, outflow_mu_offset=outflow_mu_offset,
                                                       outflow_sig=outflow_sig,
                                                       init_mu_nl_gauss=init_mu_nl_gauss,
                                                       init_sig_nl_gauss=init_sig_nl_gauss
                                                       )

        h_beta_rest_air = 4861.333
        h_alpha_rest_air = 6562.819
        ha_continuum_window_left_rest_air = (6475.0, 6540.0)
        ha_continuum_window_right_rest_air = (6595.0, 6625.0)

        balmer_redshift = np.exp(balmer_kin_comp[0] / speed_of_light_kmps) - 1

        observed_h_beta = h_beta_rest_air * (1 + balmer_redshift)
        observed_h_alpha = h_alpha_rest_air * (1 + balmer_redshift)

        observed_sigma_h_alpha = (balmer_kin_comp[1] / speed_of_light_kmps) * h_alpha_rest_air
        observed_sigma_h_alpha = np.sqrt(observed_sigma_h_alpha ** 2 + get_MUSE_polyFWHM(observed_h_alpha))
        observed_sigma_h_beta = (balmer_kin_comp[1] / speed_of_light_kmps) * h_beta_rest_air
        observed_sigma_h_beta = np.sqrt(observed_sigma_h_beta ** 2 + get_MUSE_polyFWHM(observed_h_beta))

        mask_ha = (wavelength > (observed_h_alpha - 3 * observed_sigma_h_alpha)) & (
                wavelength < (observed_h_alpha + 3 * observed_sigma_h_alpha))
        mask_hb = (wavelength > (observed_h_beta - 3 * observed_sigma_h_beta)) & (
                wavelength < (observed_h_beta + 3 * observed_sigma_h_beta))
        # get ha component
        ha_line_comp = (total_flux - continuum_best_fit)[mask_ha]
        ha_line_comp_err = total_flux_err[mask_ha]

        # get the continuum component as a constant
        mask_cont_region = (((wavelength > (ha_continuum_window_left_rest_air[0] * (1 + balmer_redshift))) &
                             (wavelength < (ha_continuum_window_left_rest_air[1] * (1 + balmer_redshift)))) |
                            ((wavelength > (ha_continuum_window_right_rest_air[0] * (1 + balmer_redshift))) &
                             (wavelength < (ha_continuum_window_right_rest_air[1] * (1 + balmer_redshift)))))

        ha_cont_comp = np.nanmean(continuum_best_fit[mask_cont_region])
        ha_cont_comp_std = np.nanstd(continuum_best_fit[mask_cont_region])

        # since the wavelength size does not change much, we take the mean value
        ha_wave_comp = wavelength[mask_ha]
        delta_lambda_ha = np.mean((ha_wave_comp[1:] - ha_wave_comp[:-1]) / 2)

        # calculate the EW
        ha_ew = np.sum(((ha_cont_comp - ha_line_comp) / ha_cont_comp) * delta_lambda_ha)

        # uncertainty
        sigma_ew_segment = np.sqrt(((delta_lambda_ha * ha_line_comp_err) / ha_cont_comp) ** 2 +
                                   ((ha_line_comp * delta_lambda_ha * ha_cont_comp_std) / (ha_cont_comp ** 2)) ** 2)
        ha_ew_err = np.sqrt(np.sum(sigma_ew_segment ** 2))

        hb_line_comp = (total_flux - continuum_best_fit)[mask_hb]
        hb_cont_comp = continuum_best_fit[mask_hb]
        hb_wave_comp = wavelength[mask_hb]
        delta_lambda_hb = np.mean((hb_wave_comp[1:] - hb_wave_comp[:-1]) / 2)
        hb_ew = np.sum(((hb_cont_comp - hb_line_comp) / hb_cont_comp) * delta_lambda_hb)

        # gas_phase_metallicity
        flux_ha = pp.gas_flux[pp.gas_names == 'Halpha']
        flux_hb = pp.gas_flux[pp.gas_names == 'Hbeta']
        flux_nii = pp.gas_flux[pp.gas_names == '[OIII]5007_d']
        flux_oiii = pp.gas_flux[pp.gas_names == '[NII]6583_d']

        # pp.plot()

        o3n2 = np.log10((flux_oiii / flux_hb) / (flux_nii / flux_ha))
        gas_phase_met = 8.73 - 0.32 * o3n2[0]
        # plt.plot(hb_wave_comp, hb_line_comp)
        # plt.plot(hb_wave_comp, hb_cont_comp)
        # plt.show()
        # exit()
        #
        # # exit()
        # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
        # plt.plot(wavelength, continuum_best_fit)
        # plt.scatter(wavelength[left_idx_ha[0][0]], continuum_best_fit[left_idx_ha[0][0]])
        # plt.scatter(wavelength[right_idx_ha[0][0]], continuum_best_fit[right_idx_ha[0][0]])
        # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
        # plt.plot(wavelength, gas_best_fit)
        # plt.plot([observed_nii_1, observed_nii_1], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_h_alpha, observed_h_alpha], [np.min(total_flux), np.max(total_flux)])
        # plt.plot([observed_nii_2, observed_nii_2], [np.min(total_flux), np.max(total_flux)])
        # plt.show()
        #
        # plt.figure(figsize=(17, 6))
        # plt.subplot(111)
        # pp.plot()
        # plt.show()

        ppxf_dict = {
            'wavelength': wavelength, 'total_flux': total_flux, 'total_flux_err': total_flux_err,
            'best_fit': best_fit, 'gas_best_fit': gas_best_fit, 'continuum_best_fit': continuum_best_fit,
            'ages': ages, 'met': met, 'mass2light': mass2light,
            'pp': pp,
            'star_red': pp.dust[0]['sol'][0], 'gas_red': pp.dust[1]['sol'][0],
            'sol_kin_comp': sol_kin_comp, 'balmer_kin_comp': balmer_kin_comp, 'forbidden_kin_comp': forbidden_kin_comp,
            'ha_ew': ha_ew, 'ha_ew_err': ha_ew_err, 'hb_ew': hb_ew, 'gas_phase_met': gas_phase_met,
            'sys_vel': sys_vel, 'redshift': redshift
        }
        return ppxf_dict, em_line_fit_dict

#####################
##### Code dump #####
#####################
#
# @staticmethod
# def compute_ha_ew(ppxf_dict):
#
#     # print(ppxf_dict)
#
#     balmer_redshift = np.exp(ppxf_dict['balmer_kin_comp'][0] / speed_of_light_kmps) - 1
#
#
#     # calculate line window:
#     ha_continuum_window_left_rest_air = (6475.0, 6540.0)
#     ha_continuum_window_right_rest_air = (6595.0, 6625.0)
#
#     continuum_window = np.array([
#         [ha_continuum_window_left_rest_air[0] * (1 + balmer_redshift),
#          ha_continuum_window_left_rest_air[1] * (1 + balmer_redshift)],
#         [ha_continuum_window_right_rest_air[0] * (1 + balmer_redshift),
#          ha_continuum_window_right_rest_air[1] * (1 + balmer_redshift)]
#     ])
#
#     h_alpha_rest_air = 6562.819
#     observed_h_alpha = h_alpha_rest_air * (1 + balmer_redshift)
#
#     observed_sigma_h_alpha = (ppxf_dict['balmer_kin_comp'][1] / speed_of_light_kmps) * h_alpha_rest_air
#     observed_sigma_h_alpha = np.sqrt(observed_sigma_h_alpha ** 2 + get_MUSE_polyFWHM(observed_h_alpha))
#
#     line_window = np.array([observed_h_alpha - 3 * observed_sigma_h_alpha,
#                             observed_h_alpha + 3 * observed_sigma_h_alpha])
#
#     ew_ha, ew_err_ha = SpecHelper.compute_ew(wave=ppxf_dict['wavelength'], flux=ppxf_dict['total_flux'],
#                          flux_err=ppxf_dict['total_flux_err'],
#                          line_window=line_window, continuum_window=continuum_window)
#     print('ew_ha, ew_err_ha ', ew_ha, ew_err_ha)
#
#
#
#
#     exit()


# started to develop an alternative fit for the Tardis pipeline
#
# def fit_tardis2spec(spec_dict, velocity, hdr, sps_name='fsps', age_range=None, metal_range=None, name='explore1'):
#     """
#
#     Parameters
#     ----------
#     spec_dict : dict
#     sps_name : str
#         can be fsps, galaxev or emiles
#
#
#
#     Returns
#     -------
#     dict
#     """
#     from os import path
#     # import ppxf.sps_util as lib
#     # from urllib import request
#     # from ppxf.ppxf import ppxf
#
#     import matplotlib.pyplot as plt
#
#     from TardisPipeline.utilities import util_ppxf, util_ppxf_stellarpops, util_sfh_quantities, util_ppxf_emlines
#     import TardisPipeline as tardis_module
#     codedir = os.path.dirname(os.path.realpath(tardis_module.__file__))
#
#     import ppxf.ppxf_util as util
#     from astropy.io import fits, ascii
#     from astropy import constants as const
#     from astropy.table import Table
#     import extinction
#
#     # tardis_path = '/home/egorov/Soft/ifu-pipeline/TardisPipeline/' # change to directory where you have installed DAP
#     ncpu = 20  # how many cpu would you like to use? (20-30 is fine for our server, but use no more than 8 for laptop)
#     # print(codedir+'/Templates/spectralTemplates/eMILES-noyoung/')
#     # exit()
#     configs = {  #'SSP_LIB': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         #'SSP_LIB_SFH': os.path.join(codedir, 'Templates/spectralTemplates/eMILES-noyoung/'),
#         'SSP_LIB': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         'SSP_LIB_SFH': codedir + '/Templates/spectralTemplates/CB07_chabrier-young-selection-MetalPoorRemoved/',
#         # stellar library to use
#         # 'SSP_LIB': codedir+'/Templates/spectralTemplates/eMILES-noyoung/',  # stellar library to use
#         'NORM_TEMP': 'LIGHT', 'REDSHIFT': velocity, 'MOM': 4, 'MC_PPXF': 0, 'PARALLEL': 1,
#         'ADEG': 12,
#         'ADEG_SFH': 12,
#         'MDEG': 0,
#         'MDEG_SFH': 0,
#         'MDEG_EMS': 24,
#         'NCPU': ncpu,
#         'ROOTNAME': name,
#         'SPECTRUM_SIZE': abs(hdr['CD1_1']) * 3600.,  # spaxel size in arcsec
#         # 'EMI_FILE': os.path.join(codedir, '/Templates/configurationTemplates/emission_lines.setup'),
#         'MC_PPXF_SFH': 10,
#         'EMI_FILE': codedir + '/Templates/configurationTemplates/emission_lines.setup',  # set of emission lines to fit
#         'SKY_LINES_RANGES': codedir + '/Templates/configurationTemplates/sky_lines_ranges.setup',
#         'OUTDIR': 'data_output/',
#         'MASK_WIDTH': 150,
#         'GAS_MOMENTS': 4}
#
#     velscale = speed_of_light_kmps * np.diff(np.log(spec_dict['lam'][-2:]))[0]  # Smallest velocity step
#     log_spec, logLam, velscale = util.log_rebin(lam=spec_dict['lam_range'], spec=spec_dict['spec_flux'],
#                                                 velscale=velscale)
#     c1 = fits.Column(name='LOGLAM', array=logLam, format='D')
#     c2 = fits.Column(name='LOGSPEC', array=log_spec, format='D')
#     t = fits.BinTableHDU.from_columns([c1, c2])
#     t.writeto('{}{}-ppxf_obsspec.fits'.format(configs['OUTDIR'], name), overwrite=True)
#     log_err, _, _ = util.log_rebin(spec_dict['lam_range'], spec_dict['spec_flux_err'], velscale=velscale)
#     ww = ~np.isfinite(log_spec) | ~np.isfinite(log_err) | (log_err <= 0)
#     log_err[ww] = 9999
#     log_spec[ww] = 0.
#     # # the DAP fitting routines expect log_spec and log_err to be 2D arrays containing N spectra,
#     # # here we add a dummy dimension since we are fitting only one spectrum
#     # # to fit more than one spectrum at the same time these lines can be easily adapted
#     log_err = np.expand_dims(log_err, axis=1)
#     log_spec = np.expand_dims(log_spec, axis=1)
#
#     # define the LSF of the MUSE data
#     LSF = get_MUSE_polyFWHM(np.exp(logLam), version="udf10")
#
#     # define the velocity scale in kms
#     velscale = (logLam[1] - logLam[0]) * speed_of_light_kmps
#
#     # this is the stellar kinematics ppxf wrapper function
#     ppxf_result = util_ppxf.runModule_PPXF(configs=configs,  #tasks='',
#                                            logLam=logLam,
#                                            log_spec=log_spec, log_error=log_err,
#                                            LSF=LSF)  #, velscale=velscale)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     # exit()
#     util_ppxf_stellarpops.runModule_PPXF_stellarpops(configs, logLam, log_spec, log_err, LSF, np.arange(1), ppxf_result)
#     masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#         configs)
#     print(masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err)
#
#     # read the output file which contains the best-fit from the emission lines fitting stage
#     ppxf_bestfit_gas = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit-emlines.fits'
#     hdu3 = fits.open(ppxf_bestfit_gas)
#     bestfit_gas = hdu3['FIT'].data["BESTFIT"][0]
#     mask = (hdu3['FIT'].data['BESTFIT'][0] == 0)
#     gas_templ = hdu3['FIT'].data["GAS_BESTFIT"][0]
#
#     ppxf_bestfit = configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf-bestfit.fits'
#     hdu_best_fit = fits.open(ppxf_bestfit)
#     cont_fit = hdu_best_fit['FIT'].data["BESTFIT"][0]
#
#     # # reddening = ppxf_sfh_data['REDDENING']
#     # hdu_best_fit_sfh = fits.open('data_output/explore1_ppxf-bestfit.fits')
#     # print(hdu_best_fit_sfh.info())
#     # print(hdu_best_fit_sfh[1].data.names)
#     #
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'])
#     # print(hdu_best_fit_sfh['FIT'].data['BESTFIT'].shape)
#     # print(logLam.shape)
#     # print(spec_dict['lam'].shape)
#     # # exit()
#     # # hdu_best_fit = fits.open('data_output/explore1_templates_SFH_info.fits')
#     # # print(hdu_best_fit.info())
#     # # print(hdu_best_fit[1].data.names)
#     # # print(hdu_best_fit[1].data['Age'])
#
#     plt.plot(spec_dict['lam'], spec_dict['spec_flux'])
#     plt.plot(np.exp(logLam), cont_fit)
#     plt.plot(np.exp(logLam), gas_templ)
#     plt.plot(np.exp(logLam), cont_fit + gas_templ)
#     plt.show()
#
#     exit()
#     # this the ppxf wrapper function to simulataneously fit the continuum plus emission lines
#     # util_ppxf_emlines.runModule_PPXF_emlines(configs,# '',
#     #                                          logLam, log_spec,
#     #                                          log_err, LSF, #velscale,
#     #                                          np.arange(1), ppxf_result)
#     util_ppxf_emlines.runModule_PPXF_emlines(configs=configs,  #tasks='',
#                                              logLam=logLam,
#                                              log_spec=log_spec, log_error=log_err,
#                                              LSF=LSF, ppxf_results=ppxf_result)
#
#     emlines = configs['OUTDIR'] + configs['ROOTNAME'] + '_emlines.fits'
#     with fits.open(emlines) as hdu_emis:
#         ems = Table(hdu_emis['EMLDATA_DATA'].data)
#
#     # This is to include SFH results, NOT TESTED!
#     with fits.open(configs['OUTDIR'] + configs['ROOTNAME'] + '_ppxf_SFH.fits') as hdu_ppxf_sfh:
#         ppxf_sfh_data = hdu_ppxf_sfh[1].data
#         masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err = util_sfh_quantities.compute_sfh_relevant_quantities(
#             configs)
#         reddening = ppxf_sfh_data['REDDENING']
#         st_props = masses_density, mass_density_err, ages_mw, ages_mw_err, z_mw, z_mw_err, ages_lw, ages_lw_err, z_lw, z_lw_err, reddening
#
#     exit()
#
#     return ems, st_props
#
#     spectra_muse_err, ln_lam_gal, velscale = util.log_rebin(lam=spec_dict['lam_range'],
#                                                             spec=spec_dict['spec_flux_err'], velscale=velscale)
#
#     # print(sum(np.isnan(spec_dict['spec_flux'])))
#     # print(sum(np.isnan(spectra_muse)))
#     #
#     # plt.plot(ln_lam_gal, spectra_muse_err)
#     # plt.show()
#
#     lsf_dict = {"lam": spec_dict['lam'], "fwhm": spec_dict['lsf']}
#     # get new wavelength array
#     lam_gal = np.exp(ln_lam_gal)
#     # goodpixels = util.determine_goodpixels(ln_lam=ln_lam_gal, lam_range_temp=spec_dict['lam_range'], z=redshift)
#     goodpixels = None
#     # goodpixels = (np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))
#     # print(sum(np.invert(np.isnan(spectra_muse) + np.isnan(spectra_muse_err) + np.isinf(spectra_muse) + np.isinf(spectra_muse_err))))
#     # print(sum(((spectra_muse > 0) & (spectra_muse < 100000000000000))))
#
#     # get stellar library
#     ppxf_dir = path.dirname(path.realpath(lib.__file__))
#     basename = f"spectra_{sps_name}_9.0.npz"
#     filename = path.join(ppxf_dir, 'sps_models', basename)
#     if not path.isfile(filename):
#         url = "https://raw.githubusercontent.com/micappe/ppxf_data/main/" + basename
#         request.urlretrieve(url, filename)
#
#     sps = lib.sps_lib(filename=filename, velscale=velscale, fwhm_gal=lsf_dict, norm_range=[5070, 5950],
#                       wave_range=None,
#                       age_range=age_range, metal_range=metal_range)
#     reg_dim = sps.templates.shape[1:]  # shape of (n_ages, n_metal)
#     stars_templates = sps.templates.reshape(sps.templates.shape[0], -1)
#
#     gas_templates, gas_names, line_wave = util.emission_lines(ln_lam_temp=sps.ln_lam_temp,
#                                                               lam_range_gal=spec_dict['lam_range'],
#                                                               FWHM_gal=get_MUSE_polyFWHM)
#
#     templates = np.column_stack([stars_templates, gas_templates])
#
#     n_star_temps = stars_templates.shape[1]
#     component = [0] * n_star_temps
#     for line_name in gas_names:
#         if '[' in line_name:
#             component += [2]
#         else:
#             component += [1]
#
#     gas_component = np.array(component) > 0  # gas_component=True for gas templates
#
#     moments = [4, 4, 4]
#
#     vel = speed_of_light_kmps * np.log(1 + redshift)  # eq.(8) of Cappellari (2017) 2017MNRAS.466..798C
#     start_gas = [vel, 150., 0, 0]  # starting guess
#     start_star = [vel, 150., 0, 0]
#     print(start_gas)
#     start = [start_star, start_gas, start_gas]
#
#     pp = ppxf(templates=templates, galaxy=spectra_muse, noise=spectra_muse_err, velscale=velscale, start=start,
#               moments=moments, degree=-1, mdegree=4, lam=lam_gal, lam_temp=sps.lam_temp,  #regul=1/rms,
#               reg_dim=reg_dim, component=component, gas_component=gas_component,  #reddening=0,
#               gas_names=gas_names, goodpixels=goodpixels)
#
#     light_weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
#     light_weights = light_weights.reshape(reg_dim)  # Reshape to (n_ages, n_metal)
#     light_weights /= light_weights.sum()  # Normalize to light fractions
#
#     # light_weights = pp.weights[~gas_component]      # Exclude weights of the gas templates
#     # light_weights = light_weights.reshape(reg_dim)
#
#     ages, met = sps.mean_age_metal(light_weights)
#     mass2light = sps.mass_to_light(light_weights, redshift=redshift)
#
#     return {'pp': pp, 'ages': ages, 'met': met, 'mass2light': mass2light}
#
#     # wavelength = pp.lam
#     # total_flux = pp.galaxy
#     # total_flux_err = pp.noise
#     #
#     # best_fit = pp.bestfit
#     # gas_best_fit = pp.gas_bestfit
#     # continuum_best_fit = best_fit - gas_best_fit
#     #
#     # plt.errorbar(wavelength, total_flux, yerr=total_flux_err)
#     # plt.plot(wavelength, continuum_best_fit + gas_best_fit)
#     # plt.plot(wavelength, gas_best_fit)
#     # plt.show()
#     #
#     #
#     #
#     #
#     # plt.figure(figsize=(17, 6))
#     # plt.subplot(111)
#     # pp.plot()
#     # plt.show()
#     #
#     # plt.figure(figsize=(9, 3))
#     # sps.plot(light_weights)
#     # plt.title("Light Weights Fractions");
#     # plt.show()
#     #
#     # exit()
