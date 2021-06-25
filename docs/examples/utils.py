"""
Tools for loading / downloading the example data files
"""
import os
import urllib.request


def normalize(x):
    return (x - x.min())/(x.max() - x.min())


def _download_aeronet(output_file):
    # Request daily 2017 AAOD for all stations
    target_url = "https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_inv_v3?year=2017&month=1&day=1&year2=2018&month2=1&day2=1&product=TAB&AVG=20&HYB15=1&if_no_html=1"
    response = urllib.request.urlopen(target_url)
    # Discard the first line which isn't standard AeroNet
    _ = next(response)
    with open(output_file, 'w') as f:
        for line in response:
            f.write(line.decode('utf-8'))


def get_aeronet_data(cache_path='.'):
    import cis
    from cis.data_io.aeronet import AERONET_HEADER_LENGTH

    aerocom_cache = os.path.join(cache_path, 'aerocom_2017_global.lev15')

    if not os.path.isfile(aerocom_cache):
        _download_aeronet(aerocom_cache)

    # Fix the header length which seems to be different for downloaded files...
    AERONET_HEADER_LENGTH["AERONET/3"] = 6

    # TODO: I might want to interpolate between 440 and 630nm to get 550nm...
    aaod = cis.read_data(aerocom_cache, 'Absorption_AOD440nm')
    # Pop off the altitude coordinate which we're not interested in
    _ = aaod._coords.pop(aaod._coords.index(aaod.coord('altitude')))
    return aaod


def get_bc_ppe_data(cache_path='.', dre=False, normalize_params=True):
    """
    Load the example BC PPE AAOD data and parameters, downloading if not present

    :param str cache_path: Path to load/store the PPE data
    :param bool dre: Output the Direct Radiative Effect data also
    :param bool normalize_params: Normalize the PPE parameters between 0.-1.
    :return:
    """
    import pandas as pd
    import numpy as np
    import cis

    bc_ppe_cache = os.path.join(cache_path, 'BC_PPE_PD_AAOD_monthly.nc')

    if not os.path.isfile(bc_ppe_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/3856645/files/BC_PPE_PD_AAOD_monthly.nc?download=1", bc_ppe_cache)

    params_cache = os.path.join(cache_path, 'aerocommmppe_bcdesign.csv')
    if not os.path.isfile(params_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/3856645/files/aerocommmppe_bcdesign.csv?download=1", params_cache)

    ppe_params = pd.read_csv(params_cache)
    ppe_aaod = cis.read_data(bc_ppe_cache, 'ABS_2D_550nm', 'NetCDF_Gridded')
    # Ensure the job dimension is at the front
    ppe_aaod.transpose((1, 0, 2, 3))
    
    if normalize_params:
        # These scaling parameters are log-uniformly distributed
        ppe_params[['BCnumber', 'Wetdep']] = np.log(ppe_params[['BCnumber', 'Wetdep']])
        ppe_params = ppe_params.apply(normalize, axis=0)
    
    if dre:
        ppe_dre_cache = os.path.join(cache_path, 'BC_PPE_PD_FORCING_monthly.nc')
        if not os.path.isfile(ppe_dre_cache):
            urllib.request.urlretrieve("https://zenodo.org/record/3856645/files/BC_PPE_PD_FORCING_monthly.nc?download=1", ppe_dre_cache)

        ppe_dre = cis.read_data(ppe_dre_cache, 'FSW_TOTAL_TOP', 'NetCDF_Gridded')        
        return ppe_params, ppe_aaod, ppe_dre
    else:
        return ppe_params, ppe_aaod

def get_crm_data(cache_path='.', preprocess=True):
    """
    Load the example cloud-resolving model data, download if not present.
    :param str cache_path: Path to load/store the data
    :param bool preprocess: Whether or not to clean and concatenate the data
    :return:
    """
    import pandas as pd
    import os
    import urllib.request

    N1_200_cache = os.path.join(cache_path, 'NARVAL1_1hr_200cdnc.csv')
    if not os.path.isfile(N1_200_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_200cdnc.csv?download=1", N1_200_cache)
        
        
    N1_20_cache = os.path.join(cache_path, 'NARVAL1_1hr_20cdnc.csv')
    if not os.path.isfile(N1_20_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_20cdnc.csv?download=1", N1_20_cache)
        
    N1_20_shal_cache = os.path.join(cache_path, 'NARVAL1_1hr_20cdnc_shal.csv')
    if not os.path.isfile(N1_20_shal_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_20cdnc_shal.csv?download=1", N1_20_shal_cache)
        
        
    N1_200_shal_cache = os.path.join(cache_path, 'NARVAL1_1hr_200cdnc_shal.csv')
    if not os.path.isfile(N1_200_shal_cache):
        urllib.request.urlretrieve("https://zenodo.org/record/4323300/files/NARVAL1_1hr_200cdnc_shal.csv?download=1", N1_200_shal_cache)

    
    if preprocess:
        df20 = pd.read_csv(N1_20_shal_cache).set_index('time').drop(columns='plev')
        df200 = pd.read_csv(N1_200_shal_cache).set_index('time').drop(columns='plev')

        new_df = pd.concat([df20, df200]).reset_index().drop(columns='time')

        return new_df
    
    else:
        df20 = pd.read_csv('NARVAL1_1hr_20cdnc_shal.csv')
        df200 = pd.read_csv('NARVAL1_1hr_200cdnc_shal.csv')
        return df20, df200
    