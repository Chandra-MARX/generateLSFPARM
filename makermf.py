import re

import numpy as np
from astropy.table import Table
import astropy.units as u
from sherpa.models import NormGauss1D, Scale1D
from sherpa.astro.models import Lorentz1D

import re

modelnames = re.compile('(GAUSS|LORENTZ)[0-9]+_PARMS')
gaussn = re.compile('GAUSS(?P<n>[0-9]+)_PARMS')
lorentzn = re.compile('LORENTZ(?P<n>[0-9]+)_PARMS')

# convert parameters from CALDB convention to Sherpa
# For NormGauss1d Sherpa uses FWHM, while CALDB uses sigma
gconv = np.array([1, 2 * np.sqrt(2 * np.log(2)), 1])
lconv = np.array([1, 1, 1])

def caldb2sherpa(caldb, row, iwidth, iwave):
    '''Convert CALDB entries to Sherpa model

    Convert entries in a CALDB lsparm files to a Sherpa model.  The
    lsfparm files can contain several rows, for different off-axis
    angles and radii and in each row there will be entries for a
    number of wavelength points and extraction width.

    This function expects as input the index numbers for row,
    wavelength as index for the wavelength array etc. In practical
    applications, the CALDB file will be queried for a specific
    position, wavelength etc., but for development purposes it is
    useful to go into the raw array, e.g. to read some unrelated CALDB
    file (say for a different detector) to use as a starting point to
    fit the lsfparm parameters or to plot different settings for
    comparison.

    caldb : `astropy.table.Table`
        CALDB lsfparm table

    '''
    model = []
    for col in caldb.colnames:
        if col == 'EE_FRACS':
            eef = Scale1D(name='EE_FRACS')
            eef.c0 = caldb['EE_FRACS'][row][iwidth, iwave]
            # model is underdetermined if this is free and the ampl of all functions
            eef.c0.frozen = True
        elif gaussn.match(col):
            newg = NormGauss1D(name=col)
            newg.ampl, newg.fwhm, newg.pos = caldb[col][row][iwidth, iwave, :] * gconv
            model.append(newg)
        elif lorentzn.match(col):
            newg = Lorentz1D(name=col)
            newg.ampl, newg.fwhm, newg.pos = caldb[col][row][iwidth, iwave, :] * lconv
            model.append(newg)

    sumampl = np.sum([m.ampl.val for m in model if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D)])
    for m in model:
        if isinstance(m, NormGauss1D) or isinstance(m, Lorentz1D):
            m.ampl.val = m.ampl.val / sumampl
    # Start value is 0, unless we explicitly set the start value. So, split models and pass [0]
    # as start value to avoid a numerical 0.0 in the model expression.
    return eef * sum(model[1:], model[0])


def flatten_sherpa_model(model):
    if hasattr(model, 'parts'):
        modellist = []
        for p in model.parts:
            modellist.extend(flatten_sherpa_model(p))
        return modellist
    else:
        return [model]

def sherpa2caldb(shmodel):
    '''
    shmodel : Sherpa model instance
    '''
    d = {}
    for comp in flatten_sherpa_model(shmodel):
        if isinstance(comp, NormGauss1D):
            modelpars = np.array([comp.ampl.val, comp.fwhm.val, comp.pos.val]) / gconv
        elif isinstance(comp, Lorentz1D):
            modelpars = np.array([comp.ampl.val, comp.fwhm.val, comp.pos.val]) / lconv
        else:
            raise Exception('Component {} not valid for LSFPARM files'.format(comp))
        d[comp.name] = modelpars
    return d


class RMF:
    '''Read and represent OGIP RMF data

    Sherpa has a similar class (in fact with more properties)
    I just want to make sure I understand everything that goes into it,
    so I make that myself here.

    Parameters
    ----------
    rmffile : filename
    '''
    def __init__(self, rmffile):
        self.rmfmatrix = Table.read(rmffile, format='fits', hdu='MATRIX')
        self.rmfebounds = Table.read(rmffile, format='fits', hdu='EBOUNDS')

    def row(self, energy):
        rowind = (energy > self.rmfmatrix['ENERG_LO']) & (energy < self.rmfmatrix['ENERG_HI'])
        if not rowind.sum() == 1:
            raise ValueError(f'Energy {energy} does not correspond to a unique row in RMF.')
        return self.rmfmatrix[rowind.nonzero()[0][0]]

    @property
    def en_mid(self):
        return 0.5 * (self.rmfebounds['E_MIN'] + self.rmfebounds['E_MAX']) # * self.rmfebounds['E_MIN'].unit

    def full_rmf(self, energy):
        '''Return rmf over the full range of channels

        Parameters
        ----------
        energy : `astropy.quantity.Quantity`

        Returns
        -------
        e_mid : array
            mid-points of energy bins in keV
        rmf : array
            rmf value in each bin
        '''
        rmf = np.zeros(len(self.en_mid))
        row = self.row(energy)
        for i in range(row['N_GRP']):
            # Python is 0 indexed, FITS is 1 indexed
            chans = slice(row['F_CHAN'][i] - 1, row['F_CHAN'][i] + row['N_CHAN'][i] - 1)
            matindex = np.cumsum(np.concatenate(([0], row['N_CHAN'])))
            mat = slice(matindex[i], matindex[i + 1])
            rmf[chans] = row['MATRIX'][mat]
        return self.rmfebounds['E_MIN'], self.rmfebounds['E_MAX'], rmf


    def rmf(self, energy):
        '''Return rmf over channels where values is non-zero

        Parameters
        ----------
        energy : `astropy.quantity.Quantity`

        Returns
        -------
        e_mid : array
            mid-points of energy bins in keV
        rmf : array
            rmf value in each bin
        '''
        row = self.row(energy)
        ind = np.concatenate([np.arange(row['F_CHAN'][i] - 1,
                                        row['F_CHAN'][i] + row['N_CHAN'][i]-1)
                              for i in range(row['N_GRP'])])
        return self.rmfebounds['E_MIN'][ind], self.rmfebounds['E_MAX'][ind], row['MATRIX']

    def rmf_ang(self, energy):
        en_lo, en_hi, rmf = self.rmf(energy)
        return en_hi.to(u.Angstrom, equivalencies=u.spectral()), en_lo.to(u.Angstrom, equivalencies=u.spectral()), rmf
