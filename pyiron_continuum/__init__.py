__version__ = "0.1"
__all__ = []


from pyiron_base import Project, JOB_CLASS_DICT

# Make classes available for new pyiron version
JOB_CLASS_DICT['Fenics'] = 'pyiron_continuum.fenics.job.generic'
JOB_CLASS_DICT['FenicsLinearElastic'] = 'pyiron_continuum.fenics.job.elastic'
JOB_CLASS_DICT['DAMASK'] = 'pyiron_continuum.damask.damaskjob'

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
