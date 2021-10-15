__version__ = "0.1"
__all__ = []


from pyiron_base import JOB_CLASS_DICT
from pyiron_continuum.project import Project
from pyiron_continuum.mesh import RectMesh
from pyiron_continuum.schroedinger.potentials import Potential

from pyiron_base import Project as ProjectBase
from pyiron_continuum.toolkit import ContinuumTools
ProjectBase.register_tools('continuum', ContinuumTools)

# Make classes available for new pyiron version
JOB_CLASS_DICT['Fenics'] = 'pyiron_continuum.fenics.job.generic'
JOB_CLASS_DICT['FenicsLinearElastic'] = 'pyiron_continuum.fenics.job.elastic'
JOB_CLASS_DICT['DAMASK'] = 'pyiron_continuum.damask.damaskjob'
JOB_CLASS_DICT['TISE'] = 'pyiron_continuum.schroedinger.schroedinger'

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
