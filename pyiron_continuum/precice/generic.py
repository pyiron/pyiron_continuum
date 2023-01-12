# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for coupling multiple simulation processes via precice (precice.org) .
"""


from pyiron_base import ImportAlarm
from pyiron_base.master.generic import GenericMaster
with ImportAlarm(
"precice coupling requires installation of precice python packages and adaptors"
) as precice_alarm:
    import precice
    import pyprecice

from multiprocessing import Process

__author__ = "Muhammad Hassani"
__copyright__ = (
    "Copyright 2022, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Muhammad Hassani"
__email__ = "hassani@mpie.de"
__status__ = "development"
__date__ = "Feb 19, 2022"


class Precice(GenericMaster):
    def __init__(self, Project, job_name):
        super(Precice, self).__init__(Project, job_name)
        self.child_list = []

    def sync_child_jobs(self):
        for job in self.child_list:
            self.child_ids.append(job.id)

    def run_static(self):

        self.sync_child_jobs()
        processes = []
        for job in self.child_list:
            p = Process(target=job.run)
            p.start()
            processes.append(p)

        for process in processes:
            process.join()

        for process in processes:
            process.close()
        self.status.finished = True

    def write_input(self):
        pass
