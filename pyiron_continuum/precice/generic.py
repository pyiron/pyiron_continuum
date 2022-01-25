from pyiron_base import ImportAlarm
from pyiron_base.master.generic import GenericMaster
with ImportAlarm(
"precice coupling requires installation of precice python packages and adaptors"
) as precice_alarm:
    import precice
    import pyprecice
from multiprocessing import Process


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
