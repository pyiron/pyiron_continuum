import unittest
from pyiron_continuum import Project
import os


class TestDamask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.file_location = os.path.dirname(os.path.abspath(__file__))
        cls.project = Project("FENICSX")
        cls.project.remove_jobs(recursive=True, silently=True)

    @classmethod
    def tearDownClass(cls):
        cls.project.remove(enable=True)

    def test_load(self):
        _ = self.project.create.job.Fenicsx("test")



if __name__ == "__main__":
    unittest.main()
