
import unittest
from helper import *

class TmpTest(unittest.TestCase):

    def test_imports(self):
      print(f"proj_dir={get_project_dir()}")

if __name__ == '__main__':
    unittest.main()
