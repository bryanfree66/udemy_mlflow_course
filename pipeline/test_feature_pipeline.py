import unittest

class TestingClass(unittest.TestCase):
    def test_first(self):    
        test_var = 9 + 1
        self.assertEqual(11,test_var)

unittest.main()