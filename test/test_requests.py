import unittest
import requests
import argparse
import sys
# parser = argparse.ArgumentParser(description='Flag for test')
# parser.add_argument('--id', type=int, help='Number ID client', default=100)
# args = parser.parse_args()

def create_parser():
    parser = argparse.ArgumentParser(description='Flag for test')
    parser.add_argument('--id', type=int, help='Number ID client', default=100)
    return parser

class TestMethods(unittest.TestCase):
    
    def setUp(self):
        self.parser = create_parser()
    
    def test_request(self):
        
        args = self.parser.parse_args(['--id', '100' ])
        URL_BEGIN_DATA = 'http://192.168.42.113:8000/predict'
        req = requests.post(URL_BEGIN_DATA, json={"id_cl": args.id})
        self.assertTrue(req.json())
if __name__ == '__main__':
    unittest.main()
