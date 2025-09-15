import unittest
import serial

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def set_devices(self):
        self.listener = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        self.talker = serial.Serial('/dev/ttyACM1', 115200, timeout=1)

    def test_decoder(self):



        while True:
if __name__ == '__main__':
    unittest.main()
