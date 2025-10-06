import unittest
from ESP_Class import ESP_wifi_module


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_read_folder(self):
        esp = ESP_wifi_module()
        folder_path = "./data/udp_data_2025-09-18_11-13-41"
        data = esp.read_folder(folder_path)
        print(data)

if __name__ == '__main__':
    unittest.main()
