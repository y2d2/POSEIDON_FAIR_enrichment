import socket
import os
import csv
from datetime import datetime


class ESP_wifi_module:
    def __init__(self):
        pass

    def read_folder(self, folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))

        data = []

        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    data.append(row[:-1])
        return data



    def udp_listener(self, lines_per_file=10000, port=5005, folder_path="./data/udp_data", debug=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", port))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        FOLDER_PATH = f"{folder_path}_{timestamp}"  # e.g. udp_data_2025-09-15_14-32-10
        os.makedirs(FOLDER_PATH, exist_ok=True)

        file_index = 1
        line_count = 0
        current_file = open(os.path.join(FOLDER_PATH, f"udp_data_{file_index}.csv"), "w", newline="")
        csv_writer = csv.writer(current_file)
        csv_writer.writerow(["id", "time", "magx", "magy", "magz", "T",
                             "gx", "gy", "gz", "ax", "ay", "az",
                             "valid_ranges", "rid", "dist  m", "fp_rssi", "rx_rssi", "..."])
        try:
            while True:
                data, addr = sock.recvfrom(1024)
                decoded = data.decode(errors="replace")  # your string like "A;B;C;D"

                # split the string by ';' into columns
                row = decoded.split(";")

                csv_writer.writerow(row)
                line_count += 1
                if debug:
                    print(f"{file_index}, {line_count}: {row}")
                # Check if we reached the line limit
                if line_count >= lines_per_file:
                    current_file.close()
                    file_index += 1
                    line_count = 0
                    current_file = open(os.path.join(FOLDER_PATH, f"udp_data_{file_index}.csv"), "w", newline="",
                                        encoding="utf-8")
                    csv_writer = csv.writer(current_file)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            current_file.close()
            sock.close()
