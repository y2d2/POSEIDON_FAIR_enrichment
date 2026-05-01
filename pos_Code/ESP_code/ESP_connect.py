import socket
import os
import csv
from datetime import datetime
#
# from ESP_Class import ESP_wifi_module
# esp = ESP_wifi_module()
# esp.



LINES_PER_FILE = 10000         # new file after 10k lines
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 5005))

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
FOLDER_PATH = f"./data/udp_data_{timestamp}"    # e.g. udp_data_2025-09-15_14-32-10
os.makedirs(FOLDER_PATH, exist_ok=True)

file_index = 1
line_count = 0
current_file = open(os.path.join(FOLDER_PATH, f"udp_data_{file_index}.csv"), "w", newline="")
csv_writer = csv.writer(current_file)
csv_writer.writerow(["id", "time", "mag_x", "mag_y", "mag_z", "T",
                     "gx","gy", "gz", "ax", "ay", "az",
                     "valid_ranges", "rid", "dist  m", "fp_rssi", "rx_rssi", "..."])
record = True
debug = True
print("Listening for UDP data...")
try:
    while True:
        data, addr = sock.recvfrom(1024)
        decoded = data.decode(errors="replace")  # your string like "A;B;C;D"

        # split the string by ';' into columns

        row = decoded.split(";")
        if record:
            csv_writer.writerow(row)
            line_count += 1
        if debug:
            print(file_index, line_count, row)
            # print(file_index, line_count)
        # Check if we reached the line limit
        if line_count >= LINES_PER_FILE:
            current_file.close()
            file_index += 1
            print(file_index)
            line_count = 0
            current_file = open(os.path.join(FOLDER_PATH, f"udp_data_{file_index}.csv"), "w", newline="", encoding="utf-8")
            csv_writer = csv.writer(current_file)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    current_file.close()
    sock.close()


