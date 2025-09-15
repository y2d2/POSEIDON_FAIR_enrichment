import struct
import serial
import time

# ==== CONFIGURATION ====
PORT = "/dev/ttyACM0"  # Change to your device path
BAUDRATE = 115200      # Typical for Nooploop devices

# ==== COMMANDS ====
# Command format: AA 00 LEN CMD MODE 00 00 CS
# MODE = 0x02 → DR_MODE0 (raw), MODE = 0x03 → DR_MODE1 (filtered)
DR_MODE0_CMD = bytes([0xAA, 0x00, 0x05, 0x41, 0x02, 0x00, 0x00, 0x43])
DR_MODE1_CMD = bytes([0xAA, 0x00, 0x05, 0x41, 0x03, 0x00, 0x00, 0x44])

# ==== OPEN SERIAL ====
ser = serial.Serial(PORT, BAUDRATE, timeout=0.5)

def crc16(data: bytes):
    """CRC-16/IBM (polynomial 0x8005, initial value 0x0000)"""
    crc = 0
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if (crc & 0x8000):
                crc = (crc << 1) ^ 0x8005
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def parse_nodeframe5(packet: bytes):
    if not (packet[0] == 0x55 and packet[1] == 0x05):
        return None

    length = packet[3]
    payload = packet[4:4 + length]
    crc_recv = struct.unpack("<H", packet[4 + length:4 + length + 2])[0]

    if crc16(packet[0:4 + length]) != crc_recv:
        print("CRC error")
        return None

    offset = 0
    system_time = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    node_id = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    role = payload[offset]
    offset += 1
    pos = struct.unpack_from("<fff", payload, offset)
    offset += 12
    vel = struct.unpack_from("<fff", payload, offset)
    offset += 12
    quat = struct.unpack_from("<ffff", payload, offset)
    offset += 16

    # Read ranges
    range_count = payload[offset]
    offset += 1
    ranges = []
    for _ in range(range_count):
        target_id = struct.unpack_from("<H", payload, offset)[0]
        offset += 2
        distance = struct.unpack_from("<f", payload, offset)[0]
        offset += 4
        ranges.append((target_id, distance))

    return {
        "system_time": system_time,
        "node_id": node_id,
        "role": role,
        "pos": pos,
        "vel": vel,
        "quat": quat,
        "ranges": ranges
    }


def parse_nodeframe2(packet: bytes):
    # Check header
    if not (packet[0] == 0x55 and packet[1] == 0x01 and packet[2] == 0x02):
        return None

    length = packet[3]
    payload = packet[4:4+length]
    crc_recv = struct.unpack("<H", packet[4+length:4+length+2])[0]

    # CRC check
    if crc16(packet[0:4+length]) != crc_recv:
        print("CRC error")
        return None

    # Unpack fields
    offset = 0
    system_time = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    node_id = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    role = payload[offset]
    offset += 1
    pos = struct.unpack_from("<fff", payload, offset)
    offset += 12
    vel = struct.unpack_from("<fff", payload, offset)
    offset += 12
    quat = struct.unpack_from("<ffff", payload, offset)
    offset += 16

    return {
        "system_time": system_time,
        "node_id": node_id,
        "role": role,
        "pos": pos,
        "vel": vel,
        "quat": quat
    }



def send_command(cmd_bytes):
    ser.write(cmd_bytes)
    print(f"Sent command: {cmd_bytes.hex(' ').upper()}")

# ==== SET TO DR MODE ====
print("Setting device to DR_MODE1 (filtered)...")
# send_command(DR_MODE0_CMD)
time.sleep(0.1)

# ==== START READING DATA ====
print("Reading data from device (CTRL+C to stop)...")
buffer = bytearray()

try:
    while True:
        data = ser.read(ser.in_waiting or 1)
        if data:
            # print(data)
            buffer.extend(data)

            # Search for header
            while len(buffer) >= 6:
                print(buffer)
                if buffer[0] == 0x55 and buffer[1] == 0x05 and buffer[2] == 0x02:
                    length = buffer[3]
                    packet_len = 4 + length + 2
                    if len(buffer) >= packet_len:
                        packet = buffer[:packet_len]
                        buffer = buffer[packet_len:]
                        parsed = parse_nodeframe5(packet)
                        if parsed:
                            print(f"[{parsed['system_time']} ms] "
                                  f"Pos={parsed['pos']} Vel={parsed['vel']} "
                                  f"Quat={parsed['quat']}")
                    else:
                        break
                else:
                    buffer.pop(0)

except KeyboardInterrupt:
    print("Stopped.")
finally:
    ser.close()

#
# import serial
# import time
# from nlink_parser.parsers import LinktrackNodeframe2
#
# # ==== CONFIGURATION ====
# PORT = "/dev/ttyACM0"  # Change to your port
# BAUDRATE = 115200
#
# # ==== COMMAND ====
# # DR_MODE0 (raw dead reckoning)
# DR_MODE0_CMD = bytes([0xAA, 0x00, 0x05, 0x41, 0x02, 0x00, 0x00, 0x43])
#
# # Create parser for NodeFrame2 (DR mode output)
# parser = LinktrackNodeframe2()
#
# # ==== SERIAL ====
# ser = serial.Serial(PORT, BAUDRATE, timeout=0.5)
#
# def send_command(cmd_bytes):
#     ser.write(cmd_bytes)
#     print(f"Sent: {cmd_bytes.hex(' ').upper()}")
#
# # ==== SET TO DR_MODE0 ====
# print("Setting device to DR_MODE0 (raw DR)...")
# send_command(DR_MODE0_CMD)
# time.sleep(0.1)
#
# # ==== READ & PARSE ====
# print("Reading DR mode data (Node Frame 2)...")
# try:
#     buffer = bytearray()
#     while True:
#         if ser.in_waiting:
#             data = ser.read(ser.in_waiting)
#             buffer.extend(data)
#
#             # Parse packets in buffer
#             while parser.unpack(buffer):
#                 # Remove parsed packet bytes
#                 buffer = buffer[parser.frame_length:]
#
#                 # Print decoded values
#                 print(f"Timestamp: {parser.system_time}")
#                 print(f"Position: x={parser.pos_3d[0]:.3f} m, "
#                       f"y={parser.pos_3d[1]:.3f} m, "
#                       f"z={parser.pos_3d[2]:.3f} m")
#                 print(f"Velocity: vx={parser.vel_3d[0]:.3f} m/s, "
#                       f"vy={parser.vel_3d[1]:.3f} m/s, "
#                       f"vz={parser.vel_3d[2]:.3f} m/s")
#                 print(f"Quaternion: {parser.quaternion}")
#                 print("-" * 40)
#
#         time.sleep(0.02)
#
# except KeyboardInterrupt:
#     print("Stopped by user.")
# finally:
#     ser.close()
#
