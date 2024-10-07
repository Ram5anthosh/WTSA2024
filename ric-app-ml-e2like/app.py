import socket
import time
from datetime import datetime
import numpy as np
from PIL import Image
import folium

from log import *

CONFIDENCE_THRESHOLD = 0.5
current_image_data = None
current_gps_coords = None
server = None

cmds = {
    'DRONE_NAVIGATE': b'navigate',
    'DRONE_HALT': b'halt',
    'DRONE_DETECT_WASTE': b'detect_waste',
    'AVOID_OBSTACLE': b'avoid_obstacle',
}

global_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

def init_e2(self):
    global server
    ip_addr = socket.gethostbyname(socket.gethostname())
    port = 5000
    log_info(self, f"E2-like enabled, connecting using SCTP on {ip_addr}")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip_addr, port))
    server.listen()
    log_info(self, 'Server started')

def entry(self):
    global current_image_data, current_gps_coords, server
    init_e2(self)
    while True:
        try:
            conn, addr = server.accept()
            log_info(self, f'Connected by {addr}')
            while True:
                conn.send(f"E2-like request at {datetime.now().strftime('%H:%M:%S')}".encode('utf-8'))
                log_info(self, "Sent E2-like request for image data")
                data = conn.recv(16384)
                if data:
                    log_info(self, f"Receiving image data...")
                    while len(data) < expected_image_size:
                        data += conn.recv(16384)
                    log_info(self, f"Received buffer size {len(data)}")
                    log_info(self, f"Finished receiving image data, processing")

                    current_image_data = data
                    current_gps_coords = get_current_gps_location()
                    result = run_prediction(self)

                    if result == 'Waste Detected':
                        log_info(self, "Waste detected, marking location on the map and halting drone.")
                        conn.send(cmds['DRONE_HALT'])
                        mark_location_on_map(current_gps_coords)
                    else:
                        log_info(self, "No waste detected, continuing navigation.")
                        conn.send(cmds['DRONE_NAVIGATE'])

                    if obstacle_detected():
                        log_info(self, "Obstacle detected, sending avoid command.")
                        conn.send(cmds['AVOID_OBSTACLE'])

        except OSError as e:
            log_error(self, e)

def get_current_gps_location():
    return (20.5937, 78.9629)

def mark_location_on_map(coords):
    global global_map
    folium.Marker(location=coords, popup="Waste Detected").add_to(global_map)
    global_map.save("drone_waste_detection_map.html")

def run_prediction(self):
    global current_image_data
    processed_image = process_image(current_image_data)
    result = predict_waste(processed_image)
    return result

def process_image(image_data) -> np.ndarray:
    image = Image.open(BytesIO(image_data))
    image = image.resize((256, 256))
    return np.array(image)

def predict_waste(image: np.ndarray) -> str:
    prediction, confidence = model_predict(ai_model, image)
    return 'Waste Detected' if confidence > CONFIDENCE_THRESHOLD else 'No Waste'

def obstacle_detected():
    return random.choice([True, False])

def model_predict(model, image_data):
    prediction = random.choice([0, 1])
    confidence = random.random()
    return prediction, confidence

def load_model_parameter():
    return None

def start(thread=False):
    global ai_model
    ai_model = load_model_parameter()
    entry(None)

if __name__ == '__main__':
    start()
