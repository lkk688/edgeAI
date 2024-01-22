#https://github.com/raspberrypi/picamera2/tree/main/examples
import time, libcamera
from picamera2 import Picamera2, Preview

picam = Picamera2()
#picam.rotation = 90

config = picam.create_preview_configuration(main={"size": (1600, 1200)})
config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picam.configure(config)

picam.start_preview(Preview.QTGL)

picam.start()
time.sleep(20)
picam.capture_file("test-python.jpg")

np_array = picam.capture_array()
print(np_array)

# for i in range(1,10):
#     picam.capture_file(f"ts{i}.jpg")
#     print(f"Captured image {i}")
#     time.sleep(3)

picam.close()

#build video
#ffmpeg -r 1 -pattern_type glob -i "ts*.jpg" -vcodec libx264 timelapse.mp4