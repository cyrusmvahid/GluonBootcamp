import picamera, time 

camera = picamera.PiCamera()

camera.start_preview()
time.sleep(30)
camera.stop_preview()