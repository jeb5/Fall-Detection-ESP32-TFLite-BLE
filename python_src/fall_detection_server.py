import threading
import uvicorn
from fastapi import FastAPI
# if using Raspberry Pi GPIO, uncomment the next line
import RPi.GPIO as GPIO
import asyncio
from bleak import BleakScanner

app = FastAPI()
currently_fallen = False
fall_lock = asyncio.Lock()
ESP32_NAME = "FallDetector"

pins = [8, 18, 23, 25]  # GPIO pins for LEDs

def start_server():
    """Run FastAPI server in a background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

async def main():
  GPIO.setmode(GPIO.BCM)
  for pin in pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)
  
  # Start FastAPI server in a separate thread
  server_thread = threading.Thread(target=start_server, daemon=True)
  server_thread.start()
  print("Server started on http://0.0.0.0:8000")
  scanner = BleakScanner(detection_callback)
  await scanner.start()
  print("Started BLE scanner")
  await asyncio.sleep(3600)  # Run for 1 hour
  await scanner.stop()

def detection_callback(device, advertisement_data):
  if device.name == ESP32_NAME:
    asyncio.create_task(fallen())

@app.post("/i_have_fallen")
async def fallen_endpoint():
  await fallen()

async def fallen():
  global currently_fallen
  # if currently_fallen:
  #   return "Already fallen"
  if fall_lock.locked():
    return "Already fallen"
  async with fall_lock:
    currently_fallen = True
    print("Fallen!")
    for i in range(25):
      for pin in pins:
        GPIO.output(pin,GPIO.HIGH)
      await asyncio.sleep(0.2)
      for pin in pins:
        GPIO.output(pin,GPIO.LOW)
      await asyncio.sleep(0.2)
    currently_fallen = False
  return "Okay"


if __name__ == "__main__":
  asyncio.run(main())
