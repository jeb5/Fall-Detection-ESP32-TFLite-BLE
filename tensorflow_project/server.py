from fastapi import FastAPI, Request
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import uvicorn

app = FastAPI()

# Data storage for plotting - using lists to handle out-of-order data
max_points = 500
plot_data = {
    'time': [],
    'button': [],
    'acc_x': [],
    'acc_y': [],
    'acc_z': [],
    'gyro_x': [],
    'gyro_y': [],
    'gyro_z': [],
}
last_time = None

# Create figure and subplots with custom layout
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :])  # Accelerometer spans both columns
ax2 = fig.add_subplot(gs[1, :])  # Gyroscope spans both columns
ax3 = fig.add_subplot(gs[2, 0])  # Button left half
ax4 = fig.add_subplot(gs[2, 1])  # Delta time right half
fig.suptitle('Real-time IMU Data', fontsize=16)

# Initialize line objects
lines_acc = {
    'acc_x': ax1.plot([], [], 'r-', label='Acc X')[0],
    'acc_y': ax1.plot([], [], 'g-', label='Acc Y')[0],
    'acc_z': ax1.plot([], [], 'b-', label='Acc Z')[0],
}

lines_gyro = {
    'gyro_x': ax2.plot([], [], 'r-', label='Gyro X')[0],
    'gyro_y': ax2.plot([], [], 'g-', label='Gyro Y')[0],
    'gyro_z': ax2.plot([], [], 'b-', label='Gyro Z')[0],
}

# Button line
line_button = ax3.plot([], [], 'purple', linewidth=2, label='Button State')[0]

# Delta time line
line_delta = ax4.plot([], [], 'orange', linewidth=2, label='Delta Time')[0]

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration')
ax1.legend(loc='upper right')
ax1.grid(True)

ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Gyroscope')
ax2.legend(loc='upper right')
ax2.grid(True)

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Button')
ax3.set_ylim(-0.1, 1.1)
ax3.legend(loc='upper right')
ax3.grid(True)

ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Delta Time (ms)')
ax4.legend(loc='upper right')
ax4.grid(True)

def update_plot(frame):
    if len(plot_data['time']) > 0:
        # Sort data by time to handle out-of-order arrivals
        sorted_indices = sorted(range(len(plot_data['time'])), key=lambda i: plot_data['time'][i])
        
        # Keep only the last max_points
        if len(sorted_indices) > max_points:
            sorted_indices = sorted_indices[-max_points:]
        
        # Convert time from milliseconds to seconds
        time_data = [plot_data['time'][i] / 1000.0 for i in sorted_indices]
        
        # Update accelerometer lines
        for key in ['acc_x', 'acc_y', 'acc_z']:
            data = [plot_data[key][i] for i in sorted_indices]
            lines_acc[key].set_data(time_data, data)
        
        # Update gyroscope lines
        for key in ['gyro_x', 'gyro_y', 'gyro_z']:
            data = [plot_data[key][i] for i in sorted_indices]
            lines_gyro[key].set_data(time_data, data)
        
        # Update button line
        button_data = [plot_data['button'][i] for i in sorted_indices]
        line_button.set_data(time_data, button_data)
        
        # Calculate delta time: difference between consecutive time values (in milliseconds)
        delta_data = [0] + [plot_data['time'][sorted_indices[i]] - plot_data['time'][sorted_indices[i-1]] for i in range(1, len(sorted_indices))]
        line_delta.set_data(time_data, delta_data)
        
        
        # Adjust y-axes limits
        ax1.relim()
        ax1.autoscale_view(scalex=False)  # Don't autoscale x since we set it manually
        ax2.relim()
        ax2.autoscale_view(scalex=False)
        ax3.relim()
        ax3.autoscale_view(scalex=False)
        ax3.set_ylim(0, 1)  # Keep button y-axis fixed
        ax4.relim()
        ax4.autoscale_view(scalex=False)

        # Set x-axis limits based on actual time data
        if len(time_data) > 1:
            time_min, time_max = min(time_data), max(time_data)
            time_range = time_max - time_min
            # Add a small margin (5%) to make the data easier to see
            margin = time_range * 0.05 if time_range > 0 else 0.1
            ax1.set_xlim(time_min - margin, time_max + margin)
            ax2.set_xlim(time_min - margin, time_max + margin)
            ax3.set_xlim(time_min - margin, time_max + margin)
            ax4.set_xlim(time_min - margin, time_max + margin)
            # Update axis to reflect changes
            ax1.figure.canvas.draw()
            ax2.figure.canvas.draw()
            ax3.figure.canvas.draw()
            ax4.figure.canvas.draw()

    
    return list(lines_acc.values()) + list(lines_gyro.values()) + [line_button, line_delta]

def start_server():
    """Run FastAPI server in a background thread"""
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

def main():
  # Create data.csv
  with open("data.csv", "w") as f:
    f.write("time_ms,button,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
  print("Created data.csv")
  
  # Start FastAPI server in a separate thread
  server_thread = threading.Thread(target=start_server, daemon=True)
  server_thread.start()
  print("Server started on http://0.0.0.0:8000")
  
  # Start plotting on the main thread
  ani = FuncAnimation(fig, update_plot, interval=50, blit=True)
  plt.tight_layout()
  plt.show()


# @app.post("/i_have_fallen")
# async def fallen():
#   print("Fallen!")
#   for i in range(25):
#     for pin in pins:
#       GPIO.output(pin,GPIO.HIGH)
#     await asyncio.sleep(0.2)
#     for pin in pins:
#       GPIO.output(pin,GPIO.LOW)
#     await asyncio.sleep(0.2)
#   return "Okay"


@app.post("/log_data")
async def process_data(request:Request):
  global last_time
  data = await request.body()
  floats = np.frombuffer(data, dtype=np.float32)
  ints = np.frombuffer(data, dtype=np.int32)
  floats = floats.reshape(8,64)
  ints = ints.reshape(8,64)
  with open("data.csv", "a") as f:
    for i in range(64):
      
      csv_line = f"{ints[0,i]},{ints[1,i]},{floats[2,i]},{floats[3,i]},{floats[4,i]},{floats[5,i]},{floats[6,i]},{floats[7,i]}"
      f.write(csv_line + "\n")
      print(csv_line)

      # Add data to plot buffers
      plot_data['time'].append(ints[0,i])
      plot_data['button'].append(ints[1,i])
      plot_data['acc_x'].append(floats[2,i])
      plot_data['acc_y'].append(floats[3,i])
      plot_data['acc_z'].append(floats[4,i])
      plot_data['gyro_x'].append(floats[5,i])
      plot_data['gyro_y'].append(floats[6,i])
      plot_data['gyro_z'].append(floats[7,i])
      
      # Trim data if it exceeds max_points to prevent memory issues
      if len(plot_data['time']) > max_points * 2:
        for key in plot_data:
          plot_data[key] = plot_data[key][-max_points:]

  return {"message":"Processed"}

main()