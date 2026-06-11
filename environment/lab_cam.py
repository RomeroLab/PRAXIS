from flask import Flask, Response, redirect, url_for, request
import cv2
import datetime
import hashlib
import getpass
import threading
import time
import ssl

# Global variables
PASSWORD_HASH = None
app = Flask(__name__)

class Camera:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.camera = None
        self.frame_lock = threading.Lock()
        self.frame = None
        self.running = False
        self.connect_camera()   
        
    def connect_camera(self):
        """Initialize camera connection with retries"""
        print(f"Connecting to camera {self.camera_id}")
        try:
            if self.camera is not None:
                self.camera.release()
                time.sleep(1)
                
            self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                raise Exception("Failed to open camera")
                
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Verify we can read a frame
            ret, _ = self.camera.read()
            if not ret:
                raise Exception("Couldn't read frame")
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.daemon = True
            self.thread.start()
            
            print(f"Successfully connected to camera {self.camera_id}")
            return True
            
        except Exception as e:
            print(f"Error connecting to camera {self.camera_id}: {e}")
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            return False
        
    def _capture_frames(self):
        while self.running:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    with self.frame_lock:
                        self.frame = frame
            time.sleep(0.033)  # ~30 FPS
                    
    def get_frame(self):
        with self.frame_lock:
            if self.frame is None:
                return None
            
            # Make a copy of the frame
            frame = self.frame.copy()
            
        # Add timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Add camera ID
        cv2.putText(frame, f"Camera {self.camera_id + 1}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            return buffer.tobytes()
        return None
        
    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1)
        if self.camera is not None:
            self.camera.release()
            self.camera = None

def check_password(password):
    """Check if password matches the stored hash"""
    return hashlib.sha256(password.encode()).hexdigest() == PASSWORD_HASH

def generate_frames(camera_id):
    """Generate frames for video streaming"""
    camera = cameras.get(camera_id)
    if not camera:
        return
    
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.1)

@app.route('/')
def index():
    password = request.args.get('password')
    if not password or not check_password(password):
        return redirect(url_for('login'))
    
    return f"""
    <html>
    <head>
        <title>Robot Lab Monitor</title>
        <style>
            body {{
                font-family: Menlo, monospace;
                margin: 20px;
                background-color: #f0f0f0;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                text-align: center;
            }}
            .btn-group {{
                text-align: center;
                margin: 20px 0;
            }}
            .camera-btn {{
                background-color: #4CAF50;
                color: white;
                padding: 10px 20px;
                margin: 0 10px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            }}
            .camera-btn:hover {{
                background-color: #45a049;
            }}
            .camera-btn.active {{
                background-color: #357a38;
                box-shadow: 0 0 5px rgba(0,0,0,0.3);
            }}
            .stream-container {{
                text-align: center;
                margin-top: 20px;
            }}
            .stream {{
                max-width: 100%;
                border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
        </style>
        <script>
            function switchCamera(camId) {{
                // Update button states
                document.querySelectorAll('.camera-btn').forEach(btn => {{
                    btn.classList.remove('active');
                }});
                event.target.classList.add('active');
                
                // Update stream source
                const stream = document.getElementById('camera-stream');
                stream.src = `/video_feed/${{camId}}?password={password}`;
            }}
        </script>
    </head>
    <body>
        <div class="container">
            <h1>Robot Lab Monitor</h1>
            
            <div class="btn-group">
                <button class="camera-btn active" onclick="switchCamera('cam0')">
                    Camera 1
                </button>
                <button class="camera-btn" onclick="switchCamera('cam1')">
                    Camera 2
                </button>
            </div>
            
            <div class="stream-container">
                <img id="camera-stream" class="stream" 
                     src="/video_feed/cam0?password={password}">
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/login')
def login():
    return """
    <html>
    <head>
        <title>Robot Camera Login</title>
        <style>
            body {
                font-family: Menlo, monospace;
                margin: 40px;
                background-color: #f0f0f0;
            }
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 20px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h2 {
                color: #333;
                text-align: center;
            }
            input[type="password"] {
                width: 100%;
                padding: 10px;
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
            }
            input[type="submit"] {
                width: 100%;
                padding: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            input[type="submit"]:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2>Robot Camera Login</h2>
            <form action="/auth">
                <input type="password" name="password" placeholder="Enter password">
                <input type="submit" value="Login">
            </form>
        </div>
    </body>
    </html>
    """

@app.route('/auth')
def auth():
    password = request.args.get('password')
    if check_password(password):
        return redirect(url_for('index', password=password))
    return redirect(url_for('login'))

@app.route('/video_feed/<camera_id>')
def video_feed(camera_id):
    password = request.args.get('password')
    if not password or not check_password(password):
        return redirect(url_for('login'))
    
    # Release other cameras
    for cam_name, cam in cameras.items():
        if cam_name != camera_id:
            cam.release()
    
    # Make sure the selected camera is connected
    camera = cameras.get(camera_id)
    if camera and (camera.camera is None or not camera.camera.isOpened()):
        camera.connect_camera()
    
    return Response(generate_frames(camera_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Get password from user
    while True:
        password = getpass.getpass("Enter password for stream access: ")
        if password:
            break
        print("Password cannot be empty")
    
    # Store password hash
    PASSWORD_HASH = hashlib.sha256(password.encode()).hexdigest()
    
    # Initialize cameras
    cameras = {
        'cam0': Camera(0),
        'cam1': Camera(1)
    }
    
    # Print access information
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    port = 8000
    
    print("\nAccess URLs:")
    print(f"Local computer: http://localhost:{port}/")
    print(f"Other computers: http://{local_ip}:{port}/")
    
    # Run the app
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        for camera in cameras.values():
            camera.release()