import http.server
import socketserver
import webbrowser
import threading
import time
import os

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def start_server():
    # Try to find an open port
    global PORT
    while True:
        try:
            httpd = socketserver.TCPServer(("", PORT), Handler)
            print(f"Serving at port {PORT}")
            httpd.serve_forever()
        except OSError:
            PORT += 1

if __name__ == '__main__':
    # Start the server in a separate thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Wait a moment for server to start
    time.sleep(1.5)
    
    # Open the web browser
    url = f"http://localhost:{PORT}/map_ridgecrest.html"
    print(f"Opening browser at {url}")
    webbrowser.open(url)
    print("Press Ctrl+C to stop the server.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
