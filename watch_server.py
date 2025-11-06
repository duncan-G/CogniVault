#!/usr/bin/env python3
"""
Watch server that monitors file changes and restarts the gRPC server
"""
import subprocess
import sys
import time
import os
import signal
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class ServerRestartHandler(FileSystemEventHandler):
    """Handler for file system events that restarts the server"""
    
    def __init__(self, script_path, port):
        self.script_path = script_path
        self.port = port
        self.process = None
        self.restart_delay = 1.0  # Delay before restart to avoid rapid restarts
        self.last_restart = 0
    
    def generate_proto_files(self):
        """Generate Python code from proto files"""
        proto_dir = os.path.join(os.path.dirname(__file__), 'proto')
        proto_file = os.path.join(proto_dir, 'cognivault.proto')
        
        if os.path.exists(proto_file):
            try:
                import subprocess
                result = subprocess.run(
                    ['python', '-m', 'grpc_tools.protoc', 
                     '-I.', 
                     '--python_out=.', 
                     '--grpc_python_out=.',
                     'proto/cognivault.proto'
                    ],
                    capture_output=True,
                    text=True,
                    cwd=os.path.dirname(__file__)
                )
                if result.returncode == 0:
                    print("Proto files generated successfully")
                else:
                    print(f"Proto generation warning: {result.stderr}")
            except Exception as e:
                print(f"Error generating proto files: {e}")
        
    def start_server(self):
        """Start the gRPC server"""
        if self.process:
            self.stop_server()
        
        print(f"\n{'='*50}")
        print(f"Starting gRPC server on port {self.port}")
        print(f"{'='*50}\n")
        
        env = os.environ.copy()
        self.process = subprocess.Popen(
            [sys.executable, self.script_path, '--port', str(self.port)],
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env
        )
    
    def stop_server(self):
        """Stop the gRPC server"""
        if self.process:
            print("\nStopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def restart_server(self):
        """Restart the server with debouncing"""
        current_time = time.time()
        if current_time - self.last_restart < self.restart_delay:
            return
        
        self.last_restart = current_time
        self.stop_server()
        time.sleep(0.5)  # Brief pause before restart
        self.start_server()
    
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
        
        # Only watch Python and proto files
        if event.src_path.endswith(('.py', '.proto')):
            print(f"\nFile changed: {event.src_path}")
            
            # If proto file changed, regenerate proto files first
            if event.src_path.endswith('.proto'):
                print("Regenerating proto files...")
                self.generate_proto_files()
                time.sleep(0.5)  # Brief pause to ensure files are written
            
            self.restart_server()
    
    def on_created(self, event):
        """Handle file creation events"""
        if event.is_directory:
            return
        
        if event.src_path.endswith(('.py', '.proto')):
            print(f"\nFile created: {event.src_path}")
            
            # If proto file created, regenerate proto files first
            if event.src_path.endswith('.proto'):
                print("Regenerating proto files...")
                self.generate_proto_files()
                time.sleep(0.5)  # Brief pause to ensure files are written
            
            self.restart_server()


def watch_and_serve(port=50052, watch_dirs=None):
    """Watch directories and restart server on changes"""
    script_path = os.path.join(os.path.dirname(__file__), 'server.py')
    
    if watch_dirs is None:
        watch_dirs = ['.']
    
    handler = ServerRestartHandler(script_path, port)
    
    # Generate proto files initially
    print("Generating proto files...")
    handler.generate_proto_files()
    
    # Start server initially
    handler.start_server()
    
    # Setup file watchers
    observers = []
    for watch_dir in watch_dirs:
        if os.path.exists(watch_dir):
            observer = Observer()
            observer.schedule(handler, watch_dir, recursive=True)
            observer.start()
            observers.append(observer)
            print(f"Watching directory: {os.path.abspath(watch_dir)}")
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\nShutting down...")
        handler.stop_server()
        for observer in observers:
            observer.stop()
        for observer in observers:
            observer.join()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("\nWatching for file changes... Press Ctrl+C to stop.\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CogniVault gRPC Server with File Watching')
    parser.add_argument('--port', type=int, default=50052, help='Server port (default: 50052)')
    parser.add_argument('--watch', nargs='*', default=None, help='Directories to watch (default: current directory)')
    args = parser.parse_args()
    
    watch_and_serve(args.port, args.watch)

