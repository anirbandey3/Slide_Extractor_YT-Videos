# run_app.py
import streamlit as st
import socket
import webbrowser

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__':
    local_ip = get_local_ip()
    port = 8501
    
    print(f"\nYou can access the application at:")
    print(f"Local URL: http://localhost:{port}")
    print(f"Network URL: http://{local_ip}:{port}")
    
    # Open browser automatically
    webbrowser.open(f"http://localhost:{port}")
    
    # Run Streamlit
    import subprocess
    subprocess.run(["streamlit", "run", "app.py", 
                   "--server.address", "0.0.0.0",
                   "--server.port", str(port)])