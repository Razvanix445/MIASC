import os
import sys
import webbrowser
import threading
import time
from app import app


def open_browser():
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    threading.Thread(target=open_browser).start()

    app.run(debug=False, host='0.0.0.0', port=5000)