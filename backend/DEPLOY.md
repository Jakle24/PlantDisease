Deployment notes — quick guide
-----------------------------

1) Create a Python virtualenv (recommended):
   python -m venv venv
   # Windows (PowerShell)
   venv\Scripts\Activate.ps1
   # Linux / Git Bash
   source venv/bin/activate

2) Install requirements:
   pip install -r requirements.txt
   # or: pip install flask flask-cors tensorflow pillow numpy

3) Use a production WSGI server:
   - Windows (simple): use waitress
       pip install waitress
       waitress-serve --listen=0.0.0.0:5000 main:app

   - Linux (recommended): use gunicorn behind nginx
       pip install gunicorn
       gunicorn -w 4 -b 0.0.0.0:5000 main:app

4) Optional: Docker
   - Build an image with Python + your model. Use a multistage build for small image.
   - Store model in image or mount as volume.

5) Environment:
   - For frontend, set REACT_APP_BACKEND_URL to production URL before building:
       REACT_APP_BACKEND_URL=https://yourdomain.com npm run build

6) Security:
   - Use HTTPS and restrict CORS in production (CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}}))
   - Do not use Flask debug mode in production.

7) Systemd example (Linux):
   [Unit]
   Description=PlantDisease backend
   After=network.target

   [Service]
   User=www-data
   Group=www-data
   WorkingDirectory=/path/to/backend
   ExecStart=/path/to/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 main:app
   Restart=always

   [Install]
   WantedBy=multi-user.target


For debugging, run the server with APP_LOG_LEVEL=DEBUG python main.py.