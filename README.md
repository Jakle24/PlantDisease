# PlantDisease


How to restart 

From backend folder:

Dev (fast)

python main.py


Windows production-ish (waitress)

cd "C:\path\to\PlantDisease\backend"
python -m pip install waitress   # once
waitress-serve --listen=127.0.0.1:5000 main:app


Linux lab (gunicorn)

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
gunicorn -w 4 -b 127.0.0.1:5000 main:app


Frontend:

cd frontend
npm install   # if not done
npm start