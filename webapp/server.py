from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="webapp")

@app.route('/')
def index():
    return send_from_directory("webapp", "index.html")

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory("webapp", path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
