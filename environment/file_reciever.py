from flask import Flask, request, jsonify
import os

app = Flask(__name__)

BASE_UPLOAD_DIR = os.path.dirname(os.fspath(__file__)) #Current dir
target_dir = 'data'
UPLOAD_FOLDER = os.path.join(BASE_UPLOAD_DIR,target_dir)
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file.save(os.path.join(UPLOAD_FOLDER, file.filename))
    return jsonify({'message': f'File {file.filename} uploaded successfully'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)