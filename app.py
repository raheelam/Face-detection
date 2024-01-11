import json
from flask import Flask, render_template, jsonify;
from face_detection import load_images, train_models, test_button_callback;

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # Create an HTML file for the frontend

@app.route('/load_images')
def load_images_route():
    # Your existing load_images function here
    faces = load_images()
    return jsonify({"status": "Images loaded successfully", "result": faces })

@app.route('/train_models')
def train_models_route():
    # Your existing train_models function here
    train_models()
    return jsonify({"status": "Models trained successfully"})

@app.route('/test_models')
def test_models_route():
    # Your existing test_button_callback function here
    test_result = test_button_callback()  # Modify this line to return the actual test result
    print(test_result)
    # return jsonify({"test_result": test_result})

    json_data = json.dumps(test_result, default=lambda x: 'Infinity' if x == float('inf') else str(x))
    return jsonify({"test_result": json_data})
    # return render_template('index.html', test_result=test_result)

if __name__ == "__main__":
    app.run(debug=True)
