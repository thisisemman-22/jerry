from flask import Flask, request, jsonify
from image_processing import process_image
from werkzeug.exceptions import BadRequest
import os

app = Flask(__name__)

@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return jsonify({"error": "Bad Request", "message": str(e)}), 400

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.route('/downscale', methods=['POST'])
def downscale_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    try:
        image_file = request.files['image']
        output_path = process_image(image_file, 'downscale')
        output_url = request.host_url + output_path  # Convert path to URL
        return jsonify({"output_url": output_url})
    except Exception as e:
        return jsonify({"error": "Processing failed", "message": str(e)}), 500

@app.route('/upscale', methods=['POST'])
def upscale_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    try:
        image_file = request.files['image']
        output_path = process_image(image_file, 'upscale')
        output_url = request.host_url + output_path  # Convert path to URL
        return jsonify({"output_url": output_url})
    except Exception as e:
        return jsonify({"error": "Processing failed", "message": str(e)}), 500

@app.route('/denoise', methods=['POST'])
def denoise_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    try:
        image_file = request.files['image']
        output_path = process_image(image_file, 'denoise')
        output_url = request.host_url + output_path  # Convert path to URL
        return jsonify({"output_url": output_url})
    except Exception as e:
        return jsonify({"error": "Processing failed", "message": str(e)}), 500

@app.route('/blur', methods=['POST'])
def blur_route():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
    if 'radius' not in request.form:
        return jsonify({"error": "No radius value provided."}), 400
    try:
        image_file = request.files['image']
        radius = int(request.form['radius'])  # Get radius from form data
        output_path = process_image(image_file, 'blur', radius=radius)
        output_url = request.host_url + output_path  # Convert path to URL
        return jsonify({"output_url": output_url})
    except ValueError:
        return jsonify({"error": "Invalid radius value."}), 400
    except Exception as e:
        return jsonify({"error": "Processing failed", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)