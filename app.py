from flask import Flask, request, send_file, jsonify
import subprocess
import tempfile
import os
import shutil

app = Flask(__name__)


NCNN_BIN = "/home/cholai/realesrgan-ncnn-vulkan-v0.2.0-ubuntu/realesrgan-ncnn-vulkan"
NVIDIA_SCRIPT = "/home/cholai/Real-ESRGAN/inference_realesrgan.py"
VENV_PYTHON = "/home/cholai/Real-ESRGAN/venv/bin/python"  


if shutil.which(VENV_PYTHON) and os.path.exists(NVIDIA_SCRIPT):
    try:
        test_cmd = [VENV_PYTHON, "-c", "import torch; print(torch.cuda.is_available())"]
        res = subprocess.run(test_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        USE_NVIDIA = "True" in res.stdout
    except Exception:
        USE_NVIDIA = False
else:
    USE_NVIDIA = False

print("✅ Backend chosen:", "NVIDIA (PyTorch)" if USE_NVIDIA else "NCNN Vulkan")


@app.route("/upscale", methods=["POST"])
def upscale_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    temp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    file.save(temp_in.name)
    temp_in_path = temp_in.name
    temp_in.close()

    temp_out_path = tempfile.mktemp(suffix=".jpg")

    scale = request.form.get("scale", "2")

    if USE_NVIDIA:
        model = "RealESRGAN_x4plus" 
        tile = "350" 
        cmd = [
            VENV_PYTHON, NVIDIA_SCRIPT,
            "-i", temp_in_path,
            "-o", temp_out_path,
            "-n", model,
            "-s", scale,
            "-t", tile
        ]
    else:
        cmd = [
            NCNN_BIN,
            "-i", temp_in_path,
            "-o", temp_out_path,
            "-s", scale
        ]

    print(f"⚙️ Running command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Upscale failed: {str(e)}"}), 500
    except FileNotFoundError:
        return jsonify({"error": "Executable not found"}), 500
    finally:
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)

    if not os.path.exists(temp_out_path):
        return jsonify({"error": "No output image generated"}), 500

    return send_file(
        temp_out_path,
        mimetype="image/jpeg",
        as_attachment=True,
        download_name="upscaled.jpg"
    )


if __name__ == "__main__":
    print("🚀 Flask Real-ESRGAN server started on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5000, debug=True)
