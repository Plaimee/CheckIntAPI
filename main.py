import datetime
import io
import os
import uuid
import requests
import urllib.parse
import websocket
import ftplib
from dotenv import load_dotenv

from flask import Flask, json, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import rembg

load_dotenv()

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = 'merged_images'
os.makedirs(OUTPUT_DIR, exist_ok = True)

FINAL_IMAGES_DIR = 'final_images'
os.makedirs(FINAL_IMAGES_DIR, exist_ok = True)


COMFYUI_URL = "127.0.0.1:8188"
COMFYUI_HTTP_URL = "http://" + COMFYUI_URL
WORKFLOW_API_JSON_PATH = "CheckInt.json"
LOAD_IMAGE_NODE_ID = "16"
SAVE_IMAGE_NODE_ID = "35"

FTP_HOST = os.getenv("FTP_HOST")
FTP_USER = os.getenv("FTP_USER")
FTP_PASS = os.getenv("FTP_PASS")
FTP_TARGET_DIR = os.getenv("FTP_TARGET_DIR")
BASE_PUBLIC_URL = os.getenv("BASE_PUBLIC_URL")

def queue_prompt(prompt_workflow, client_id):
    payload = {
        "prompt": prompt_workflow,
        "client_id": client_id
    }

    try:
        response = requests.post(f"{COMFYUI_HTTP_URL}/prompt", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error queuing prompt: {e}")
        return None
    
def upload_image_to_comfyui(image_path, filename):
    try:
        with open(image_path, 'rb') as f:
            files = { 'image': (filename, f, 'image/png')}
            data = { 'overwrite': 'true'}
            response = requests.post(f"{COMFYUI_HTTP_URL}/upload/image", files = files, data = data)
            response.raise_for_status()
            print("Image uploaded to ComfyUI successfully.")
            return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error uploading image to ComfyUI: {e}")
        return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    
def get_image_from_comfyui(filename, subfolder, folder_type):
    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }
    url = f"{COMFYUI_HTTP_URL}/view?{urllib.parse.urlencode(params)}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image from ComfyUI: {e}")
        return None
    
def get_final_image_path_from_websocket(prompt_id, client_id):
    ws_url = f"ws://{COMFYUI_URL}/ws?clientId={client_id}"
    ws = websocket.create_connection(ws_url)

    print(f"Waiting for prompt {prompt_id} to fininshed...")

    try:
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message['type'] == 'executed' and message['data']['prompt_id'] == prompt_id:
                    print(f"Prompt {prompt_id} finished execution.")

                    output_data = message['data']['output']['images']
                    if not output_data:
                        print("Execution finished, but no images were found in the output.")
                        return None
                    
                    first_image_data = output_data[0]
                    image_content = get_image_from_comfyui(
                        first_image_data['filename'],
                        first_image_data['subfolder'],
                        first_image_data['type']
                    )

                    if image_content:
                        final_filename = first_image_data['filename']
                        final_image_path = os.path.join(FINAL_IMAGES_DIR, final_filename)
                        with open(final_image_path, 'wb') as f:
                            f.write(image_content)

                        absolute_path = os.path.abspath(final_image_path)
                        print(f"Final image saved to: {absolute_path}")
                        return final_filename
    finally:
        ws.close()
    return None

def upload_final_image_to_ftp(file_path, filename):
    print(f"Attempting to upload {filename} to FTP server at {FTP_HOST}...")

    if not all([
        FTP_HOST, 
        FTP_USER, 
        FTP_PASS, 
        FTP_TARGET_DIR
        ]):
        error_msg = "FTP configuration is missing. Please check your .env file"
        print(f"Error: {error_msg}")
        return False, error_msg
    
    try:
        ftp = ftplib.FTP(FTP_HOST, FTP_USER, FTP_PASS)
        ftp.cwd(FTP_TARGET_DIR)
        with open (file_path, 'rb') as f:
            ftp.storbinary(f'STOR {filename}', f)
        ftp.quit()

        print(f"File {filename} uploaded to FTP successfully.")
        return True, f"File '{filename}' uploaded to FTP successfully!"
    except Exception as e:
        print(f"An error occurred during FTP uploaded: {e}")
        return False, str(e)


@app.route('/merge_images', methods=['POST'])
def merge_images():
    if 'foreground_file' not in request.files or 'background_file' not in request.files:
        return jsonify({ "error": "Missing one or both files in the request"}), 400
    
    foreground_file = request.files['foreground_file']
    background_file = request.files['background_file']

    if foreground_file.filename == '' or background_file.filename == '':
        return jsonify({ "error": "One or both files are not selected"}), 400
    
    try:
        foreground_image = Image.open(foreground_file.stream)
        background_image = Image.open(background_file.stream)

        foreground_array = np.array(foreground_image)
        foreground_no_bg_array = rembg.remove(foreground_array)

        foreground_no_bg = Image.fromarray(foreground_no_bg_array)

        background_width, background_height = background_image.size
        fg_width, fg_height = foreground_no_bg.size

        aspect_ratio = fg_height / fg_width
        new_foreground_width = background_width
        new_foreground_height = int(new_foreground_width * aspect_ratio)

        resized_foreground = foreground_no_bg.resize(
            (new_foreground_width, new_foreground_height),
            Image.Resampling.LANCZOS
        )

        pos_x = (background_width - new_foreground_width) // 2
        pos_y = background_height - new_foreground_height
        position = (pos_x, pos_y)

        background_image.paste(resized_foreground, position, resized_foreground)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
        merged_filename = f"merged_result_{timestamp}.png"
        merged_path = os.path.join(OUTPUT_DIR, merged_filename)
        background_image.save(merged_path, 'PNG')
        merged_absolute_path = os.path.abspath(merged_path)
        print(f"Image saved successfully to: {merged_absolute_path}")

        upload_response = upload_image_to_comfyui(merged_path, merged_filename)
        if not upload_response or 'name' not in upload_response:
            return jsonify({
                "error": "Failed to upload merge image to ComfyUI"
            }), 500
        
        comfyui_filename = upload_response['name']

        with open(WORKFLOW_API_JSON_PATH, 'r') as f:
            prompt_workflow = json.load(f)

        prompt_workflow[LOAD_IMAGE_NODE_ID]['inputs']['image'] = comfyui_filename

        output_prefix = f"checkint_{timestamp}"
        prompt_workflow[SAVE_IMAGE_NODE_ID]['inputs']['filename_prefix'] = output_prefix

        client_id = str(uuid.uuid4())
        queue_response = queue_prompt(prompt_workflow, client_id)
        if not queue_response or 'prompt_id' not in queue_response:
            return jsonify({
                "error": "Failed to queue prompt in ComfyUI"
            }), 500
        
        prompt_id = queue_response['prompt_id']
        final_filename = get_final_image_path_from_websocket(prompt_id, client_id)

        if final_filename:
            final_image_path = os.path.join(FINAL_IMAGES_DIR, final_filename)
            success, message = upload_final_image_to_ftp(final_image_path, final_filename)

            if success:
                if not BASE_PUBLIC_URL:
                    return jsonify({
                        "error": "BASE_PUBLIC_URL is not set in .env file"
                    }), 500
                
                final_image_url = f"{BASE_PUBLIC_URL}{final_filename}"

                return jsonify({
                    "message": "Workflow complete. File uploaded successfully.",
                    "final_image_url": final_image_url
                }), 200
            else:
                return jsonify({
                    "message": "Workflow completed, but failed to upload image to FTP server.",
                    "error": "Could not upload the final image via FTP.",
                    "final_image_filename": final_filename,
                    "ftp_error_details": message
                }), 500
        else:
            return jsonify({
                "error": "Workflow completed but could not retrived final image from ComfyUI"
            }), 500
    
    except Exception as exception:
        print(f"!!! ANn unexpected error occurred: {exception}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "An internal server error occurred ",
            "details": str(exception)
        }), 500

@app.route('/final_image/<filename>')
def get_final_image(filename):
    try:
        return send_from_directory(FINAL_IMAGES_DIR, filename, as_attachment = False)
    except FileNotFoundError:
        return jsonify({
            "error": "File not found."
        }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)