import os
import io
from flask import Flask, request, render_template, send_file, flash, redirect, url_for, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageOps
import threading
import time
import numpy as np
import NDIlib as ndi
import yaml

# Load config from YAML
with open('config.yaml', 'r', encoding='utf-8') as f:
    config_data = yaml.safe_load(f)

UPLOAD_FOLDER = config_data.get('UPLOAD_FOLDER', 'uploads')
OUTPUT_FOLDER = config_data.get('OUTPUT_FOLDER', 'outputs')
FONT_PATH = config_data.get('FONT_PATH', 'static/fonts/Kanit-Regular.ttf')
NDI_NAME = config_data.get('NDI_NAME', 'ImageGen_NDI')
CANVAS_WIDTH = config_data.get('CANVAS_WIDTH', 1920)
CANVAS_HEIGHT = config_data.get('CANVAS_HEIGHT', 1080)
BACKGROUND_COLOR = tuple(config_data.get('BACKGROUND_COLOR', [0, 0, 0]))
TEXT_COLOR = tuple(config_data.get('TEXT_COLOR', [255, 255, 255]))
BACKGROUND_IMAGE_PATH = config_data.get('BACKGROUND_IMAGE_PATH', 'background.jpg')
IMAGE_SCALE_FACTOR = config_data.get('IMAGE_SCALE_FACTOR', 1.5)
MAX_QUEUE_SIZE = config_data.get('MAX_QUEUE_SIZE', 30)
TIME_SHOW_IMAGE = config_data.get('TIME_SHOW_IMAGE', 20.0)
TIME_GREEN_SCREEN = config_data.get('TIME_GREEN_SCREEN', 10.0)

# Load Bad Words
BAD_WORDS = []
try:
    with open('bad_words.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip().lower()
            if word and not word.startswith('#'):
                BAD_WORDS.append(word)
except Exception as e:
    print(f"Warning: Could not load bad_words.txt: {e}")

def contains_bad_word(text):
    text_lower = text.lower()
    for word in BAD_WORDS:
        if word in text_lower:
            return True
    return False

app = Flask(__name__)
app.secret_key = 'super_secret_key'  # Needed for flash messages

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- NDI State and Thread ---
ndi_lock = threading.Lock()
image_queue = []
ndi_state = 'GREEN' # 'GREEN' or 'SHOWING'
state_transition_time = 0.0

def ndi_stream_loop():
    global image_queue, ndi_state, state_transition_time
    
    if not ndi.initialize():
        print("Cannot run NDI.")
        return

    ndi_send_create_desc = ndi.SendCreate()
    ndi_send_create_desc.ndi_name = NDI_NAME
    ndi_send = ndi.send_create(ndi_send_create_desc)
    
    if not ndi_send:
        print("Could not create NDI sender.")
        return
        
    # Precompute green screen (ensure it is contiguous)
    green_screen = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
    green_screen[:, :, 1] = 255 # Green
    green_screen[:, :, 3] = 255 # Alpha
    green_screen = np.ascontiguousarray(green_screen)
    
    print("NDI Stream started: ImageGen_NDI")
    
    # Pre-load or generate the background texture (larger than canvas for panning)
    bg_img_path = BACKGROUND_IMAGE_PATH
    bg_w, bg_h = CANVAS_WIDTH + 300, CANVAS_HEIGHT + 300
    
    if os.path.exists(bg_img_path):
        try:
            bg_pil = Image.open(bg_img_path).convert('RGB')
            bg_pil = bg_pil.resize((bg_w, bg_h), Image.Resampling.LANCZOS)
            bg_texture = np.array(bg_pil)
        except Exception as e:
            print(f"Error loading {bg_img_path}: {e}")
            bg_texture = None
    else:
        bg_texture = None
        
    if bg_texture is None:
        # Generate procedural purple/pink wavy texture
        y_c = np.linspace(0, 10, bg_h).reshape(-1, 1)
        x_c = np.linspace(0, 10, bg_w).reshape(1, -1)
        noise = np.sin(x_c * 0.8) + np.cos(y_c * 0.6) + np.sin((x_c - y_c) * 0.5)
        noise = (noise + 3) / 6.0
        
        bg_texture = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        # Shift the noise offsets significantly to create large areas of pure black
        bg_texture[:, :, 0] = np.clip((noise - 0.5) * 250, 0, 255) # Pinkish Red
        bg_texture[:, :, 1] = np.clip((noise - 0.6) * 50, 0, 255)  # Low green
        bg_texture[:, :, 2] = np.clip((noise - 0.4) * 200, 0, 255) # Dark Purple Blue
        
    was_green = True
    current_frame_data = green_screen
    cached_generated_frame = None
    last_seen_image = None
    
    while True:
        current_time = time.time()
        
        with ndi_lock:
            if ndi_state == 'GREEN':
                if current_time >= state_transition_time and len(image_queue) > 0:
                    active_image = image_queue.pop(0)
                    ndi_state = 'SHOWING'
                    state_transition_time = current_time + TIME_SHOW_IMAGE
                    
                    if active_image.mode != 'RGBA':
                        active_image = active_image.convert('RGBA')
                    cached_generated_frame = np.ascontiguousarray(active_image, dtype=np.uint8)
                    print(f"NDI: Showing next image. Queue remaining: {len(image_queue)}")
            elif ndi_state == 'SHOWING':
                if current_time >= state_transition_time:
                    ndi_state = 'GREEN'
                    state_transition_time = current_time + TIME_GREEN_SCREEN
                    cached_generated_frame = None
                    print(f"NDI: Switching to Green Screen ({TIME_GREEN_SCREEN}s interval)")
                    
        if ndi_state == 'SHOWING' and cached_generated_frame is not None:
            # --- Animated Panning Background ---
            t = time.time()
            pan_x = int(150 + 140 * np.sin(t * 0.5))
            pan_y = int(150 + 140 * np.cos(t * 0.4))
            
            bg_frame = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
            bg_frame[:, :, :3] = bg_texture[pan_y:pan_y+CANVAS_HEIGHT, pan_x:pan_x+CANVAS_WIDTH]
            bg_frame[:, :, 3] = 255
            
            # Composite active image on top
            alpha = cached_generated_frame[:, :, 3:4] / 255.0
            composite = (cached_generated_frame * alpha + bg_frame * (1.0 - alpha)).astype(np.uint8)
            # Ensure alpha is fully opaque for NDI
            composite[:, :, 3] = 255
            
            current_frame_data = np.ascontiguousarray(composite)
        else:
            # Send green screen
            current_frame_data = green_screen
            
        video_frame = ndi.VideoFrameV2()
        video_frame.xres = CANVAS_WIDTH
        video_frame.yres = CANVAS_HEIGHT
        video_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_RGBA
        video_frame.line_stride_in_bytes = CANVAS_WIDTH * 4
        video_frame.data = current_frame_data
            
        ndi.send_send_video_v2(ndi_send, video_frame)
        
        # Target roughly 30 FPS
        time.sleep(0.033)

def generate_image(image_stream, text):
    # 1. Create base canvas (Transparent so we can composite over fire frame later)
    canvas = Image.new('RGBA', (CANVAS_WIDTH, CANVAS_HEIGHT), (0, 0, 0, 0))
    
    paste_x = 0
    paste_y = 0
    uploaded_image_width = 0
    uploaded_image_height = 0
    
    if image_stream:
        # 2. Open uploaded image and apply EXIF orientation
        uploaded_image = Image.open(image_stream)
        uploaded_image = ImageOps.exif_transpose(uploaded_image)
        # Force convert to RGBA to ensure alpha channel exists during paste
        uploaded_image = uploaded_image.convert('RGBA')
        
        # 3. Calculate max image size using scale factor
        max_img_width = int((CANVAS_WIDTH // 2) * IMAGE_SCALE_FACTOR)
        max_img_height = int((CANVAS_HEIGHT // 2) * IMAGE_SCALE_FACTOR)
        
        # Resize image maintaining aspect ratio
        uploaded_image.thumbnail((max_img_width, max_img_height), Image.Resampling.LANCZOS)
        
        uploaded_image_width = uploaded_image.width
        uploaded_image_height = uploaded_image.height
        
        # 4. Paste image
        if text == '':
            # Center if no text
            paste_x = (CANVAS_WIDTH - uploaded_image_width) // 2
            paste_y = (CANVAS_HEIGHT - uploaded_image_height) // 2
        else:
            paste_x = (CANVAS_WIDTH - uploaded_image_width) // 2
            paste_y = ((CANVAS_HEIGHT * 4 // 5) - uploaded_image_height) // 2 # Center within the top 4/5ths
        
        # Create edge fade mask to blend with the fire frame
        from PIL import ImageFilter, ImageChops
        fade_radius = 20
        fade_mask = Image.new('L', (uploaded_image_width, uploaded_image_height), 0)
        draw_mask = ImageDraw.Draw(fade_mask)
        draw_mask.rectangle(
            [fade_radius, fade_radius, uploaded_image_width - fade_radius, uploaded_image_height - fade_radius],
            fill=255
        )
        fade_mask = fade_mask.filter(ImageFilter.GaussianBlur(fade_radius))
        
        r, g, b, a = uploaded_image.split()
        a = ImageChops.multiply(a, fade_mask)
        uploaded_image = Image.merge('RGBA', (r, g, b, a))
        
        # Use itself as mask to preserve original transparency, and keep the rest of its bounds opaque
        canvas.paste(uploaded_image, (paste_x, paste_y), uploaded_image)
        
    # 5. Draw text
    if text != '':
        draw = ImageDraw.Draw(canvas)
        
        # Determine font size (adjust dynamically or use fixed)
        font_size = 80
        try:
            font = ImageFont.truetype(FONT_PATH, font_size)
        except IOError:
            # Fallback if font is missing
            print(f"Warning: Could not load font at {FONT_PATH}")
            font = ImageFont.load_default()
            
        # Get text bounding box for centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        if image_stream:
            # Position text at about 1/5 from the bottom of the screen
            text_x = (CANVAS_WIDTH - text_width) // 2
            text_y = CANVAS_HEIGHT - (CANVAS_HEIGHT // 5) # 1/5 from the bottom
        else:
            # Center text vertically if no image
            text_x = (CANVAS_WIDTH - text_width) // 2
            text_y = (CANVAS_HEIGHT - text_height) // 2
        
        draw.text((text_x, text_y), text, font=font, fill=TEXT_COLOR)
    
    # (Global state is now managed by the web route pushing to the queue)
    return canvas

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    file = request.files.get('image')
    text = request.form.get('text', '').strip()
    
    has_file = file and file.filename != ''
    
    if not has_file and text == '':
        return jsonify({'status': 'error', 'message': 'Please provide an image or text'})
        
    if text != '' and contains_bad_word(text):
        return jsonify({'status': 'error', 'message': 'ข้อความมีคำที่ไม่เหมาะสม กรุณาแก้ไข'})
        
    try:
        image_stream = file.stream if has_file else None
        # Generate the image in memory
        result_image = generate_image(image_stream, text)
        
        # Add to global queue
        global image_queue
        with ndi_lock:
            if len(image_queue) < MAX_QUEUE_SIZE:
                image_queue.append(result_image)
                return jsonify({'status': 'success', 'message': 'เพิ่มภาพลงคิวเรียบร้อยแล้ว!'})
            else:
                return jsonify({'status': 'error', 'message': 'คิวเต็มแล้ว (สูงสุด 30 รูป) กรุณาลองใหม่ภายหลัง'})
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred while processing the image. Please try another.'})

if __name__ == '__main__':
    # Start NDI background thread safely
    ndi_thread = threading.Thread(target=ndi_stream_loop, daemon=True)
    ndi_thread.start()
    
    # Disable reloader since NDI binaries crash on multi-process restarts
    app.run(host='0.0.0.0', debug=True, use_reloader=False, port=5000)
