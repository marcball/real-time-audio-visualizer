import numpy as np
import pygame
import math
import threading
import time
from scipy.io import wavfile
import sys
import sounddevice as sd
from scipy.fftpack import fft
import os
from pydub import AudioSegment


# === Initialize Pygame ===
pygame.init()
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Audio Visualizer")
clock = pygame.time.Clock()

# === Font Setup ===
menu_font = pygame.font.SysFont("segoeui", 32)
control_font = pygame.font.SysFont("segoeui", 20)

# === Available Audio Files ===

# Changing this to check for any '.mp3' / '.wav' file in 'music' folder

AUDIO_FILES = []

script_dir = os.path.dirname(os.path.abspath(__file__))

for filename in os.scandir(script_dir + '/music'): # scandir() works like OOP, take .name
    if filename.name.lower().endswith('wav'):
        AUDIO_FILES.append(filename.path)

    """
    elif filename.name.lower().endswith('.mp3'):
        # If it's an .mp3: pygame has limited support
        # Using 'pydub' to convert .mp3 -> .wav

        mp3_path = filename.path # path to .mp3 file
        wav_path = os.path.splitext(mp3_path)[0] + '.wav' # Removing .mp3 // adding .wav

        mp3_song = AudioSegment.from_mp3(mp3_path)
        mp3_song.export(wav_path, format="wav")

        AUDIO_FILES.append(wav_path)

        THIS WOULDN'T WORK. WE NEED 'FFmpeg' to convert. :(
    """





# === Constants ===
BUFFER_SIZE = 1024
fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.set_alpha(80)  # Faster fade
fade_surface.fill((5, 5, 10))  # Darker fade to prevent ghosting
triangle = [(CENTER[0], CENTER[1] + 100), (CENTER[0] - 100, CENTER[1] - 80), (CENTER[0] + 100, CENTER[1] - 80)]

# === FFT Globals ===
fft_values = np.zeros(BUFFER_SIZE // 2)
fft_lock = threading.Lock()
current_pos = [0]
running = True
shape_mode = 2  # 0=circle, 1=spiral, 2=triangle
stream = None
prev_fft = np.zeros(BUFFER_SIZE // 2)


# === Color Functions ===
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def get_blended_color(angle_ratio):
    palette = [(255, 0, 0), (255, 69, 0), (255, 255, 0), (0, 0, 255), (138, 43, 226)]
    n = len(palette)
    idx = int(angle_ratio * n) % n
    t = (angle_ratio * n) - idx
    return lerp_color(palette[idx], palette[(idx + 1) % n], t)

def clamp_color(color):
    return tuple(min(255, max(0, c)) for c in color)

# === Audio Stream Functions ===
def audio_callback(indata, frames, time, status):
    global fft_values, prev_fft

    if status:
        print(status)

    audio_data = np.mean(indata, axis=1)
    audio_data -= np.mean(audio_data)



    fft_result = np.abs(fft(audio_data))[:BUFFER_SIZE // 2]
    fft_result = np.log1p(fft_result)           # Compress peaks
    fft_result[0] = 0                           # Suppress DC/low freq
    fft_result = fft_result / np.max(fft_result + 1e-6)


    fft_result = np.convolve(fft_result, np.ones(3)/3, mode='same')

    alpha = 0.9 # Higher = smoother & slower
    fft_smoothed = alpha * prev_fft + (1 - alpha) * fft_result
    prev_fft[:] = fft_smoothed

    with fft_lock:
        fft_values[:] = fft_smoothed

def start_microphone_stream():
    global stream
    try:
        stream = sd.InputStream(samplerate=44100, channels=1, callback=audio_callback, blocksize=BUFFER_SIZE)
        stream.start()
        return True
    except Exception as e:
        print(f"Microphone error: {e}")
        return False

def stop_microphone_stream():
    global stream
    if stream:
        stream.stop()
        stream.close()
        stream = None

# === Menu UI ===
def main_menu():
    global running, shape_mode

    # Returns name of song - .wav
    display_names = [os.path.splitext(os.path.basename(path))[0] for path in AUDIO_FILES] 
    
    options = display_names + ["Real-time Microphone Input"]
    selected = 0

    while running:
        screen.fill((0, 0, 30))
        title = menu_font.render("Select an audio source:", True, (255, 255, 255))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

        mouse_x, mouse_y = pygame.mouse.get_pos()

        for i, option in enumerate(options):
            option_rect = pygame.Rect(WIDTH // 2 - 200, 160 + i * 40, 400, 36)
            hovered = option_rect.collidepoint(mouse_x, mouse_y)
            if hovered:
                selected = i
            color = (255, 255, 0) if i == selected else (180, 180, 180)
            label = menu_font.render(option, True, color)
            screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 160 + i * 40))

        # Control instructions
        controls = " |  Space: Change Shape  |  ESC: Back/Quit  | "
        control_text = control_font.render(controls, True, (150, 150, 150))
        screen.blit(control_text, (WIDTH // 2 - control_text.get_width() // 2, HEIGHT - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if selected < len(AUDIO_FILES):
                        visualize_track(AUDIO_FILES[selected])
                    else:
                        if start_microphone_stream():
                            visualize_realtime()
                elif event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if selected < len(AUDIO_FILES):
                        visualize_track(AUDIO_FILES[selected])
                    else:
                        if start_microphone_stream():
                            visualize_realtime()

# === Visualization Logic ===
def visualize_track(filename):
    global fft_values, current_pos, running
    try:
        sample_rate, data = wavfile.read(filename)
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return

    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data / np.max(np.abs(data))
    total_samples = len(data)

    current_pos = [0]
    fft_values = np.zeros(BUFFER_SIZE // 2)

    pygame.mixer.init(frequency=sample_rate)
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    def fft_thread():
        while pygame.mixer.music.get_busy() and running:
            with fft_lock:
                start = current_pos[0]
                end = start + BUFFER_SIZE
                if end > total_samples:
                    break
                window = data[start:end] * np.hanning(BUFFER_SIZE)
                window -= np.mean(window)

                spectrum = np.abs(np.fft.fft(window))[:BUFFER_SIZE // 2]
                fade_bins = 25  # First 10 bins will be scaled from 0 to 1
                fade = np.ones_like(spectrum)
                fade[:fade_bins] = np.power(np.linspace(0.0, 1.0, fade_bins), 3)
                spectrum *= fade

                spectrum = np.log1p(spectrum)              # Compress dynamic range
                spectrum /= np.max(spectrum + 1e-6)
                spectrum *= 0.3


                spectrum = np.convolve(spectrum, np.ones(2) / 2, mode='same')
                fft_values[:] = spectrum
                current_pos[0] += BUFFER_SIZE
            time.sleep(BUFFER_SIZE / sample_rate)

    threading.Thread(target=fft_thread, daemon=True).start()
    run_visualizer()
    pygame.mixer.music.stop()

def visualize_realtime():
    run_visualizer()
    stop_microphone_stream()

def run_visualizer():
    global running, shape_mode

    while running:
        screen.blit(fade_surface, (0, 0))
        num_points = 512
        base_radius = 100
        points = []
        fade_bins = 25

        with fft_lock:
            smoothed_fft = np.convolve(fft_values, np.ones(3)/3, mode='same')
            active_fft = smoothed_fft[fade_bins:]
            num_points = len(active_fft)

            # Resample to match num_points around the shape
            resampled_fft = np.interp(
                np.linspace(0, len(smoothed_fft) - 1, num_points),
                np.arange(len(smoothed_fft)),
                smoothed_fft
            )

            for i in range(num_points):
                angle = 2 * math.pi * (i / num_points)
                amplitude = active_fft[i] ** 0.7 # DISPERSED IT MORE
                amplitude *= 0.42

                if shape_mode == 0:  # Circle
                    radius = base_radius + amplitude * 300
                    x = CENTER[0] + radius * math.cos(angle)
                    y = CENTER[1] + radius * math.sin(angle)

                elif shape_mode == 1:  # Heart shape
                    t = (i / num_points) * 2 * math.pi
                    heart_x = 16 * math.sin(t) ** 3
                    heart_y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)

                    scale = 10 * (1 + amplitude * 2)  # Size and amplitude response
                    x = CENTER[0] + heart_x * scale
                    y = CENTER[1] - heart_y * scale  # Invert y for screen coordinates

                elif shape_mode == 2:  # Triangle
                    band = i / num_points
                    if band < 1/3:
                        t = band * 3
                        x = triangle[0][0] + (triangle[1][0] - triangle[0][0]) * t
                        y = triangle[0][1] + (triangle[1][1] - triangle[0][1]) * t
                    elif band < 2/3:
                        t = (band - 1/3) * 3
                        x = triangle[1][0] + (triangle[2][0] - triangle[1][0]) * t
                        y = triangle[1][1] + (triangle[2][1] - triangle[1][1]) * t
                    else:
                        t = (band - 2/3) * 3
                        x = triangle[2][0] + (triangle[0][0] - triangle[2][0]) * t
                        y = triangle[2][1] + (triangle[0][1] - triangle[2][1]) * t
                    dx, dy = x - CENTER[0], y - CENTER[1]
                    scale = 1 + amplitude * 2
                    x = CENTER[0] + dx * scale
                    y = CENTER[1] + dy * scale

                color = get_blended_color(i / num_points) # Change this check Marc commit 1 if like old >
                points.append((x, y, color, amplitude))

        if len(points) > 1:
            points.append(points[0])

        for i in range(1, len(points)):
            x1, y1, c1, a1 = points[i - 1]
            x2, y2, c2, a2 = points[i]
            glow = clamp_color(tuple(min(255, c + int(a2 * 350)) for c in c2))
            core = clamp_color(c2)
            pygame.draw.line(screen, glow, (x1, y1), (x2, y2), 6)
            pygame.draw.line(screen, core, (x1, y1), (x2, y2), max(1, int(a2 * 10)))

        # Control instructions inside visualizer
        controls = " |  Space: Change Shape  |  ESC: Back/Quit  | "
        control_text = control_font.render(controls, True, (180, 180, 180))
        screen.blit(control_text, (WIDTH // 2 - control_text.get_width() // 2, HEIGHT - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    shape_mode = (shape_mode + 1) % 3
                elif event.key == pygame.K_ESCAPE:
                    return

        clock.tick(60)

# === Start Program ===
if __name__ == "__main__":
    try:
        main_menu()
    finally:
        stop_microphone_stream()
        pygame.mixer.quit()
        pygame.quit()
