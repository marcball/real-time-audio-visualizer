# Sam Lato CSC567 Project – Audio File Visualizer (WAV, Pygame, NumPy, FFT)

import numpy as np
import pygame
import math
import threading
import time
from scipy.io import wavfile
import sys

#=== Initialize Pygame ===#
pygame.init()
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Audio Visualizer")
clock = pygame.time.Clock()

#=== Font Setup ===#
menu_font = pygame.font.SysFont("segoeui", 32)
control_font = pygame.font.SysFont("segoeui", 20)

#=== Available Audio Files ===#
AUDIO_FILES = [
    "watermarked_Ghost_Beatz_Penn_Station.wav",
    "watermarked_Mayr_Make_It_Out.wav",
    "watermarked_Tony_Sopiano_Sour_Peachez.wav"
]

#=== Constants ===#
BUFFER_SIZE = 1024
fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.set_alpha(30)
fade_surface.fill((10, 10, 20))
triangle = [(CENTER[0], CENTER[1] + 100), (CENTER[0] - 100, CENTER[1] - 80), (CENTER[0] + 100, CENTER[1] - 80)]

#=== FFT Globals ===#
fft_values = np.zeros(BUFFER_SIZE // 2)
fft_lock = threading.Lock()
current_pos = [0]
running = True

#=== Color Functions ===#
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

#=== Menu UI ===#
def main_menu():
    selected = 0
    while running:
        screen.fill((0, 0, 30))
        title = menu_font.render("Select a track (↑/↓ then Enter, ESC to quit)", True, (255, 255, 255))
        screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 80))

        for i, file in enumerate(AUDIO_FILES):
            color = (255, 255, 0) if i == selected else (180, 180, 180)
            label = menu_font.render(file, True, color)
            screen.blit(label, (WIDTH // 2 - label.get_width() // 2, 160 + i * 40))

        # Control instructions
        controls = " ↑/↓: Navigate  |  Enter: Select  |  Space: Change Shape  |  ESC: Back/Quit "
        control_text = control_font.render(controls, True, (150, 150, 150))
        screen.blit(control_text, (WIDTH // 2 - control_text.get_width() // 2, HEIGHT - 40))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(AUDIO_FILES)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(AUDIO_FILES)
                elif event.key == pygame.K_RETURN:
                    visualize_track(AUDIO_FILES[selected])
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

#=== Visualization Logic ===#
def visualize_track(filename):
    global fft_values, current_pos
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

    shape_mode = 2  # Start with triangle

    def fft_thread():
        while pygame.mixer.music.get_busy():
            with fft_lock:
                start = current_pos[0]
                end = start + BUFFER_SIZE
                if end > total_samples:
                    break
                window = data[start:end] * np.hanning(BUFFER_SIZE)
                spectrum = np.abs(np.fft.fft(window))[:BUFFER_SIZE // 2]
                spectrum /= np.max(spectrum + 1e-6)
                spectrum = np.convolve(spectrum, np.ones(2) / 2, mode='same')
                fft_values[:] = spectrum
                current_pos[0] += BUFFER_SIZE
            time.sleep(BUFFER_SIZE / sample_rate)

    threading.Thread(target=fft_thread, daemon=True).start()

    while pygame.mixer.music.get_busy():
        screen.blit(fade_surface, (0, 0))
        num_points = len(fft_values)
        base_radius = 100
        log_min, log_max = np.log10(1), np.log10(num_points)
        points = []

        with fft_lock:
            for i in range(num_points):
                log_i = np.log10(i + 1)
                angle_ratio = (log_i - log_min) / (log_max - log_min)
                angle = 2 * math.pi * (i / num_points)
                amplitude = fft_values[i]

                if shape_mode == 0:  # Circle
                    radius = base_radius + amplitude * 300
                    x = CENTER[0] + radius * math.cos(angle)
                    y = CENTER[1] + radius * math.sin(angle)

                elif shape_mode == 1:  # Spiral
                    spiral_radius = base_radius + i * 0.3 + amplitude * 120
                    x = CENTER[0] + spiral_radius * math.cos(angle)
                    y = CENTER[1] + spiral_radius * math.sin(angle)

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

                color = get_blended_color(angle_ratio)
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

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.mixer.music.stop()
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    shape_mode = (shape_mode + 1) % 3
                elif event.key == pygame.K_ESCAPE:
                    pygame.mixer.music.stop()
                    return  # Go back to menu

        clock.tick(60)

#=== Start Program ===#
if __name__ == "__main__":
    main_menu()
