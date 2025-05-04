# Sam Lato CSC567 Project – Audio File Visualizer (WAV, Pygame, NumPy, FFT)

import numpy as np
import pygame
import math
import threading
import time
from scipy.io import wavfile

#=== Load WAV file and extract audio samples ===#
FILENAME = "watermarked_Ghost_Beatz_Penn_Station.wav"  #<<< Replace with your WAV file
SAMPLE_RATE, data = wavfile.read(FILENAME)
if data.ndim > 1:
    data = data.mean(axis=1)  #Convert to mono if stereo
data = data / np.max(np.abs(data))  #Normalize to [-1, 1]
total_samples = len(data)

#=== Pygame Initialization ===#
pygame.init()
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
pygame.mixer.init(frequency=SAMPLE_RATE)
pygame.mixer.music.load(FILENAME)

#=== Settings ===#
BUFFER_SIZE = 1024
shape_mode = 2  #2=triangle to start
fft_values = np.zeros(BUFFER_SIZE // 2)

#=== Color interpolation helpers ===#
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

#=== Triangle layout for triangle mode ===#
triangle = [
    (CENTER[0], CENTER[1] + 100),
    (CENTER[0] - 100, CENTER[1] - 80),
    (CENTER[0] + 100, CENTER[1] - 80),
]

#=== Fading surface for trails ===#
fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.set_alpha(30)
fade_surface.fill((10, 10, 20))

#=== Shared buffer and FFT computation ===#
current_pos = [0]
fft_lock = threading.Lock()

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
            spectrum = np.convolve(spectrum, np.ones(2)/2, mode='same')
            fft_values[:] = spectrum
            current_pos[0] += BUFFER_SIZE
        time.sleep(BUFFER_SIZE / SAMPLE_RATE)

#=== Start playback and analysis ===#
pygame.mixer.music.play()
threading.Thread(target=fft_thread, daemon=True).start()

#=== Main loop ===#
running = True
while running:
    screen.blit(fade_surface, (0, 0))
    num_points = len(fft_values)
    base_radius = 100

    log_min = np.log10(1)
    log_max = np.log10(num_points)

    points = []

    with fft_lock:
        for i in range(num_points):
            log_i = np.log10(i + 1)
            angle_ratio = (log_i - log_min) / (log_max - log_min)
            angle = 2 * math.pi * (i / num_points)
            amplitude = fft_values[i]

            if shape_mode == 0:  #Circle
                radius = base_radius + amplitude * 300
                x = CENTER[0] + radius * math.cos(angle)
                y = CENTER[1] + radius * math.sin(angle)

            elif shape_mode == 1:  #Spiral
                spiral_radius = base_radius + i * 0.3 + amplitude * 120
                x = CENTER[0] + spiral_radius * math.cos(angle)
                y = CENTER[1] + spiral_radius * math.sin(angle)

            elif shape_mode == 2:  #Triangle
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
                dx = x - CENTER[0]
                dy = y - CENTER[1]
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
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            shape_mode = (shape_mode + 1) % 3

    clock.tick(60)

pygame.quit()
