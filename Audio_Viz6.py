# Sam Lato CSC567 Project Real-Time Audio Visualizer Concept

import numpy as np
import pygame
import sounddevice as sd
from scipy.fftpack import fft
import math

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Audio Settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 1024
fft_values = np.zeros(BUFFER_SIZE // 2)

def audio_callback(indata, frames, time, status):
    global fft_values
    if status:
        print(status)
    audio_data = np.mean(indata, axis=1)
    fft_result = np.abs(fft(audio_data))[:BUFFER_SIZE // 2]
    fft_result = fft_result / np.max(fft_result + 1e-6)
    smoothed = np.convolve(fft_result, np.ones(2)/2, mode='same')
    fft_values = smoothed

# Start Audio Stream
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BUFFER_SIZE)
stream.start()

def lerp_color(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )

def get_blended_color(angle_ratio):
    palette = [
        (255, 0, 0),       # Red
        (255, 69, 0),      # Hot Orange
        (255, 255, 0),     # Neon Yellow
        (0, 0, 255),       # Blue
        (138, 43, 226),    # Purple
    ]
    num_colors = len(palette)
    segment = angle_ratio * num_colors
    index = int(segment) % num_colors
    t = segment - index
    c1 = palette[index]
    c2 = palette[(index + 1) % num_colors]
    return lerp_color(c1, c2, t)

fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.set_alpha(30)
fade_surface.fill((10, 10, 20))

def clamp_color(color):
    return tuple(min(255, max(0, c)) for c in color)

# Smaller triangle (fits nicely in center)
triangle = [
    (CENTER[0], CENTER[1] + 100),    # Bass (Bottom)
    (CENTER[0] - 100, CENTER[1] - 80),  # Treble (Left)
    (CENTER[0] + 100, CENTER[1] - 80),  # Mids (Right)
]

shape_mode = 2  # Start directly in triangle mode (optional)

running = True
while running:
    screen.blit(fade_surface, (0, 0))
    num_points = len(fft_values)
    base_radius = 100

    log_min = np.log10(1)
    log_max = np.log10(num_points)

    points = []

    for i in range(num_points):
        log_i = np.log10(i + 1)
        angle_ratio = (log_i - log_min) / (log_max - log_min)
        angle = (2 * math.pi) * (i / num_points)
        amplitude = fft_values[i] if i < len(fft_values) else 0

        if shape_mode == 0:  # Circle
            radius = base_radius + amplitude * 300
            x = CENTER[0] + radius * math.cos(angle)
            y = CENTER[1] + radius * math.sin(angle)

        elif shape_mode == 1:  # Spiral
            spiral_radius = base_radius + i * 0.3 + amplitude * 120
            x = CENTER[0] + spiral_radius * math.cos(angle)
            y = CENTER[1] + spiral_radius * math.sin(angle)

        elif shape_mode == 2:  # Triangle shape
            band = i / num_points
            if band < 1/3:  # Bass → vertex 0 to vertex 1
                t = band * 3
                x = triangle[0][0] + (triangle[1][0] - triangle[0][0]) * t
                y = triangle[0][1] + (triangle[1][1] - triangle[0][1]) * t
            elif band < 2/3:  # Mids → vertex 1 to vertex 2
                t = (band - 1/3) * 3
                x = triangle[1][0] + (triangle[2][0] - triangle[1][0]) * t
                y = triangle[1][1] + (triangle[2][1] - triangle[1][1]) * t
            else:  # Treble → vertex 2 to vertex 0
                t = (band - 2/3) * 3
                x = triangle[2][0] + (triangle[0][0] - triangle[2][0]) * t
                y = triangle[2][1] + (triangle[0][1] - triangle[2][1]) * t

            # Push outward based on amplitude
            dx = x - CENTER[0]
            dy = y - CENTER[1]
            scale = 1 + amplitude * 2
            x = CENTER[0] + dx * scale
            y = CENTER[1] + dy * scale

        color = get_blended_color(angle_ratio)
        points.append((x, y, color, amplitude))

    if len(points) > 1:
        points.append(points[0])  # Close loop

    for i in range(1, len(points)):
        x1, y1, color1, amp1 = points[i - 1]
        x2, y2, color2, amp2 = points[i]

        glow_intensity = int(min(255, amp2 * 350))
        glow_color = clamp_color(tuple(min(255, max(0, c + glow_intensity)) for c in color2))
        pygame.draw.line(screen, glow_color, (x1, y1), (x2, y2), 6)

        core_intensity = max(1, int(amp2 * 10))
        color2_clamped = clamp_color(color2)
        pygame.draw.line(screen, color2_clamped, (x1, y1), (x2, y2), core_intensity)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                shape_mode = (shape_mode + 1) % 3

    clock.tick(60)

pygame.quit()
stream.stop()
stream.close()
