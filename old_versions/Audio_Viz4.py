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
    audio_data = np.mean(indata, axis=1)  # mono
    fft_result = np.abs(fft(audio_data))[:BUFFER_SIZE // 2]
    fft_values = fft_result / np.max(fft_result + 1e-6)  # normalize

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
        (0, 0, 255),       # Blue
        (138, 43, 226),    # Purple
        (255, 69, 0),      # Hot Orange
        (255, 255, 0),     # Neon Yellow
    ]
    num_colors = len(palette)
    segment = angle_ratio * num_colors
    index = int(segment) % num_colors
    t = segment - index
    c1 = palette[index]
    c2 = palette[(index + 1) % num_colors]
    return lerp_color(c1, c2, t)

# Motion trail fade surface
fade_surface = pygame.Surface((WIDTH, HEIGHT))
fade_surface.set_alpha(40)  # Trail length (lower is longer)
fade_surface.fill((10, 10, 20))

running = True
while running:
    screen.blit(fade_surface, (0, 0))  # Leave trails

    num_points = BUFFER_SIZE // 4
    base_radius = 150

    first_point = None
    first_color = None

    for i in range(num_points):
        angle = (2 * math.pi / num_points) * i
        amplitude = fft_values[i] if i < len(fft_values) else 0

        radius = base_radius + amplitude * 200

        x = CENTER[0] + radius * math.cos(angle)
        y = CENTER[1] + radius * math.sin(angle)

        angle_ratio = i / num_points
        color = get_blended_color(angle_ratio)

        if i == 0:
            first_point = (x, y)
            first_color = color
        else:
            angle_prev = (2 * math.pi / num_points) * (i - 1)
            prev_radius = base_radius + fft_values[i - 1] * 200
            x_prev = CENTER[0] + prev_radius * math.cos(angle_prev)
            y_prev = CENTER[1] + prev_radius * math.sin(angle_prev)

            # Draw outer glow
            glow_color = tuple(min(255, c + 60) for c in color)
            pygame.draw.line(screen, glow_color, (x_prev, y_prev), (x, y), 6)

            # Draw vivid core line
            pygame.draw.line(screen, color, (x_prev, y_prev), (x, y), 3)

    # ✅ Connect the last point to the first to close the circle
    if first_point:
        last_angle = (2 * math.pi / num_points) * (num_points - 1)
        last_radius = base_radius + fft_values[num_points - 1] * 200
        last_x = CENTER[0] + last_radius * math.cos(last_angle)
        last_y = CENTER[1] + last_radius * math.sin(last_angle)

        glow_color = tuple(min(255, c + 60) for c in first_color)
        pygame.draw.line(screen, glow_color, (last_x, last_y), first_point, 6)
        pygame.draw.line(screen, first_color, (last_x, last_y), first_point, 3)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(60)

pygame.quit()
stream.stop()
stream.close()
