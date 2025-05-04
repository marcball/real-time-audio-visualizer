import numpy as np
import pygame
import sounddevice as sd
from scipy.fftpack import fft
import math
import random

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

def get_color(t):
    """Smooth gradient from hue."""
    r = int(127 * math.sin(t) + 128)
    g = int(127 * math.sin(t + 2) + 128)
    b = int(127 * math.sin(t + 4) + 128)
    return (r, g, b)

running = True
while running:
    screen.fill((10, 10, 20))

    num_points = BUFFER_SIZE // 4
    base_radius = 150

    for i in range(num_points):
        angle = (2 * math.pi / num_points) * i
        amplitude = fft_values[i] if i < len(fft_values) else 0

        radius = base_radius + amplitude * 200

        x = CENTER[0] + radius * math.cos(angle)
        y = CENTER[1] + radius * math.sin(angle)

        size = max(2, int(amplitude * 15))
        color = get_color(i * 0.1 + pygame.time.get_ticks() * 0.001)

        # Draw a circle for each frequency point
        pygame.draw.circle(screen, color, (int(x), int(y)), size)

        # Optional: connect points with lines
        if i > 0:
            angle_prev = (2 * math.pi / num_points) * (i - 1)
            prev_radius = base_radius + fft_values[i - 1] * 200
            x_prev = CENTER[0] + prev_radius * math.cos(angle_prev)
            y_prev = CENTER[1] + prev_radius * math.sin(angle_prev)
            pygame.draw.line(screen, color, (x_prev, y_prev), (x, y), 2)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(60)

pygame.quit()
stream.stop()
stream.close()
