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
    fft_result = fft_result / np.max(fft_result + 1e-6)  # normalize

    # Increase sensitivity (less smoothing)
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
        (138, 43, 226),    # Purple
        (0, 0, 255),       # Blue
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
fade_surface.set_alpha(30)
fade_surface.fill((10, 10, 20))

def clamp_color(color):
    return tuple(min(255, max(0, c)) for c in color)

running = True
while running:
    screen.blit(fade_surface, (0, 0))

    num_points = len(fft_values)
    base_radius = 150

    log_min = np.log10(1)
    log_max = np.log10(num_points)

    points = []

    for i in range(num_points):
        angle = (2 * math.pi) * (i / num_points)
        log_i = np.log10(i + 1)
        angle_ratio = (log_i - log_min) / (log_max - log_min)

        amplitude = fft_values[i] if i < len(fft_values) else 0
        radius = base_radius + amplitude * 300

        x = CENTER[0] + radius * math.cos(angle)
        y = CENTER[1] + radius * math.sin(angle)

        color = get_blended_color(angle_ratio)
        points.append((x, y, color, amplitude))

    for i in range(1, len(points)):
        x1, y1, color1, amplitude1 = points[i - 1]
        x2, y2, color2, amplitude2 = points[i]

        glow_intensity = int(min(255, amplitude2 * 350))
        glow_color = clamp_color(tuple(min(255, max(0, c + glow_intensity)) for c in color2))
        pygame.draw.line(screen, glow_color, (x1, y1), (x2, y2), 6)

        core_intensity = max(1, int(amplitude2 * 10))
        color2_clamped = clamp_color(color2)
        pygame.draw.line(screen, color2_clamped, (x1, y1), (x2, y2), core_intensity)

    # Close the circle
    if len(points) > 1:
        x1, y1, color1, amplitude1 = points[-1]
        x2, y2, color2, amplitude2 = points[0]

        glow_intensity = int(min(255, amplitude2 * 350))
        glow_color = clamp_color(tuple(min(255, max(0, c + glow_intensity)) for c in color2))
        pygame.draw.line(screen, glow_color, (x1, y1), (x2, y2), 6)

        core_intensity = max(1, int(amplitude2 * 10))
        color2_clamped = clamp_color(color2)
        pygame.draw.line(screen, color2_clamped, (x1, y1), (x2, y2), core_intensity)

    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(60)

pygame.quit()
stream.stop()
stream.close()
