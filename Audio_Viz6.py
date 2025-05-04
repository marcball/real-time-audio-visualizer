#CSC567 Project Real-Time Audio Visualizer Concept

import numpy as np
import pygame
import sounddevice as sd
from scipy.fftpack import fft
import math

#Initialize Pygame and screen dimensions
pygame.init()
WIDTH, HEIGHT = 800, 600  #Set screen size
CENTER = (WIDTH // 2, HEIGHT // 2)  #Compute center point for drawing reference
screen = pygame.display.set_mode((WIDTH, HEIGHT))  #Create window
clock = pygame.time.Clock()  #Create clock for framerate control

#Audio Settings and FFT buffer setup
SAMPLE_RATE = 44100  #Standard CD-quality sample rate
BUFFER_SIZE = 1024  #Number of samples per audio block
fft_values = np.zeros(BUFFER_SIZE // 2)  #Initialize FFT result buffer

#Audio callback function triggered in real-time as mic data streams in
def audio_callback(indata, frames, time, status):
    global fft_values
    if status:
        print(status)  #Log any errors or stream warnings
    audio_data = np.mean(indata, axis=1)  #Convert multi-channel audio to mono
    fft_result = np.abs(fft(audio_data))[:BUFFER_SIZE // 2]  #Compute magnitude spectrum and ignore mirrored half
    fft_result = fft_result / np.max(fft_result + 1e-6)  #Normalize to range [0, 1] with epsilon to prevent /0
    smoothed = np.convolve(fft_result, np.ones(2)/2, mode='same')  #Smooth FFT values for less jitter
    fft_values = smoothed  #Store processed FFT values globally for visualization

#Start real-time microphone stream using sounddevice
stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BUFFER_SIZE)
stream.start()  #Begin streaming audio input

#Function to interpolate between two RGB colors
def lerp_color(color1, color2, t):
    return (
        int(color1[0] + (color2[0] - color1[0]) * t),
        int(color1[1] + (color2[1] - color1[1]) * t),
        int(color1[2] + (color2[2] - color1[2]) * t),
    )

#Function to blend between colors in a circular palette based on position
def get_blended_color(angle_ratio):
    palette = [
        (255, 0, 0),       #Red
        (255, 69, 0),      #Hot Orange
        (255, 255, 0),     #Neon Yellow
        (0, 0, 255),       #Blue
        (138, 43, 226),    #Purple
    ]
    num_colors = len(palette)
    segment = angle_ratio * num_colors  #Determine location between color segments
    index = int(segment) % num_colors  #Get base color index
    t = segment - index  #Fractional offset between two adjacent colors
    c1 = palette[index]
    c2 = palette[(index + 1) % num_colors]
    return lerp_color(c1, c2, t)

#Create transparent surface for fade effect to simulate trails
fade_surface = pygame.Surface((WIDTH, HEIGHT))  #Create surface matching screen size
fade_surface.set_alpha(30)  #Low alpha for fading old frames
fade_surface.fill((10, 10, 20))  #Dark blueish tint

#Ensure RGB values stay within 0–255
def clamp_color(color):
    return tuple(min(255, max(0, c)) for c in color)

#Triangle layout used for triangle mode visualization
triangle = [
    (CENTER[0], CENTER[1] + 100),    #Bass vertex (bottom of triangle)
    (CENTER[0] - 100, CENTER[1] - 80),  #Treble vertex (left top)
    (CENTER[0] + 100, CENTER[1] - 80),  #Mids vertex (right top)
]

shape_mode = 2  #0=circle, 1=spiral, 2=triangle (start in triangle)

running = True
while running:
    screen.blit(fade_surface, (0, 0))  #Overlay fading surface to create motion blur
    num_points = len(fft_values)  #Determine number of frequency bins
    base_radius = 100  #Base radius or distance from center for shape drawing

    log_min = np.log10(1)
    log_max = np.log10(num_points)  #Used for logarithmic scaling of low-to-high frequencies

    points = []  #Store visualized points for this frame

    for i in range(num_points):
        log_i = np.log10(i + 1)
        angle_ratio = (log_i - log_min) / (log_max - log_min)  #Normalized log-scale ratio
        angle = (2 * math.pi) * (i / num_points)  #Angle around circle
        amplitude = fft_values[i] if i < len(fft_values) else 0  #Amplitude of frequency bin

        if shape_mode == 0:  #Circle mode visualization
            radius = base_radius + amplitude * 300  #Modulate radius based on amplitude
            x = CENTER[0] + radius * math.cos(angle)
            y = CENTER[1] + radius * math.sin(angle)

        elif shape_mode == 1:  #Spiral mode visualization
            spiral_radius = base_radius + i * 0.3 + amplitude * 120  #Spiral out over bins
            x = CENTER[0] + spiral_radius * math.cos(angle)
            y = CENTER[1] + spiral_radius * math.sin(angle)

        elif shape_mode == 2:  #Triangle mode visualization
            band = i / num_points  #Normalize to [0, 1] to find edge segment
            if band < 1/3:  #Segment from vertex 0 to 1 (bass range)
                t = band * 3
                x = triangle[0][0] + (triangle[1][0] - triangle[0][0]) * t
                y = triangle[0][1] + (triangle[1][1] - triangle[0][1]) * t
            elif band < 2/3:  #Segment from vertex 1 to 2 (mids range)
                t = (band - 1/3) * 3
                x = triangle[1][0] + (triangle[2][0] - triangle[1][0]) * t
                y = triangle[1][1] + (triangle[2][1] - triangle[1][1]) * t
            else:  #Segment from vertex 2 to 0 (treble range)
                t = (band - 2/3) * 3
                x = triangle[2][0] + (triangle[0][0] - triangle[2][0]) * t
                y = triangle[2][1] + (triangle[0][1] - triangle[2][1]) * t

            dx = x - CENTER[0]
            dy = y - CENTER[1]
            scale = 1 + amplitude * 2  #Scale out from center based on amplitude
            x = CENTER[0] + dx * scale
            y = CENTER[1] + dy * scale

        color = get_blended_color(angle_ratio)  #Get frequency-dependent color
        points.append((x, y, color, amplitude))  #Add vertex data to point list

    if len(points) > 1:
        points.append(points[0])  #Close shape by linking last to first

    for i in range(1, len(points)):
        x1, y1, color1, amp1 = points[i - 1]
        x2, y2, color2, amp2 = points[i]

        glow_intensity = int(min(255, amp2 * 350))  #Higher amplitude yields brighter glow
        glow_color = clamp_color(tuple(min(255, max(0, c + glow_intensity)) for c in color2))
        pygame.draw.line(screen, glow_color, (x1, y1), (x2, y2), 6)  #Draw outer glow line

        core_intensity = max(1, int(amp2 * 10))  #Core line width modulated by amplitude
        color2_clamped = clamp_color(color2)
        pygame.draw.line(screen, color2_clamped, (x1, y1), (x2, y2), core_intensity)  #Draw core inner line

    pygame.display.flip()  #Render frame to display

    for event in pygame.event.get():  #Handle window events
        if event.type == pygame.QUIT:
            running = False  #Exit loop on window close
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                shape_mode = (shape_mode + 1) % 3  #Cycle between circle, spiral, and triangle modes

    clock.tick(60)  #Maintain 60 frames per second

#Clean up audio and graphics resources on exit
pygame.quit()
stream.stop()
stream.close()
