import pygame
import logging
import random
import numpy as np
import math
import time

# Initialize pygame
pygame.init()

# Constants

WIDTH, HEIGHT = 1200, 750 # Reduced screen size for better windowed mode experience
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
FPS_MIN = 10
FPS_MAX = 240
FPS_DEFAULT = 60
ANIMAL_RADIUS = 15
FOOD_SIZE = 10
DETECTION_RANGE = 100
MUTATION_INTERVAL = 20000
ANIMAL_COUNT = 20
HUNGER_DECAY_RATE = 0.2
PASSIVE_HUNGER_DECAY = 1  # Increased to make animals more active
CARNIVORE_MOVEMENT_PENALTY = 0.5
REPRODUCTION_HUNGER_COST = 2.0
REPRODUCTION_THRESHOLD = 3.0
FOOD_SPAWN_INTERVAL = 500
CARNIVORE_EAT_HUNGER_GAIN = 2

# Setup the log file (overwrite each time the simulation starts)
logging.basicConfig(filename='simulation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s', filemode='w')


# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Removed RESIZABLE flag
pygame.display.set_caption("Evolutionary Neural Network Animals with DNA")




# Neural Network (Simple Feed-forward Network)
class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(4, 3)
        self.biases = np.random.randn(3)

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

    def mutate(self):
        mutation_strength = 0.1
        self.weights += mutation_strength * np.random.randn(*self.weights.shape)
        self.biases += mutation_strength * np.random.randn(*self.biases.shape)

    def copy_from(self, other):
        self.weights = np.copy(other.weights)
        self.biases = np.copy(other.biases)

# Animal class with DNA
# Updated Animal class with detect_food method
class Animal:
    def __init__(self, x, y, dna=None):
        self.x = x
        self.y = y
        self.radius = ANIMAL_RADIUS // 2  # Reduce size to fit larger game field
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.angle = random.uniform(0, 360)
        self.velocity = 0
        self.rotation_speed = 0
        self.nn = None
        self.hunger = 2.0
        self.fruits_eaten = 0
        self.is_carnivore = False  # Disable carnivores for now

        # DNA Attributes
        if dna:
            self.dna = dna
        else:
            self.dna = {
                "speed": random.uniform(5, 10),  # Maximum speed, smaller for fitting the larger field
                "vision_range": random.uniform(50, 150),  # Vision range now smaller for field scaling
                "eye_vision_length": random.uniform(10, 30),  # Length of vision line
                "reproduction_threshold": random.uniform(REPRODUCTION_THRESHOLD, REPRODUCTION_THRESHOLD + 2.5),
                "mutation_rate": 0.1,  # Probability of mutation affecting DNA
                "hidden_neurons": random.randint(2, 5)  # Number of hidden neurons now part of DNA
            }

        # Initialize Neural Network with updated input structure and variable hidden neurons
        self.nn = NeuralNetwork(input_size=4, hidden_size=self.dna["hidden_neurons"], output_size=3)

    def rotate(self, rotation):
        max_rotation_speed = 5
        self.rotation_speed = max(min(rotation, max_rotation_speed), -max_rotation_speed)
        self.angle += self.rotation_speed
        self.angle %= 360

    def move(self):
        max_velocity = self.dna["speed"]
        self.velocity = max(min(self.velocity, max_velocity), -max_velocity)

        radians = math.radians(self.angle)
        self.x += self.velocity * math.cos(radians)
        self.y += self.velocity * math.sin(radians)

        if self.x - self.radius < 0: self.x = self.radius
        if self.x + self.radius > WIDTH: self.x = WIDTH - self.radius
        if self.y - self.radius < 0: self.y = self.radius
        if self.y + self.radius > HEIGHT: self.y = HEIGHT - self.radius

    def detect_colors(self, screen):
        """Detect RGB colors in front of the animal to simulate vision."""
        eye_x = int(self.x + math.cos(math.radians(self.angle)) * self.dna["vision_range"])
        eye_y = int(self.y + math.sin(math.radians(self.angle)) * self.dna["vision_range"])

        # Ensure coordinates are within screen bounds
        if 0 <= eye_x < WIDTH and 0 <= eye_y < HEIGHT:
            # Get the RGB color at the detected position
            detected_color = screen.get_at((eye_x, eye_y))[:3]
        else:
            detected_color = (0, 0, 0)  # Default to black if out of bounds

        return detected_color

    # Animal class update modifications for focused profiling logging
    def update(self, food_items, animals, delta_time, game_speed_factor, screen):
        start_time = time.time()  # Start timing the update process

        # Hunger processing
        passive_hunger_cost = PASSIVE_HUNGER_DECAY * game_speed_factor * (delta_time / 1000.0)
        hunger_cost = abs(self.velocity) * HUNGER_DECAY_RATE * game_speed_factor * (delta_time / 1000.0)
        total_hunger_loss = hunger_cost + passive_hunger_cost
        self.hunger -= total_hunger_loss

        if self.hunger <= 0:
            return False

        # Detect RGB colors in front of the animal
        detect_start = time.time()
        r, g, b = self.detect_colors(screen)
        detect_time = time.time() - detect_start
        logging.info(f"Time for color detection: {detect_time:.4f} seconds")

        # Feedforward in the neural network
        nn_start = time.time()
        inputs = np.array([r / 255.0, g / 255.0, b / 255.0, self.hunger])
        output = self.nn.forward(inputs)
        nn_time = time.time() - nn_start
        logging.info(f"Time for neural network processing: {nn_time:.4f} seconds")

        # Update movement and rotation
        self.velocity = output[0]
        rotation = output[1]
        speed_factor = output[2]

        move_start = time.time()
        self.rotate(rotation)
        self.move()
        move_time = time.time() - move_start
        logging.info(f"Time for movement and rotation: {move_time:.4f} seconds")

        # Handle collisions
        collision_start = time.time()
        self.handle_collision(animals)
        collision_time = time.time() - collision_start
        logging.info(f"Time for collision handling: {collision_time:.4f} seconds")

        # Reproduction condition
        if self.hunger >= self.dna["reproduction_threshold"]:
            logging.info(f"Animal at ({self.x}, {self.y}) is reproducing. Hunger level: {self.hunger}")
            self.reproduce(animals)

        total_time = time.time() - start_time
        logging.info(f"Total time for update of animal at ({self.x}, {self.y}): {total_time:.4f} seconds")

        return True

    def handle_collision(self, animals):
        for animal in animals:
            if animal != self:
                dist = math.hypot(self.x - animal.x, self.y - animal.y)
                if dist < self.radius + animal.radius:
                    overlap = (self.radius + animal.radius) - dist
                    angle_between = math.atan2(animal.y - self.y, animal.x - self.x)
                    self.x -= overlap / 2 * math.cos(angle_between)
                    self.y -= overlap / 2 * math.sin(angle_between)
                    animal.x += overlap / 2 * math.cos(angle_between)
                    animal.y += overlap / 2 * math.sin(angle_between)

    def eat_fruit(self):
        self.hunger += 1.0
        self.fruits_eaten += 1
        logging.info(f"Animal at ({self.x}, {self.y}) ate fruit. Hunger increased to {self.hunger}")

    def reproduce(self, animals):
        if self.hunger >= REPRODUCTION_HUNGER_COST:
            self.hunger -= REPRODUCTION_HUNGER_COST
            logging.info(f"Animal at ({self.x}, {self.y}) reproduced. Hunger reduced by 2.")

            new_dna = self.dna.copy()
            for key in new_dna:
                if random.random() < new_dna["mutation_rate"]:
                    if key == "hidden_neurons":
                        new_dna[key] = max(2, new_dna[key] + random.randint(-1, 1))  # Mutate number of hidden neurons
                    else:
                        new_dna[key] += random.uniform(-1, 1)  # Slight mutation for other attributes

            new_animal = Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2), dna=new_dna)
            new_animal.nn.copy_from(self.nn)
            new_animal.nn.mutate()

            animals.append(new_animal)

    def draw(self, screen, font):
        # Draw animal body
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

        # Draw animal hunger (rounded to two decimal places)
        hunger_text = font.render(f"{self.hunger:.2f}", True, WHITE)
        screen.blit(hunger_text, (self.x - hunger_text.get_width() // 2, self.y - hunger_text.get_height() // 2))

        # Draw the eye (red for carnivores, white for others)
        eye_radius = 5
        eye_distance = self.radius + 5
        radians = math.radians(self.angle)
        eye_x = int(self.x + eye_distance * math.cos(radians))
        eye_y = int(self.y + eye_distance * math.sin(radians))

        eye_color = RED if self.is_carnivore else WHITE
        pygame.draw.circle(screen, eye_color, (eye_x, eye_y), eye_radius)

        # Draw a line indicating the direction of the eye based on DNA
        line_length = self.dna["eye_vision_length"]  # Line length is now part of DNA
        line_x = int(eye_x + line_length * math.cos(radians))
        line_y = int(eye_y + line_length * math.sin(radians))
        pygame.draw.line(screen, eye_color, (eye_x, eye_y), (line_x, line_y), 2)
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights for input to hidden layer and hidden to output layer
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_hidden = np.random.randn(hidden_size)
        self.bias_output = np.random.randn(output_size)

    def forward(self, inputs):
        # Input to hidden layer
        hidden = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_activation = self.sigmoid(hidden)  # Activation function

        # Hidden to output layer
        output = np.dot(hidden_activation, self.weights_hidden_output) + self.bias_output
        return output

    def mutate(self):
        mutation_strength = 0.1
        self.weights_input_hidden += mutation_strength * np.random.randn(*self.weights_input_hidden.shape)
        self.weights_hidden_output += mutation_strength * np.random.randn(*self.weights_hidden_output.shape)
        self.bias_hidden += mutation_strength * np.random.randn(*self.bias_hidden.shape)
        self.bias_output += mutation_strength * np.random.randn(*self.bias_output.shape)

    def copy_from(self, other):
        self.weights_input_hidden = np.copy(other.weights_input_hidden)
        self.weights_hidden_output = np.copy(other.weights_hidden_output)
        self.bias_hidden = np.copy(other.bias_hidden)
        self.bias_output = np.copy(other.bias_output)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # Sigmoid activation function to introduce non-linearity

# Food class
class Food:
    def __init__(self, x, y, size):
        self.rect = pygame.Rect(x, y, size, size)
        self.color = GREEN

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)

# Function to spawn new animals if population drops below 5
def spawn_best_animals(animals):
    if len(animals) < 5:
        print(f"Spawning new animals. Current animal count: {len(animals)}")

        if len(animals) == 0:
            for _ in range(5):
                new_animal = Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2))
                animals.append(new_animal)
        else:
            best_animals = sorted(animals, key=lambda animal: animal.hunger, reverse=True)[:5]

            for best_animal in best_animals:
                # Only spawn herbivores
                new_animal = Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2), dna=best_animal.dna.copy())
                new_animal.nn.copy_from(best_animal.nn)
                new_animal.nn.mutate()
                animals.append(new_animal)


def game_loop():
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Initialize animals and food
    animals = [Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2)) for _ in range(ANIMAL_COUNT)]
    food_items = []

    # Timer to spawn food at regular intervals
    pygame.time.set_timer(pygame.USEREVENT, FOOD_SPAWN_INTERVAL)

    # Slider settings (positioned visibly on the top right corner of the screen)
    slider_width, slider_height = 20, 200
    fps_slider_rect = pygame.Rect(WIDTH - 50 - slider_width, 50, slider_width, slider_height)  # Slider at the top right
    fps_value = FPS_DEFAULT
    game_speed_factor = 1.0
    simulation_time = 0  # Initialize simulation time

    running = True
    while running:
        delta_time = clock.tick(fps_value) / 1000 * game_speed_factor
        simulation_time += delta_time  # Increment simulation time with delta_time scaled by game speed factor

        # Get the current FPS
        current_fps = clock.get_fps()
        target_fps = fps_value

        # Logging both actual FPS and target FPS
        logging.info(f"Actual FPS: {current_fps:.2f}, Target FPS: {target_fps}")

        # Clear the screen
        screen.fill(BLACK)

        # Draw FPS slider at the top right corner
        pygame.draw.rect(screen, WHITE, fps_slider_rect)  # Draw slider background
        pygame.draw.rect(screen, RED, (fps_slider_rect.x, fps_slider_rect.y, fps_slider_rect.width, (fps_value - FPS_MIN) * fps_slider_rect.height / (FPS_MAX - FPS_MIN)))  # Draw slider value

        # Draw speed indicators on the slider from x1 to x10
        for i in range(1, 11):  # Indicators for x1 to x10
            indicator_y = fps_slider_rect.y + int(fps_slider_rect.height * (i - 1) / 10)
            speed_text = font.render(f"x{i}", True, WHITE)
            screen.blit(speed_text, (fps_slider_rect.right + 5, indicator_y - speed_text.get_height() // 2))

        # Display simulation timer and FPS information on screen
        sim_time_text = font.render(f"Time: {simulation_time:.2f} s", True, WHITE)
        actual_fps_text = font.render(f"Actual FPS: {int(current_fps)}", True, WHITE)
        target_fps_text = font.render(f"Target FPS: {int(target_fps)}", True, WHITE)

        # Display timer and FPS counters at a visible position
        screen.blit(sim_time_text, (10, 10))  # Timer at the top left
        screen.blit(actual_fps_text, (10, 35))  # Actual FPS below timer
        screen.blit(target_fps_text, (10, 60))  # Target FPS below actual FPS

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.USEREVENT:
                # Adjust food spawning to match new screen size (left and right halves)
                if random.random() < game_speed_factor:
                    # Spawn food on the left half densely
                    food_x = random.randint(0, WIDTH // 2 - FOOD_SIZE)
                    food_y = random.randint(0, HEIGHT - FOOD_SIZE)
                    food_items.append(Food(food_x, food_y, FOOD_SIZE))

                if random.random() < game_speed_factor * 0.5:
                    # Spawn food on the right half less frequently and smaller in size
                    food_x = random.randint(WIDTH // 2, WIDTH - FOOD_SIZE)
                    food_y = random.randint(0, HEIGHT - FOOD_SIZE)
                    food_items.append(Food(food_x, food_y, FOOD_SIZE // 2))

            # Handle FPS slider adjustments
            if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.MOUSEMOTION and event.buttons[0]):
                if fps_slider_rect.collidepoint(event.pos):
                    # Calculate FPS value and game speed factor based on slider position
                    fps_value = FPS_MIN + (FPS_MAX - FPS_MIN) * (event.pos[1] - fps_slider_rect.y) / fps_slider_rect.height
                    fps_value = int(max(FPS_MIN, min(FPS_MAX, fps_value)))
                    game_speed_factor = 1 + 9 * ((fps_value - FPS_MIN) / (FPS_MAX - FPS_MIN))

        # Update animals
        alive_animals = []
        for animal in animals:
            if animal.update(food_items, animals, delta_time, game_speed_factor, screen):
                alive_animals.append(animal)
                animal.draw(screen, font)

        animals = alive_animals

        # Spawn best animals if population drops below 5
        spawn_best_animals(animals)

        # Herbivores eating food
        for animal in animals:
            if not animal.is_carnivore:
                for food in food_items[:]:
                    if math.hypot(animal.x - food.rect.centerx, animal.y - food.rect.centery) < animal.radius:
                        animal.eat_fruit()
                        food_items.remove(food)

        # Draw food
        for food in food_items:
            food.draw(screen)

        # Update the display
        pygame.display.flip()

    pygame.quit()


# Run the game
game_loop()
