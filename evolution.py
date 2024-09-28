import pygame
import logging
import random
import numpy as np
import math

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1280, 720
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
screen = pygame.display.set_mode((WIDTH, HEIGHT))
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
    def __init__(self, x, y, dna=None, genome="normal"):
        self.x = x
        self.y = y
        self.radius = ANIMAL_RADIUS
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.angle = random.uniform(0, 360)
        self.velocity = 0
        self.rotation_speed = 0
        self.nn = NeuralNetwork()
        self.hunger = 2.0
        self.fruits_eaten = 0
        self.genome = genome
        self.is_carnivore = genome == "carnivore"

        # DNA Attributes
        if dna:
            self.dna = dna
        else:
            self.dna = {
                "speed": random.uniform(5, 15),  # Maximum speed
                "vision_range": random.uniform(100, 250),  # Detection range, now variable in DNA
                "eye_vision_length": random.uniform(20, 50),  # Length of the line showing eye vision
                "reproduction_threshold": random.uniform(REPRODUCTION_THRESHOLD, REPRODUCTION_THRESHOLD + 5),
                "mutation_rate": 0.1,  # Probability of mutation affecting DNA
            }

    def rotate(self, rotation):
        max_rotation_speed = 5
        self.rotation_speed = max(min(rotation, max_rotation_speed), -max_rotation_speed)
        self.angle += self.rotation_speed
        self.angle %= 360

    def move(self):
        max_velocity = self.dna["speed"]
        if self.is_carnivore:
            max_velocity += 5  # Carnivores are generally faster

        self.velocity = max(min(self.velocity, max_velocity), -max_velocity)

        radians = math.radians(self.angle)
        self.x += self.velocity * math.cos(radians)
        self.y += self.velocity * math.sin(radians)

        if self.x - self.radius < 0: self.x = self.radius
        if self.x + self.radius > WIDTH: self.x = WIDTH - self.radius
        if self.y - self.radius < 0: self.y = self.radius
        if self.y + self.radius > HEIGHT: self.y = HEIGHT - self.radius

    def detect_food(self, food_items, animals):
        eye_x = self.x + math.cos(math.radians(self.angle)) * self.dna["vision_range"]
        eye_y = self.y + math.sin(math.radians(self.angle)) * self.dna["vision_range"]
        detected_food = None
        detected_animal = None

        if not self.is_carnivore:
            for food in food_items:
                if math.hypot(food.rect.centerx - self.x, food.rect.centery - self.y) < self.dna["vision_range"]:
                    detected_food = food
                    break

        if self.is_carnivore:
            for animal in animals:
                if animal != self and math.hypot(self.x - animal.x, self.y - animal.y) < self.dna["vision_range"]:
                    detected_animal = animal
                    break

        return detected_food, detected_animal

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

                    if self.is_carnivore and overlap > 0:
                        logging.info(f"Carnivore at ({self.x}, {self.y}) ate another animal.")
                        self.eat_animal(animals, animal)

    def update(self, food_items, animals, delta_time, game_speed_factor):
        # Calculate passive hunger cost
        passive_hunger_cost = PASSIVE_HUNGER_DECAY * game_speed_factor * (delta_time / 1000.0)
        movement_penalty = CARNIVORE_MOVEMENT_PENALTY if self.is_carnivore else 1.0
        hunger_cost = abs(self.velocity) * HUNGER_DECAY_RATE * movement_penalty * game_speed_factor * (delta_time / 1000.0)
        total_hunger_loss = hunger_cost + passive_hunger_cost
        self.hunger -= total_hunger_loss

        # Check if the animal has died from hunger
        if self.hunger <= 0:
            logging.info(f"Animal at ({self.x}, {self.y}) died due to hunger.")
            return False

        detected_food, detected_animal = self.detect_food(food_items, animals)
        if detected_food:
            logging.info(f"Animal at ({self.x}, {self.y}) detected food at ({detected_food.rect.centerx}, {detected_food.rect.centery})")
        if detected_animal:
            logging.info(f"Carnivore at ({self.x}, {self.y}) detected another animal at ({detected_animal.x}, {detected_animal.y})")

        food_detected = 1 if detected_food else 0
        animal_detected = 1 if detected_animal else 0
        inputs = np.array([food_detected, animal_detected, self.x, self.y])

        output = self.nn.forward(inputs)
        self.velocity = output[0]
        rotation = output[1]
        speed_factor = output[2]

        self.rotate(rotation)
        self.move()
        self.velocity *= speed_factor
        self.handle_collision(animals)

        # Check for collision with food
        if not self.is_carnivore:
            for food in food_items[:]:
                if math.hypot(self.x - food.rect.centerx, self.y - food.rect.centery) < self.radius:
                    self.eat_fruit()
                    food_items.remove(food)

        # Reproduction condition
        if self.hunger >= self.dna["reproduction_threshold"]:
            logging.info(f"Animal at ({self.x}, {self.y}) is reproducing. Hunger level: {self.hunger}")
            self.reproduce(animals)

        return True


    def eat_fruit(self):
        if not self.is_carnivore:
            self.hunger += 1.0
            self.fruits_eaten += 1
            logging.info(f"Animal at ({self.x}, {self.y}) ate fruit. Hunger increased to {self.hunger}")

    def eat_animal(self, animals, target_animal):
        if target_animal in animals:
            animals.remove(target_animal)
            self.hunger += CARNIVORE_EAT_HUNGER_GAIN
            logging.info(f"Carnivore at ({self.x}, {self.y}) ate another animal. Hunger increased to {self.hunger}")

    def reproduce(self, animals):
        if self.hunger >= REPRODUCTION_HUNGER_COST:
            self.hunger -= REPRODUCTION_HUNGER_COST
            logging.info(f"Animal at ({self.x}, {self.y}) reproduced. Hunger reduced by 2.")

            new_dna = self.dna.copy()
            for key in new_dna:
                if random.random() < new_dna["mutation_rate"]:
                    new_dna[key] += random.uniform(-1, 1)  # Slight mutation

            new_animal = Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2), dna=new_dna, genome=self.genome)
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
                new_animal = Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2), dna=best_animal.dna.copy(), genome=best_animal.genome)
                new_animal.nn.copy_from(best_animal.nn)
                new_animal.nn.mutate()
                animals.append(new_animal)

# Game loop with speed control
def game_loop():
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    animals = [Animal(random.randint(0, WIDTH - ANIMAL_RADIUS * 2), random.randint(0, HEIGHT - ANIMAL_RADIUS * 2), genome=random.choice(["normal", "carnivore"])) for _ in range(ANIMAL_COUNT)]
    food_items = []

    pygame.time.set_timer(pygame.USEREVENT, FOOD_SPAWN_INTERVAL)

    # Slider settings
    fps_slider_rect = pygame.Rect(10, 10, 20, 200)
    fps_value = FPS_DEFAULT
    game_speed_factor = 1.0
    simulation_time = 0  # Initialize simulation time

    running = True
    while running:
        screen.fill(BLACK)
        delta_time = clock.tick(fps_value) / 1000 * game_speed_factor
        simulation_time += delta_time  # Increment simulation time with delta_time scaled by game speed factor

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.USEREVENT:
                if random.random() < game_speed_factor:
                    food_x = random.randint(0, WIDTH - FOOD_SIZE)
                    food_y = random.randint(0, HEIGHT - FOOD_SIZE)
                    food_items.append(Food(food_x, food_y, FOOD_SIZE))

            if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION:
                if fps_slider_rect.collidepoint(event.pos):
                    # Calculate FPS value and game speed factor based on slider position
                    fps_value = FPS_MIN + (FPS_MAX - FPS_MIN) * (event.pos[1] - fps_slider_rect.y) / fps_slider_rect.height
                    fps_value = int(max(FPS_MIN, min(FPS_MAX, fps_value)))
                    game_speed_factor = 1 + 9 * ((fps_value - FPS_MIN) / (FPS_MAX - FPS_MIN))  # Scale from 1x to 10x

                    # Log the current FPS value and game speed factor
                    logging.info(f"Slider adjusted: FPS = {fps_value}, Game Speed Factor = x{game_speed_factor:.2f}")

        # Draw FPS slider
        pygame.draw.rect(screen, WHITE, fps_slider_rect)
        pygame.draw.rect(screen, RED, (fps_slider_rect.x, fps_slider_rect.y, fps_slider_rect.width, (fps_value - FPS_MIN) * fps_slider_rect.height / (FPS_MAX - FPS_MIN)))

        # Draw simulation timer on the left
        sim_time_text = font.render(f"Time: {simulation_time:.2f} s", True, WHITE)
        screen.blit(sim_time_text, (10, fps_slider_rect.bottom + 10))

        # Draw speed indicators on the slider from x1 to x10
        for i in range(1, 11):  # Indicators for x1 to x10
            indicator_y = fps_slider_rect.y + int(fps_slider_rect.height * (i - 1) / 10)
            speed_text = font.render(f"x{i}", True, WHITE)
            screen.blit(speed_text, (fps_slider_rect.right + 5, indicator_y - speed_text.get_height() // 2))

        alive_animals = []
        for animal in animals:
            if animal.update(food_items, animals, delta_time, game_speed_factor):
                alive_animals.append(animal)
                animal.draw(screen, font)
            else:
                food_items.append(Food(int(animal.x), int(animal.y), FOOD_SIZE))

        animals = alive_animals

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
