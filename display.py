from multiprocessing import Queue
from threading import Condition

def pygame_thread(frame_queue: Queue, frame_ready: Condition, event_queue: Queue, size: tuple[int, int]):
    FRAME_WIDTH, FRAME_HEIGHT = size
    # pygame setup
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    clock = pygame.time.Clock()

    screen.fill("purple")
    pygame.draw.circle(screen, "green", pygame.mouse.get_pos(), 40)
    pygame.display.flip()
    running = True
    a = True
    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            else:
                event_queue.put((event.type, event.dict))

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("purple")

        pygame.draw.circle(screen, "red", pygame.mouse.get_pos(), 40)

        # action = np.random.normal(size=59)  # 59 is the walking action dimension.
        # timestep = env.step(action)

        # physics: Physics = env.physics
        # pixels: npt.NDArray = physics.render(camera_id=1)
        queue_len = frame_queue.qsize()
        if queue_len > 0:
            print("RUNNING BEHIND BY", queue_len, "FRAMES")
        frame = frame_queue.get()
        pygame.draw.circle(screen, "blue", (600, 500), 40)

        image = pygame.image.frombytes(frame, (FRAME_WIDTH, FRAME_HEIGHT), "RGB")
        screen.blit(image, (0, 0))

        # flip() the display to put your work on screen
        pygame.display.flip()

        # clock.tick(60)  # limits FPS to 60

    pygame.quit()

