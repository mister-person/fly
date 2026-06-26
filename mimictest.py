import time
from dm_control import mujoco
import display
from torch import multiprocessing
from FlyMimic import flymimic
import numpy as np

if __name__ == "__main__":
    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    frame_queue = mp_context.Queue()
    event_queue = mp_context.Queue()
    WIDTH, HEIGHT = (640, 480)
    process = mp_context.Process(target=display.pygame_thread, args=(frame_queue, event_queue, (WIDTH, HEIGHT)))
    process.start()

    #env = flymimic.fly.mocap_tracking_muscle.mocap_tracking_muscle()
    path = "FlyMimic/flymimic/assets/models/best_combined_arm_cvt3.xml"
    physics = mujoco.Physics.from_xml_path(path)
    
    camera = mujoco.Camera(
        physics, height=HEIGHT, width=WIDTH, camera_id=0
    )

    while True:
        action = np.full(15, .0001)
        obs = physics.step(1)
        print(obs)
        frame = camera.render()
        frame_queue.put((0, frame.tobytes()))

        time.sleep(1/60)
