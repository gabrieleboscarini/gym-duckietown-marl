import gymnasium as gym
import itertools

def launch_env(id=None):
    env = None
    if id is None:
        # Launch the environment
        from src.gym_duckietown.simulator import Simulator

        env = Simulator(
            seed=123,  # random seed
            map_name="4way_multi",
            max_steps=500001,  # we don't want the gym to reset itself
            domain_rand=False,
            camera_width=640,
            camera_height=480,
            accept_start_angle_deg=4,  # start close to straight
            full_transparency=True,
            distortion=False,
            draw_trajectory=[2]
        )
    else:
        env = gym.make(id)

    return env

def load_env_obstacles(local_env):
    ##### Loading Circular Obstacles
    list_obstacles = []
    for obstacle in local_env.objects:
        pose = obstacle.pos
        robot_radius = 0.1
        radius = obstacle.safety_radius + robot_radius
        list_obstacles.append([pose[0], pose[2], radius])
    
    ###### Non drivable tiles
    for i in range(local_env.grid_width):
        for j in range(local_env.grid_height):
            tile = local_env.grid[j * local_env.grid_width + i]
            if not tile['drivable']:
                coords = tile['coords']
                list_obstacles.append([(coords[0]+.5) * local_env.road_tile_size, (coords[1]+.5) * local_env.road_tile_size, 1.3*local_env.road_tile_size + 2*robot_radius])
    return list_obstacles

def wrap_env(env, results_path='./gym_results'):
    from gymnasium import wrappers
    return wrappers.RecordVideo(env=env, video_folder=results_path)

def get_4way_coord(env):
    
        # For each grid tile
        for i, j in itertools.product(range(env.grid_width), range(env.grid_height)):

            # Get the tile type and angle
            tile = env._get_tile(i, j)
            if tile["kind"] == "4way":
                return i,j 