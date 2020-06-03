import math

def distance(p1, p2):
    '''Compute the eucledian distance between two points'''
    dx, dy = difference(p2, p1)
    return math.sqrt(dx**2 + dy**2)

def difference(v2, v1):
    '''Compute the vector between two points'''
    return [v2[0]-v1[0], v2[1]-v1[1]]

def normalize(vec):
    '''Normalize a vector to unit amplitude'''
    magnitude = math.sqrt(vec[0]**2 + vec[1]**2)
    return [v/(magnitude + 1e-9) for v in vec]

def cosine_similarity(v1, v2):
    '''Find cos(theta) between vectors v1, v2'''
    return sum([a*b for a, b in zip(v1, v2)])

def is_along_st_line(waypoints, prev_waypoint_index, next_waypoint_index, current_location):
    '''Determine if the car is currently on a straight line'''
    if distance(waypoints[prev_waypoint_index], current_location) < distance(waypoints[next_waypoint_index], current_location):
        nearest_waypoint_index = prev_waypoint_index
    else:
        nearest_waypoint_index = next_waypoint_index
    
    mid_waypoint = waypoints[nearest_waypoint_index]
    prev_waypoint = waypoints[(nearest_waypoint_index - 1) % len(waypoints)]
    next_waypoint = waypoints[(nearest_waypoint_index + 1) % len(waypoints)]
    
    prev_vec = normalize(difference(mid_waypoint, prev_waypoint))
    next_vec = normalize(difference(next_waypoint, mid_waypoint))
    
    if cosine_similarity(prev_vec, next_vec) >= math.cos(0.1 * math.pi):
        # They are very similar, probably a straight line
        return True
    return False
    
def reward_function(params):
    # Read input parameters
    distance_from_center = params['distance_from_center']
    speed = params['speed']
    track_width = params['track_width']
    steering = abs(params['steering_angle']) # Only need the absolute steering angle
    all_wheels_on_track = params['all_wheels_on_track']
    waypoints = params['waypoints']
    prev_waypoint_index, next_waypoint_index = params['closest_waypoints']
    
    current_location = [params['x'], params['y']]
    next_waypoint = waypoints[next_waypoint_index]
    
    heading = params['heading']
    # Map heading to range [0, 360[ instead of [-180, +180[
    if heading < 0:
        heading += 360

    # Try to follow the line between the two waypoints
    # Compute the angle between:
    # vector joining the car center and the next waypoint AND x-axis
    delta_x = next_waypoint[0] - current_location[0]
    delta_y = next_waypoint[1] - current_location[1]
    angle = math.acos(delta_x/math.sqrt(delta_x**2 + delta_y**2))
    angle *= (180 / math.pi)
    if delta_y < 0:
        angle = 360 - angle
        
    # Compute the deviation between the car's current heading
    # and the path joining the car's center to the next waypoint
    deviation = angle - heading
    deviation *= math.pi/180
    
    # cos(deviation) = 1 if deviation is 0, this looks nice as a reward function
    # Be more aggressive than basic cosine function
    reward = math.cos(deviation) * abs(math.cos(deviation))

    # Steering penality threshold, change the number based on your action space setting
    ABS_STEERING_THRESHOLD = 20
    MAX_SPEED = 2

    # Penalize reward if the agent is steering too much
    if steering > ABS_STEERING_THRESHOLD:
        reward *= 0.8
    
    # Penalize reward if car is on a straight line and moving slowly
    if is_along_st_line(waypoints, prev_waypoint_index, next_waypoint_index, current_location):
        reward *= speed / MAX_SPEED
    
    # Penalize reward if car goes out of the track
    if not all_wheels_on_track:
        reward *= 0.5

    return float(reward)
