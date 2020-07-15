#!/usr/bin/env python3

import math
import time

import rospy
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation as Rot


def mod2pi(theta):
    return theta - 2.0 * math.pi * math.floor(theta/2.0/math.pi)

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def left_straight_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d + sa - sb

    mode = ["L", "S", "L"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sa - sb))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((cb - ca), tmp0)
    t = mod2pi(-alpha + tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(beta - tmp1)

    return t, p, q, mode

def right_straight_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    tmp0 = d - sa + sb
    mode = ["R", "S", "R"]
    p_squared = 2 + (d * d) - (2 * c_ab) + (2 * d * (sb - sa))
    if p_squared < 0:
        return None, None, None, mode
    tmp1 = math.atan2((ca - cb), tmp0)
    t = mod2pi(alpha - tmp1)
    p = math.sqrt(p_squared)
    q = mod2pi(-beta + tmp1)

    return t, p, q, mode

def left_straight_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = -2 + (d * d) + (2 * c_ab) + (2 * d * (sa + sb))
    mode = ["L", "S", "R"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
    t = mod2pi(-alpha + tmp2)
    q = mod2pi(-mod2pi(beta) + tmp2)

    return t, p, q, mode

def right_straight_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    p_squared = (d * d) - 2 + (2 * c_ab) - (2 * d * (sa + sb))
    mode = ["R", "S", "L"]
    if p_squared < 0:
        return None, None, None, mode
    p = math.sqrt(p_squared)
    tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
    t = mod2pi(alpha - tmp2)
    q = mod2pi(beta - tmp2)

    return t, p, q, mode


def right_left_right(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["R", "L", "R"]
    tmp_rlr = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (sa - sb)) / 8.0
    if abs(tmp_rlr) > 1.0:
        return None, None, None, mode

    p = mod2pi(2 * math.pi - math.acos(tmp_rlr))
    t = mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + mod2pi(p / 2.0))
    q = mod2pi(alpha - beta - t + mod2pi(p))
    return t, p, q, mode


def left_right_left(alpha, beta, d):
    sa = math.sin(alpha)
    sb = math.sin(beta)
    ca = math.cos(alpha)
    cb = math.cos(beta)
    c_ab = math.cos(alpha - beta)

    mode = ["L", "R", "L"]
    tmp_lrl = (6.0 - d * d + 2.0 * c_ab + 2.0 * d * (- sa + sb)) / 8.0
    if abs(tmp_lrl) > 1:
        return None, None, None, mode
    p = mod2pi(2 * math.pi - math.acos(tmp_lrl))
    t = mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
    q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p))

    return t, p, q, mode


def dubins_path_planning_from_origin(end_x, end_y, end_yaw, curvature,
                                     step_size):
    dx = end_x
    dy = end_y
    D = math.hypot(dx, dy)
    d = D * curvature

    theta = mod2pi(math.atan2(dy, dx))
    alpha = mod2pi(- theta)
    beta = mod2pi(end_yaw - theta)

    planners = [left_straight_left, right_straight_right, left_straight_right,
                right_straight_left, right_left_right,
                left_right_left]

    best_cost = float("inf")
    bt, bp, bq, best_mode = None, None, None, None

    for planner in planners:
        t, p, q, mode = planner(alpha, beta, d)
        if t is None:
            continue

        cost = (abs(t) + abs(p) + abs(q))
        if best_cost > cost:
            bt, bp, bq, best_mode = t, p, q, mode
            best_cost = cost
    lengths = [bt, bp, bq]

    x_list, y_list, yaw_list, directions = generate_local_course(
        sum(lengths), lengths, best_mode, curvature, step_size)

    return x_list, y_list, yaw_list, best_mode, best_cost


def interpolate(ind, length, mode, max_curvature, origin_x, origin_y,
                origin_yaw, path_x, path_y, path_yaw, directions):
    if mode == "S":
        path_x[ind] = origin_x + length / max_curvature * math.cos(origin_yaw)
        path_y[ind] = origin_y + length / max_curvature * math.sin(origin_yaw)
        path_yaw[ind] = origin_yaw
    else:  # curve
        ldx = math.sin(length) / max_curvature
        ldy = 0.0
        if mode == "L":  # left turn
            ldy = (1.0 - math.cos(length)) / max_curvature
        elif mode == "R":  # right turn
            ldy = (1.0 - math.cos(length)) / -max_curvature
        gdx = math.cos(-origin_yaw) * ldx + math.sin(-origin_yaw) * ldy
        gdy = -math.sin(-origin_yaw) * ldx + math.cos(-origin_yaw) * ldy
        path_x[ind] = origin_x + gdx
        path_y[ind] = origin_y + gdy

    if mode == "L":  # left turn
        path_yaw[ind] = origin_yaw + length
    elif mode == "R":  # right turn
        path_yaw[ind] = origin_yaw - length

    if length > 0.0:
        directions[ind] = 1
    else:
        directions[ind] = -1

    return path_x, path_y, path_yaw, directions


def dubins_path_planning(s_x, s_y, s_yaw, g_x, g_y, g_yaw, c, step_size=0.1):
    """
    Dubins path planner
    input:
        s_x x position of start point [m]
        s_y y position of start point [m]
        s_yaw yaw angle of start point [rad]
        g_x x position of end point [m]
        g_y y position of end point [m]
        g_yaw yaw angle of end point [rad]
        c curvature [1/m]
    """

    g_x = g_x - s_x
    g_y = g_y - s_y

    l_rot = Rot.from_euler('z', s_yaw).as_dcm()[0:2, 0:2]
    le_xy = np.stack([g_x, g_y]).T @ l_rot
    le_yaw = g_yaw - s_yaw

    lp_x, lp_y, lp_yaw, mode, lengths = dubins_path_planning_from_origin(
        le_xy[0], le_xy[1], le_yaw, c, step_size)

    rot = Rot.from_euler('z', -s_yaw).as_dcm()[0:2, 0:2]
    converted_xy = np.stack([lp_x, lp_y]).T @ rot
    x_list = converted_xy[:, 0] + s_x
    y_list = converted_xy[:, 1] + s_y
    yaw_list = [pi_2_pi(i_yaw + s_yaw) for i_yaw in lp_yaw]

    return x_list, y_list, yaw_list, mode, lengths


def generate_local_course(total_length, lengths, mode, max_curvature,
                          step_size):
    n_point = math.trunc(total_length / step_size) + len(lengths) + 4

    path_x = [0.0 for _ in range(n_point)]
    path_y = [0.0 for _ in range(n_point)]
    path_yaw = [0.0 for _ in range(n_point)]
    directions = [0.0 for _ in range(n_point)]
    index = 1

    if lengths[0] > 0.0:
        directions[0] = 1
    else:
        directions[0] = -1

    ll = 0.0

    for (m, l, i) in zip(mode, lengths, range(len(mode))):
        if l > 0.0:
            d = step_size
        else:
            d = -step_size

        # set origin state
        origin_x, origin_y, origin_yaw = \
            path_x[index], path_y[index], path_yaw[index]

        index -= 1
        if i >= 1 and (lengths[i - 1] * lengths[i]) > 0:
            pd = - d - ll
        else:
            pd = d - ll

        while abs(pd) <= abs(l):
            index += 1
            path_x, path_y, path_yaw, directions = interpolate(
                index, pd, m, max_curvature, origin_x, origin_y, origin_yaw,
                path_x, path_y, path_yaw, directions)
            pd += d

        ll = l - pd - d  # calc remain length

        index += 1
        path_x, path_y, path_yaw, directions = interpolate(
            index, l, m, max_curvature, origin_x, origin_y, origin_yaw,
            path_x, path_y, path_yaw, directions)

    if len(path_x) <= 1:
        return [], [], [], []

    # remove unused data
    while len(path_x) >= 1 and path_x[-1] == 0.0:
        path_x.pop()
        path_y.pop()
        path_yaw.pop()
        directions.pop()

    return path_x, path_y, path_yaw, directions



def callback(msg):
    global roll, pitch, yaw

    end_x = msg.pose.position.x
    end_y = msg.pose.position.y
    angle = msg.pose.orientation

    orientation_list = [angle.x, angle.y, angle.z, angle.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

    end_yaw = yaw

    path_x, path_y, path_yaw, mode, path_length = dubins_path_planning(
            start_x, start_y, start_yaw,
            end_x, end_y, end_yaw, curvature)

    path_pub = rospy.Publisher('/path', Path, queue_size=10)
    current_time = rospy.Time.now()
    path = Path()
   # path.header.stamp = current_time
    path.header.frame_id = "/map"

                
    for i in range(len(path_x)):
        print(path_x[i], path_y[i])
        goal = PoseStamped()
        goal.pose.position.x = path_x[i]
        goal.pose.position.y = path_y[i]
        print(goal.pose.position.x, goal.pose.position.y)
    #    goal.header.stamp = current_time
        goal.header.frame_id = "/map"
        path.poses.append(goal)
        path_pub.publish(path)

        last_time = current_time	


def main():
    print("Dubins path planner sample start!!")

    global start_x, start_y, start_yaw, end_x, end_y, end_yaw, curvature
    

    start_x = 1.0
    start_y = 1.0
    start_yaw = np.deg2rad(45.0)

    curvature = 1.0;

    rospy.init_node('dubin_node')
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback)

    path_pub = rospy.Publisher('/path', Path, queue_size=10)

    current_time = rospy.Time.now()
    last_time = rospy.Time.now()

  #  path = Path()
 #   path.header.stamp = current_time
#    path.header.frame_id = "/map"

    rospy.spin()

if __name__ == '__main__':
    main()
