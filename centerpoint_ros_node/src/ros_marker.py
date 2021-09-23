import numpy as np
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion
from matplotlib import cm

# some default visualization settings
cmap = cm.hsv  # colormap
color_count = 256  # colormap size


def make_marker(obj, header, tracking_names, color=None):
    # check for class
    class_name = None
    try:
        class_name = tracking_names[obj.label]
    except IndexError:
        class_name = obj.label
        rospy.logerr("Class name is undefined!")

    if color == "class":
        color = cmap(obj.label % len(tracking_names))
    else:
        color = cmap(obj.id % color_count)

    # text
    v_absolute = np.sqrt(obj.velocity.linear.x**2 + obj.velocity.linear.y**2)
    v_absolute_km_h = v_absolute * 3.6
    marker_t = Marker()
    marker_t.header = header
    marker_t.ns = "ids"
    marker_t.id = obj.id
    marker_t.type = Marker.TEXT_VIEW_FACING
    marker_t.text = f"{class_name}: {obj.id} ({obj.score:.2f})"
    marker_t.action = Marker.ADD
    marker_t.scale.z = 0.5
    marker_t.color.a = color[3]
    marker_t.color.r = color[0]
    marker_t.color.g = color[1]
    marker_t.color.b = color[2]
    marker_t.lifetime = rospy.Duration()
    marker_t.pose.orientation.w = 1.0
    marker_t.pose.position.x = obj.pose.position.x
    marker_t.pose.position.y = obj.pose.position.y
    marker_t.pose.position.z = obj.pose.position.z + obj.size.z / 2 + 0.15
    # object
    marker_o = Marker()
    marker_o.header = header
    marker_o.ns = "objects"
    marker_o.id = obj.id
    marker_o.type = Marker.LINE_LIST
    marker_o.action = Marker.ADD
    marker_o.scale.x = 0.1
    marker_o.scale.y = 0.1
    marker_o.scale.z = 0.1
    marker_o.color.a = color[3]
    marker_o.color.r = color[0]
    marker_o.color.g = color[1]
    marker_o.color.b = color[2]
    marker_o.lifetime = rospy.Duration()
    marker_o.pose.orientation.w = 1.0
    yaw = euler_from_quaternion([obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w])[2]

    xdir = np.array([np.cos(yaw), np.sin(yaw), 0])
    ydir = np.array([-xdir[1], xdir[0], 0])
    center = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z])
    vs = np.zeros((8, 3))
    l2 = obj.size.x / 2.
    w2 = obj.size.y / 2.
    h = np.array([0., 0., obj.size.z / 2.])

    # bbox's vertices
    #          |x
    #      C   |   D-----------
    #          |              |
    # y---------------     length
    #          |              |
    #      B   |   A-----------

    vs[0] = center + xdir * -l2 + ydir * -w2 - h  # A bot
    vs[1] = center + xdir * -l2 + ydir * w2 - h   # B bot
    vs[2] = center + xdir * l2 + ydir * w2 - h    # C bot
    vs[3] = center + xdir * l2 + ydir * -w2 - h   # D bot
    vs[4] = vs[0] + 2 * h                         # E top
    vs[5] = vs[1] + 2 * h                         # F top
    vs[6] = vs[2] + 2 * h                         # G top
    vs[7] = vs[3] + 2 * h                         # H top

    # filling 12 edges
    # A -> B
    marker_o.points.append(Point(vs[0, 0], vs[0, 1], vs[0, 2]))
    marker_o.points.append(Point(vs[1, 0], vs[1, 1], vs[1, 2]))
    # B -> C
    marker_o.points.append(Point(vs[1, 0], vs[1, 1], vs[1, 2]))
    marker_o.points.append(Point(vs[2, 0], vs[2, 1], vs[2, 2]))
    # C -> D
    marker_o.points.append(Point(vs[2, 0], vs[2, 1], vs[2, 2]))
    marker_o.points.append(Point(vs[3, 0], vs[3, 1], vs[3, 2]))
    # D -> A
    marker_o.points.append(Point(vs[3, 0], vs[3, 1], vs[3, 2]))
    marker_o.points.append(Point(vs[0, 0], vs[0, 1], vs[0, 2]))
    # E -> F
    marker_o.points.append(Point(vs[4, 0], vs[4, 1], vs[4, 2]))
    marker_o.points.append(Point(vs[5, 0], vs[5, 1], vs[5, 2]))
    # F -> G
    marker_o.points.append(Point(vs[5, 0], vs[5, 1], vs[5, 2]))
    marker_o.points.append(Point(vs[6, 0], vs[6, 1], vs[6, 2]))
    # G -> H
    marker_o.points.append(Point(vs[6, 0], vs[6, 1], vs[6, 2]))
    marker_o.points.append(Point(vs[7, 0], vs[7, 1], vs[7, 2]))
    # H -> E
    marker_o.points.append(Point(vs[7, 0], vs[7, 1], vs[7, 2]))
    marker_o.points.append(Point(vs[4, 0], vs[4, 1], vs[4, 2]))
    # A -> E
    marker_o.points.append(Point(vs[0, 0], vs[0, 1], vs[0, 2]))
    marker_o.points.append(Point(vs[4, 0], vs[4, 1], vs[4, 2]))
    # B -> F
    marker_o.points.append(Point(vs[1, 0], vs[1, 1], vs[1, 2]))
    marker_o.points.append(Point(vs[5, 0], vs[5, 1], vs[5, 2]))
    # C -> G
    marker_o.points.append(Point(vs[2, 0], vs[2, 1], vs[2, 2]))
    marker_o.points.append(Point(vs[6, 0], vs[6, 1], vs[6, 2]))
    # D -> H
    marker_o.points.append(Point(vs[3, 0], vs[3, 1], vs[3, 2]))
    marker_o.points.append(Point(vs[7, 0], vs[7, 1], vs[7, 2]))

    # direction as part of bbox
    ar0 = center + ydir * w2; ar0 = Point(*ar0)
    ar1 = center - ydir * w2; ar1 = Point(*ar1)
    ar2 = center + xdir * l2; ar2 = Point(*ar2)
    marker_o.points.extend([ar0, ar1, ar1, ar2, ar2, ar0])

    # velocity
    marker_v = Marker()
    marker_v.header = header
    marker_v.ns = "velocities"
    start_p = Point(
        obj.pose.position.x,
        obj.pose.position.y,
        obj.pose.position.z
    )
    v = np.array([obj.velocity.linear.x, obj.velocity.linear.y])
    if v_absolute > 0.1:
        v = v * 2.5 / v_absolute
    end_p = Point(
        obj.pose.position.x + v[0],
        obj.pose.position.y + v[1],
        obj.pose.position.z
    )
    marker_v.points.append(start_p)
    marker_v.points.append(end_p)
    marker_v.scale.x = 0.08
    marker_v.scale.y = 0.16
    marker_v.scale.z = 0.08
    marker_v.id = obj.id
    marker_v.type = Marker.ARROW
    marker_v.action = Marker.ADD
    marker_v.color.a = color[3]
    marker_v.color.r = color[0]
    marker_v.color.g = color[1]
    marker_v.color.b = color[2]
    marker_v.lifetime = rospy.Duration()
    marker_v.pose.orientation.w = 1.0
    # velocity text
    marker_vt = Marker()
    marker_vt.header = header
    marker_vt.ns = "velocities_text"
    marker_vt.id = obj.id
    marker_vt.type = Marker.TEXT_VIEW_FACING
    marker_vt.text = f"{v_absolute_km_h:.2f} km/h"
    marker_vt.action = Marker.ADD
    marker_vt.scale.z = 0.5
    marker_vt.color.a = color[3]
    marker_vt.color.r = color[0]
    marker_vt.color.g = color[1]
    marker_vt.color.b = color[2]
    marker_vt.lifetime = rospy.Duration()
    marker_vt.pose.orientation.w = 1.0
    marker_vt.pose.position.x = end_p.x + v[0] * 0.06
    marker_vt.pose.position.y = end_p.y + v[1] * 0.06
    marker_vt.pose.position.z = end_p.z + 0.1

    # trajectory points
    marker_p = Marker()
    marker_p.header = header
    marker_p.ns = "trajectories"
    marker_p.id = obj.id
    marker_p.type = Marker.LINE_STRIP
    marker_p.pose.orientation.w = 1.
    marker_p.scale.x = 0.2
    for p in obj.trajectory:
        marker_p.points.append(p.point)
    marker_p.action = Marker.ADD
    marker_p.color.a = color[3]
    marker_p.color.r = color[0]
    marker_p.color.g = color[1]
    marker_p.color.b = color[2]
    marker_p.lifetime = rospy.Duration()

    # future trajectory points
    marker_f = Marker()
    marker_f.header = header
    marker_f.ns = "predictions"
    marker_f.id = obj.id
    marker_f.type = Marker.POINTS
    marker_f.pose.orientation.w = 1.
    marker_f.scale.x = 0.2
    marker_f.scale.y = 0.2
    for i in np.arange(0.5, 3., 0.25):
        p = Point(
            start_p.x + i * obj.velocity.linear.x,
            start_p.y + i * obj.velocity.linear.y,
            start_p.z
        )
        marker_f.points.append(p)
    marker_f.action = Marker.ADD
    marker_f.color.a = color[3]
    marker_f.color.r = color[0]
    marker_f.color.g = color[1]
    marker_f.color.b = color[2]
    marker_f.lifetime = rospy.Duration()

    return [marker_o, marker_t, marker_v, marker_vt, marker_p, marker_f]
