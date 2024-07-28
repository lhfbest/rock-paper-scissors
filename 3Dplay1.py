import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D, art3d


import matplotlib.image as mpimg
from matplotlib import rcParams
import time
# 设置字体以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 初始化点集数量
num_points = 10

# 生成初始点集在单位球面上
def generate_points_on_sphere(num_points):
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    costheta = np.random.uniform(-1, 1, num_points)
    u = np.random.uniform(0, 1, num_points)

    theta = np.arccos(costheta)
    r = u ** (1/3)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.vstack((x, y, z)).T

rock_points = generate_points_on_sphere(num_points)
paper_points = generate_points_on_sphere(num_points)
scissors_points = generate_points_on_sphere(num_points)

rock_img = mpimg.imread('C:\others\play\石头.jpg')
paper_img = mpimg.imread('C:\others\play\布.jpg')
scissors_img = mpimg.imread('C:\others\play\剪刀.jpg')


# 定义运动步长
step_size = 0.02
weight_power = 0.6

# 球面距离函数
def spherical_distance(point1, point2):
    return np.arccos(np.clip(np.dot(point1, point2), -1.0, 1.0))

# 定义同化函数
def assimilate(points1, points2, distance_threshold=0.1):
    if len(points1) == 0 or len(points2) == 0:
        return points1, points2

    new_points2 = []
    for i, point in enumerate(points1):
        distances = np.array([spherical_distance(point, p) for p in points2])
        if np.any(distances < distance_threshold):
            new_points2.append(point)
    if new_points2:
        points2 = np.vstack((points2, new_points2))
        points1 = np.array([point for point in points1 if not np.any([spherical_distance(point, p) < distance_threshold for p in points2])])
    return points1, points2



# 更新点集位置，所有天敌的斥力的和向量
def update_points_attack1(points, target_points, enemy_points):
    if len(points) == 0:
        return points

    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0, 0.0])

        # 计算所有天敌对其的斥力
        if len(enemy_points) > 0:
            for enemy in enemy_points:
                enemy_dir = point - enemy
                enemy_length = np.linalg.norm(enemy_dir)
                if enemy_length > 0:
                    move_vector += enemy_dir / enemy_length**2  # 斥力反比例缩放

        # 当天敌消失时，朝着猎物移动
        if  len(target_points) > 0:
            distances = np.array([spherical_distance(point, p) for p in target_points])
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length**2  # 影响力反比例缩放

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 更新点的位置，并确保在单位球面上
        if len(points) > 0:
            weight = len(points)**(weight_power)
        else:
            weight = 1
        new_point = point + move_vector * step_size * weight
        new_point = new_point / np.linalg.norm(new_point)
        points[i] = new_point

    return points





# 更新点集位置，所有天敌的斥力的和向量
def update_points_run1(points, target_points, enemy_points):
    if len(points) == 0:
        return points

    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0, 0.0])

        # 计算所有天敌对其的斥力
        if len(enemy_points) > 0:
            for enemy in enemy_points:
                enemy_dir = point - enemy
                enemy_length = np.linalg.norm(enemy_dir)
                if enemy_length > 0:
                    move_vector += enemy_dir / enemy_length**2  # 斥力反比例缩放

        # 当天敌消失时，朝着猎物移动
        if len(enemy_points) == 0 and len(target_points) > 0:
            distances = np.array([spherical_distance(point, p) for p in target_points])
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length**2  # 影响力反比例缩放

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 更新点的位置，并确保在单位球面上
        if len(points) > 0:
            weight = len(points)**(weight_power)
        else:
            weight = 1
        new_point = point + move_vector * step_size * weight
        new_point = new_point / np.linalg.norm(new_point)
        points[i] = new_point

    return points


# 更新点集位置，根据最近的猎物或天敌来判定移动方向
def update_points_run2(points, target_points, enemy_points):
    if len(points) == 0:
        return points

    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0, 0.0])

        if len(enemy_points) > 0:
            # 远离最近的天敌
            distances = np.array([spherical_distance(point, p) for p in enemy_points])
            nearest_enemy = enemy_points[np.argmin(distances)]
            move_vector = point - nearest_enemy
        elif len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.array([spherical_distance(point, p) for p in target_points])
            nearest_target = target_points[np.argmin(distances)]
            move_vector = nearest_target - point

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 更新点的位置，并确保在单位球面上

        if len(points)>0:
            weight = len(points)**(weight_power)
        else:
            weight = 1
        new_point = point + move_vector * step_size * weight
        new_point = new_point / np.linalg.norm(new_point)
        points[i] = new_point

    return points



# 更新点集位置，根据最近的猎物或天敌来判定移动方向
def update_points_attack2(points, target_points, enemy_points):
    if len(points) == 0:
        return points

    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0, 0.0])

        if len(enemy_points) > 0:
            # 远离最近的天敌
            distances = np.array([spherical_distance(point, p) for p in enemy_points])
            nearest_enemy = enemy_points[np.argmin(distances)]
            move_vector += point - nearest_enemy

        if len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.array([spherical_distance(point, p) for p in target_points])
            nearest_target = target_points[np.argmin(distances)]
            move_vector += nearest_target - point

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 更新点的位置，并确保在单位球面上
        if len(points) > 0:
            weight = len(points)**(weight_power)
        else:
            weight = 1
        new_point = point + move_vector * step_size * weight
        new_point = new_point / np.linalg.norm(new_point)
        points[i] = new_point

    return points



# 创建全屏图像
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mng = plt.get_current_fig_manager()
mng.window.state('zoomed')  # 全屏（在Windows上）


# 绘制球体的经纬线
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)


no_assimilation_counter_rock = np.zeros(num_points, dtype=int)
no_assimilation_counter_paper = np.zeros(num_points, dtype=int)
no_assimilation_counter_scissors = np.zeros(num_points, dtype=int)
reverse_steps_rock = np.zeros(num_points, dtype=int)
reverse_steps_paper = np.zeros(num_points, dtype=int)
reverse_steps_scissors = np.zeros(num_points, dtype=int)
max_no_assimilation_steps = 100  # 迭代次数阈值
reverse_iterations = 5  # 反向移动的步数



def plot_image_on_sphere(ax, img, point, size=0.1):
    x, y, z = point
    dx, dy, dz = size, size, size

    img_plane = art3d.Poly3DCollection([[
        (x - dx, y - dy, z),
        (x + dx, y - dy, z),
        (x + dx, y + dy, z),
        (x - dx, y + dy, z)
    ]], facecolors='none', edgecolors='none')

    ax.add_collection3d(img_plane)
    ax.imshow(img, aspect='auto', extent=[x - dx, x + dx, y - dy, y + dy], zorder=1)


def animate(frame):
    global rock_points, paper_points, scissors_points

    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 更新点的位置
    # rock_points = update_points2(rock_points, paper_points, scissors_points)     # 石头朝最近的布移动，或远离剪刀
    # paper_points = update_points(paper_points, scissors_points, rock_points) # 布朝最近的剪刀移动，或远离石头
    # scissors_points = update_points(scissors_points, rock_points, paper_points)  # 剪刀朝最近的石头移动，或远离布



    if len(rock_points) > 0:
        rock_points = update_points_run2(rock_points,  scissors_points, paper_points)  # 石头朝最近的布移动，或远离剪刀
    if len(paper_points) > 0:
        paper_points = update_points_attack2(paper_points,  rock_points,scissors_points)  # 布朝最近的剪刀移动，或远离石头
    if len(scissors_points) > 0:
        scissors_points = update_points_attack2(scissors_points, paper_points, rock_points)  # 剪刀朝最近的石头移动，或远离布


    # 执行同化
    # if len(rock_points) > 0 and len(paper_points) > 0:
    #     rock_points, paper_points, assimilation_occurred_rock_paper = assimilate(rock_points, paper_points)
    #     no_assimilation_counter_rock += 1 - assimilation_occurred_rock_paper
    #     no_assimilation_counter_paper += 1 - assimilation_occurred_rock_paper
    # if len(paper_points) > 0 and len(scissors_points) > 0:
    #     paper_points, scissors_points, assimilation_occurred_paper_scissors = assimilate(paper_points, scissors_points)
    #     no_assimilation_counter_paper += 1 - assimilation_occurred_paper_scissors
    #     no_assimilation_counter_scissors += 1 - assimilation_occurred_paper_scissors
    # if len(scissors_points) > 0 and len(rock_points) > 0:
    #     scissors_points, rock_points, assimilation_occurred_scissors_rock = assimilate(scissors_points, rock_points)
    #     no_assimilation_counter_scissors += 1 - assimilation_occurred_scissors_rock
    #     no_assimilation_counter_rock += 1 - assimilation_occurred_scissors_rock

    if len(rock_points) > 0 and len(paper_points) > 0:
        rock_points, paper_points = assimilate(rock_points, paper_points)     # 石头碰到布变布
    if len(paper_points) > 0 and len(scissors_points) > 0:
        paper_points, scissors_points = assimilate(paper_points, scissors_points) # 布碰到剪刀变剪刀
    if len(scissors_points) > 0 and len(rock_points) > 0:
        scissors_points, rock_points = assimilate(scissors_points, rock_points)  # 剪刀碰到石头变石头

    # 绘制图片
    if len(rock_points) > 0:
        ax.scatter(rock_points[:, 0], rock_points[:, 1], rock_points[:, 2], c='r', marker='o', label='Rock')
    if len(paper_points) > 0:
        ax.scatter(paper_points[:, 0], paper_points[:, 1], paper_points[:, 2], c='b', marker='s', label='Paper')
    if len(scissors_points) > 0:
        ax.scatter(scissors_points[:, 0], scissors_points[:, 1], scissors_points[:, 2], c='g', marker='^', label='Scissors')
    #
    # fig_size = 0.01
    # for point in rock_points:
    #     ax.imshow(rock_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')
    # for point in paper_points:
    #     ax.imshow(paper_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')
    # for point in scissors_points:
    #     ax.imshow(scissors_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')


    # for point in rock_points:
    #     plot_image_on_sphere(ax, rock_img, point)
    # for point in paper_points:
    #     plot_image_on_sphere(ax, paper_img, point)
    # for point in scissors_points:
    #     plot_image_on_sphere(ax, scissors_img, point)


    # 绘制球体的经纬线
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)

    # 添加标题，显示当前各个类别的数量
    ax.set_title(f'石头: {len(rock_points)}, 布: {len(paper_points)}, 剪刀: {len(scissors_points)}')

    # 检查停止条件
    if len(rock_points) == 0 and len(paper_points) == 0:
        ax.text2D(0.5, 0.5, '剪刀胜出！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ani.event_source.stop()
    elif len(rock_points) == 0 and len(scissors_points) == 0:
        ax.text2D(0.5, 0.5, '布胜出！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ani.event_source.stop()
    elif len(paper_points) == 0 and len(scissors_points) == 0:
        ax.text2D(0.5, 0.5, '石头胜出！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red', transform=ax.transAxes)
        ani.event_source.stop()

    return []

# 绘制初始状态
# 绘制初始状态
def init():
    ax.clear()
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # 绘制球体的经纬线
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.3)
    #
    # ax.scatter(rock_points[:, 0], rock_points[:, 1], rock_points[:, 2], c='r', marker='o', label='Rock')
    # ax.scatter(paper_points[:, 0], paper_points[:, 1], paper_points[:, 2], c='b', marker='s', label='Paper')
    # ax.scatter(scissors_points[:, 0], scissors_points[:, 1], scissors_points[:, 2], c='g', marker='^', label='Scissors')

    # 绘制图片
    if len(rock_points) > 0:
        ax.scatter(rock_points[:, 0], rock_points[:, 1], rock_points[:, 2], c='r', marker='o', label='Rock')
    if len(paper_points) > 0:
        ax.scatter(paper_points[:, 0], paper_points[:, 1], paper_points[:, 2], c='b', marker='s', label='Paper')
    if len(scissors_points) > 0:
        ax.scatter(scissors_points[:, 0], scissors_points[:, 1], scissors_points[:, 2], c='g', marker='^', label='Scissors')
    #



    # fig_size = 0.01
    # for point in rock_points:
    #     ax.imshow(rock_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')
    # for point in paper_points:
    #     ax.imshow(paper_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')
    # for point in scissors_points:
    #     ax.imshow(scissors_img, extent=(point[0]-fig_size, point[0]+fig_size, point[1]-fig_size, point[1]+fig_size), aspect='auto')


    # for point in rock_points:
    #     plot_image_on_sphere(ax, rock_img, point)
    # for point in paper_points:
    #     plot_image_on_sphere(ax, paper_img, point)
    # for point in scissors_points:
    #     plot_image_on_sphere(ax, scissors_img, point)



    # 添加标题，显示当前各个类别的数量
    ax.set_title(f'石头: {len(rock_points)}, 布: {len(paper_points)}, 剪刀: {len(scissors_points)}')

    return []

# 创建动画并暂停3秒钟
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=10, blit=True)
ani.event_source.stop()

def start_animation():
    ani.event_source.start()

# 暂停3秒后开始动画
fig.canvas.draw()
# time.sleep(3)
start_animation()

# 将动画保存为 GIF 文件
# ani.save('rock_paper_scissors.gif', writer='pillow', fps=10)

plt.show()
