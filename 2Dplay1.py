import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg
from matplotlib import rcParams

# 设置字体以支持中文显示
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 初始化点集数量
num_points = 10

# 生成初始点集
rock_points = np.random.rand(num_points, 2)
paper_points = np.random.rand(num_points, 2)
scissors_points = np.random.rand(num_points, 2)

# 定义运动步长
step_size = 0.1
boundary_margin = 0.02  # 定义一个边界范围，当点接近边界时处理滑动

def assimilate(points1, points2, distance_threshold=0.02):
    if len(points1) == 0 or len(points2) == 0:
        return points1, points2

    # 新的同化点集
    new_points2 = []

    # 需要被移除的点的索引
    remove_indices = []

    # 检查每个点1是否与点2集中的点距离小于阈值
    for i, point1 in enumerate(points1):
        distances = np.linalg.norm(points2 - point1, axis=1)
        if np.any(distances < distance_threshold):
            new_points2.append(point1)  # 如果距离小于阈值，则添加到新的点集
            remove_indices.append(i)  # 记录需要移除的点的索引

    # 如果有新的点被同化，更新points2和points1
    if new_points2:
        # 将新同化的点添加到points2
        points2 = np.vstack((points2, new_points2))
        # 从points1中移除已同化的点
        points1 = np.delete(points1, remove_indices, axis=0)

    # 打印同化的点数
    if len(new_points2) > 0:
        print('同化了 %d 点' % len(new_points2))

    return points1, points2



# 更新点集位置，朝着最近的猎物移动，或远离最近的天敌
def update_points_attack(points, target_points, enemy_points, step_size=0.01):
    if len(points) == 0:
        return points
    for i, point in enumerate(points):
        if len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.linalg.norm(target_points - point, axis=1)
            nearest_target = target_points[np.argmin(distances)]
            move_vector = (nearest_target - point) * step_size
        elif len(enemy_points) > 0:
            # 远离最近的天敌
            distances = np.linalg.norm(enemy_points - point, axis=1)
            nearest_enemy = enemy_points[np.argmin(distances)]
            move_vector = (point - nearest_enemy) * step_size
        else:
            continue
        points[i] += move_vector
        # 检查并处理边界

        new_point = point + move_vector * step_size
        if new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin:
            move_vector[0] = 0  # 沿Y轴滑动
        if new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin:
            move_vector[1] = 0  # 沿X轴滑动

        escape_strength = 1
        # 如果在死角，随机选择一个方向沿着边界滑动
        if (new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin) and \
                (new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin):
            if np.random.rand() > 0.5:
                move_vector[0] = 0  # 随机选择沿X轴或Y轴滑动
            else:
                move_vector[1] = 0

            escape_strength = 1 + 0.9 * np.random.rand()  # 随机增加逃逸强度

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        points[i] += move_vector * step_size * escape_strength

    points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
    return points


# 更新点集位置，朝着最近的猎物移动，或远离最近的天敌
def update_points_attack_and_run_nearest(points, target_points, enemy_points, step_size=0.01):
    if len(points) == 0:
        return points
    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0])
        if len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.linalg.norm(target_points - point, axis=1)
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length ** 2  # 影响力反比例缩放
        if len(enemy_points) > 0:
            # 远离最近的天敌
            distances = np.linalg.norm(enemy_points - point, axis=1)
            nearest_enemy = enemy_points[np.argmin(distances)]
            enemy_vector = point - nearest_enemy
            enemy_length = np.linalg.norm(enemy_vector)
            if enemy_length > 0:
                move_vector += enemy_vector / enemy_length ** 2  # 影响力反比例缩放
        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量
        # points[i] += move_vector * step_size

        # 检查并处理边界
        new_point = point + move_vector * step_size
        if new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin:
            move_vector[0] = 0  # 沿Y轴滑动
        if new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin:
            move_vector[1] = 0  # 沿X轴滑动

        escape_strength = 1
        # 如果在死角，随机选择一个方向沿着边界滑动
        if (new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin) and \
           (new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin):
            move_vector = np.array([0.5, 0.5]) - point
            if abs(move_vector[0]) > abs(move_vector[1]):
                move_vector[1] = 0  # 主要沿X轴滑动
            else:
                move_vector[0] = 0  # 主要沿Y轴滑动

            escape_strength = 1 + 0.9 * np.random.rand()  # 随机增加逃逸强度

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        points[i] += move_vector * step_size * escape_strength


    points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
    return points



# 更新点集位置，当天敌存在时远离天敌，天敌消失时朝向猎物移动
def update_points_run_nearest_than_attack_nearst(points, target_points, enemy_points,step_size):
    if len(points) == 0:
        return points
    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0])
        if len(enemy_points) > 0:
            # 远离最近的天敌
            distances = np.linalg.norm(enemy_points - point, axis=1)
            nearest_enemy = enemy_points[np.argmin(distances)]
            enemy_vector = point - nearest_enemy
            enemy_length = np.linalg.norm(enemy_vector)
            if enemy_length > 0  and  enemy_length < 0.6:
                move_vector += enemy_vector / enemy_length**2  # 影响力反比例缩放
        elif len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.linalg.norm(target_points - point, axis=1)
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length**2  # 影响力反比例缩放

        # points[i] += move_vector * step_size

        # 检查并处理边界
        new_point = point + move_vector * step_size
        if new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin:
            move_vector[0] = 0  # 沿Y轴滑动
        if new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin:
            move_vector[1] = 0  # 沿X轴滑动

        escape_strength = 1
        # 如果在死角，随机选择一个方向沿着边界滑动
        if (new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin) and \
                (new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin):
            move_vector = np.array([0.5, 0.5]) - point
            if abs(move_vector[0]) > abs(move_vector[1]):
                move_vector[1] = 0  # 主要沿X轴滑动
            else:
                move_vector[0] = 0  # 主要沿Y轴滑动

            escape_strength = 1 + 2 * np.random.rand()  # 随机增加逃逸强度

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        points[i] += move_vector * step_size * escape_strength


    points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
    return points


def update_points_run_all_than_attack_nearest(points, target_points, enemy_points,step_size,include_boundary_repulsion=True):
    if len(points) == 0:
        return points
    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0])
        if len(enemy_points) > 0:
            # 计算所有天敌对其的斥力
            for enemy in enemy_points:
                enemy_vector = point - enemy
                enemy_length = np.linalg.norm(enemy_vector)
                if enemy_length > 0:
                    move_vector += enemy_vector / enemy_length ** 2  # 斥力反比例缩放


        elif len(target_points) > 0:
            # 朝着最近的猎物移动
            distances = np.linalg.norm(target_points - point, axis=1)
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length ** 2  # 影响力反比例缩放

        # # 加入最近边界点斥力
        # if include_boundary_repulsion:
        #     boundary_point = nearest_boundary_point(point)
        #     boundary_vector = point - boundary_point
        #     boundary_length = np.linalg.norm(boundary_vector)
        #     if boundary_length > 0:
        #         move_vector += boundary_vector / (boundary_length ** 2 + 0.01)  # 斥力计算，加0.01防止除以0


        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 检查并处理边界
        new_point = point + move_vector * step_size
        if new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin:
            move_vector[0] = 0  # 沿Y轴滑动
        if new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin:
            move_vector[1] = 0  # 沿X轴滑动

        escape_strength = 1
        # 如果在死角，随机选择一个方向沿着边界滑动
        if (new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin) and \
                (new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin):
            move_vector = np.array([0.5, 0.5]) - point
            if abs(move_vector[0]) > abs(move_vector[1]):
                move_vector[1] = 0  # 主要沿X轴滑动
            else:
                move_vector[0] = 0  # 主要沿Y轴滑动

            escape_strength = 1 + 2 * np.random.rand()  # 随机增加逃逸强度

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        points[i] += move_vector * step_size * escape_strength

    points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
    return points





def update_points_condition_run_all_than_attack_nearest(points, target_points, enemy_points,step_size,include_boundary_repulsion=True):
    if len(points) == 0:
        return points
    for i, point in enumerate(points):
        move_vector = np.array([0.0, 0.0])
        if len(enemy_points) > 0:
            # 计算所有天敌对其的斥力
            for enemy in enemy_points:
                enemy_vector = point - enemy
                enemy_length = np.linalg.norm(enemy_vector)
                if enemy_length > 0:
                    move_vector += enemy_vector / enemy_length ** 2  # 斥力反比例缩放


        if len(enemy_points) ==0 or (len(target_points) > 0  and  len(points)<8):
            # 朝着最近的猎物移动
            distances = np.linalg.norm(target_points - point, axis=1)
            nearest_target = target_points[np.argmin(distances)]
            target_vector = nearest_target - point
            target_length = np.linalg.norm(target_vector)
            if target_length > 0:
                move_vector += target_vector / target_length ** 2  # 影响力反比例缩放

        # 加入最近边界点斥力
        if include_boundary_repulsion:
            boundary_point = nearest_boundary_point(point)
            boundary_vector = point - boundary_point
            boundary_length = np.linalg.norm(boundary_vector)
            if boundary_length > 0:
                move_vector += boundary_vector * 0.1 / (boundary_length ** 2 + 0.01)  # 斥力计算，加0.01防止除以0


        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        # 检查并处理边界
        new_point = point + move_vector * step_size
        if new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin:
            move_vector[0] = 0  # 沿Y轴滑动
        if new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin:
            move_vector[1] = 0  # 沿X轴滑动

        escape_strength = 1
        # 如果在死角，随机选择一个方向沿着边界滑动
        if (new_point[0] < boundary_margin or new_point[0] > 1 - boundary_margin) and \
                (new_point[1] < boundary_margin or new_point[1] > 1 - boundary_margin):
            move_vector = np.array([0.5, 0.5]) - point
            if abs(move_vector[0]) > abs(move_vector[1]):
                move_vector[1] = 0  # 主要沿X轴滑动
            else:
                move_vector[0] = 0  # 主要沿Y轴滑动

            escape_strength = 1 + 2 * np.random.rand()  # 随机增加逃逸强度

        if np.linalg.norm(move_vector) > 0:
            move_vector = move_vector / np.linalg.norm(move_vector)  # 归一化向量

        points[i] += move_vector * step_size * escape_strength

    points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
    return points




def nearest_boundary_point(point):
    """ 计算每个点到最近边界的最近点 """
    x, y = point
    boundary_points = np.array([
        [0, y],  # 最近的左边界点
        [1, y],  # 最近的右边界点
        [x, 0],  # 最近的下边界点
        [x, 1]  # 最近的上边界点
    ])

    # 计算到这四个边界点的距离并找出最近的
    distances = np.linalg.norm(boundary_points - point, axis=1)
    nearest_index = np.argmin(distances)
    return boundary_points[nearest_index]





# 载入图片
rock_img = mpimg.imread('C:\others\play\石头2.jpg')
paper_img = mpimg.imread('C:\others\play\布2.jpg')
scissors_img = mpimg.imread('C:\others\play\剪刀2.jpg')

fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')  # 保持比例，可以尝试 'box' 或 'datalim'
mng = plt.get_current_fig_manager()
# mng.resize(*mng.window.maxsize())
ax = fig.add_subplot(111)
ax.set_aspect('equal', 'box')  # 保持比例

mng.window.wm_geometry("+500+50")

def animate(frame):
    global rock_points, paper_points, scissors_points

    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


    # 更新点的位置
    speed = 0.03
    # if len(rock_points) > 9:
    #     rock_points = update_points_run_all_than_attack_nearest(rock_points,  scissors_points, paper_points,speed)  # 石头朝最近的布移动，或远离剪刀
    # else:
    #     rock_points = update_points_attack_and_run_nearest(rock_points,  scissors_points, paper_points,speed)  # 石头朝最近的布移动，或远离剪刀

    # 获取前一半的点
    half_index = len(rock_points) // 2  # 使用整除来获取中间索引
    rock_points[:half_index] = update_points_run_all_than_attack_nearest(rock_points[:half_index] ,  scissors_points, paper_points,speed)  # 石头朝最近的布移动，或远离剪刀
    rock_points[half_index:]  = update_points_attack_and_run_nearest(rock_points[half_index:],  scissors_points, paper_points,speed)  # 石头朝最近的布移动，或远离剪刀


    paper_points = update_points_attack_and_run_nearest(paper_points,  rock_points,scissors_points,speed)  # 布朝最近的剪刀移动，或远离石头
    scissors_points = update_points_attack_and_run_nearest(scissors_points, paper_points, rock_points,speed)  # 剪刀朝最近的石头移动，或远离布

    # 执行同化
    rock_points, paper_points = assimilate(rock_points, paper_points)  # 石头碰到布变布
    paper_points, scissors_points = assimilate(paper_points, scissors_points)  # 布碰到剪刀变剪刀
    scissors_points, rock_points = assimilate(scissors_points, rock_points)  # 剪刀碰到石头变石头

    fig_size = 0.02
    # 绘制图片
    for point in rock_points:
        ax.imshow(rock_img, extent=(point[0] - fig_size, point[0] + fig_size, point[1] - fig_size, point[1] + fig_size), aspect='auto',alpha=1)
    for point in paper_points:
        ax.imshow(paper_img, extent=(point[0] - fig_size, point[0] + fig_size, point[1] - fig_size, point[1] + fig_size), aspect='auto',alpha=1)
    for point in scissors_points:
        ax.imshow(scissors_img, extent=(point[0] - fig_size, point[0] + fig_size, point[1] - fig_size, point[1] + fig_size),
                  aspect='auto',alpha=1)

    # 添加标题，显示当前各个类别的数量
    ax.set_title(f'石头：一半苟 一半开拓（按照当前点集对半分）'
                 f'\n石头: {len(rock_points)}   剪刀: {len(scissors_points)}   布: {len(paper_points)} ')
    # if len(rock_points)>9:
    #     ax.set_title(f'石头：稳辣，苟！'
    #                  f'\n石头: {len(rock_points)}   剪刀: {len(scissors_points)}   布: {len(paper_points)} ')
    # else:
    #     ax.set_title(f'石头：不稳，开拓！'
    #                  f'\n石头: {len(rock_points)}   剪刀: {len(scissors_points)}   布: {len(paper_points)} ')
    # 检查停止条件
    if len(rock_points) == 0 and len(paper_points) == 0:
        ax.text(0.5, 0.5, '剪刀  win！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red',
                transform=ax.transAxes)
        ani.event_source.stop()
    elif len(rock_points) == 0 and len(scissors_points) == 0:
        ax.text(0.5, 0.5, '布  win！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red',
                transform=ax.transAxes)
        ani.event_source.stop()
    elif len(paper_points) == 0 and len(scissors_points) == 0:
        ax.text(0.5, 0.5, '石头  win！', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red',
                transform=ax.transAxes)
        ani.event_source.stop()

    return []



ani = animation.FuncAnimation(fig, animate, frames=200, interval=10, blit=True)  # 刷新间隔0.1秒

plt.show()