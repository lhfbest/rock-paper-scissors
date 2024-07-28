
class play_tools:
    def __init__(self):

        pass



    # 定义同化函数
    def assimilate(self, points1, points2, distance_threshold=0.05):
        if len(points1) == 0 or len(points2) == 0:
            return points1, points2

        new_points2 = []
        for i, point in enumerate(points1):
            distances = np.linalg.norm(points2 - point, axis=1)
            if np.any(distances < distance_threshold):
                new_points2.append(point)
        if new_points2:
            points2 = np.vstack((points2, new_points2))
            points1 = np.array(
                [point for point in points1 if not np.any(np.linalg.norm(points2 - point, axis=1) < distance_threshold)])
        return points1, points2


    # 更新点集位置
    def update_points(self,points, target_centroid):
        if len(points) == 0:
            return points
        movement = (target_centroid - points) * step_size
        points += movement
        points = np.clip(points, 0, 1)  # 确保点在1x1的单位区间内
        return points