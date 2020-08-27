# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        # STEP 1: at each time T, firstly we predict x' of each Track obj with KF
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            # for each obj, predict state on time T with KF based on t-1
            track.predict(self.kf)  # 只更新KF的参数mean variance, appearance feature 还是之前的

    def update(self, detections):
        # STEP 2: Then we update
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
            each Detection obj maintain the location(bbox_tlwh), confidence(conf), and appearance feature
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)     # cascade matching(appearance) + IOU matching

        # Update track set.
        # MT: track成功, 根据当前的观测detection，更新KF 矩阵, 加入当前的 appearance feature
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        # UT: 丢失的track，标记miss
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # UD: 新建新的track, 分类新ID
        for detection_idx in unmatched_detections:
            #
            self._initiate_track(detections[detection_idx])     # 新建Track obj, 加入self.track list

        # 舍弃删除的track
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]      # 保留confirmed track 的ID

        features, targets = [], []
        for track in self.tracks:           # 对所有confirmed 的 track object
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features] # 有几个feature，就copy ID 几次
            track.features = []     # 清空当前track obj 的feature, 下次update的时候添加一个feature

        # 对所有confirmed 的track 进行部分拟合，用新的数据更新测量距离
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        # 基于外观信息和马氏距离，计算卡尔曼滤波预测的tracks和当前时刻检测到的detections的代价矩阵
        def gated_metric(tracks, dets, track_indices, detection_indices):
            # Tracks
            features = np.array([dets[i].feature for i in detection_indices])   # 当前帧观测到的appearance feature
            # (检测到的人数, 512)

            targets = np.array([tracks[i].track_id for i in track_indices])     #

            # 基于外观信息，计算tracks和detections的余弦距离代价矩阵
            cost_matrix = self.metric.distance(features, targets)   # (track 中的人数，检测到的人数)

            # 基于马氏距离，过滤掉代价矩阵中一些不合适的项 (将其设置为一个较大的值)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        """
        KF predict 
            -- confirmed 
                Matching_Cascade (appearance feature + distance)
                    -- matched Tracks  成功匹配
                    -- unmatched tracks
                        -- 
                    -- unmatched detection
            -- unconfirmed 
        """

        # Split track set into confirmed and unconfirmed tracks. ********************************************
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]   # confirmed: directly apply Matching_Cascade
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]   # unconfirmed: directly go to IOU match

        # Associate confirmed tracks using appearance features.(Matching_Cascade) ***************************
        # 外观特征 + 马氏距离筛选     仅对confirmed track
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU *****************
        # for IOU match: unconfirmed + u
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]  # # 刚刚没有匹配上

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        # IOU matching *************************************************************************************
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b # associated matching + IOU matching
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):

        mean, covariance = self.kf.initiate(detection.to_xyah())    # 根据位置初始化KF

        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature)) # for new obj, create a new Track object for it
        self._next_id += 1
