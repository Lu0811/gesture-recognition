import cv2
import numpy as np
import time

class HandDetector:
    def __init__(self, hsv_file, width, height):
        self.scale = 2
        self.smallest_area = 600.0
        self.min_finger_depth = 20
        self.max_finger_angle = 60
        self.min_thumb = 120
        self.max_thumb = 200
        self.min_index = 60
        self.max_index = 120

        self.cog_pt = None
        self.prev_cog_pt = None
        self.contour_axis_angle = None
        self.finger_tips = []
        self.named_fingers = []
        self.last_gesture = None

        self.hue_lower, self.hue_upper, self.sat_lower, self.sat_upper, self.val_lower, self.val_upper = self.read_hsv_ranges(hsv_file)
        self.width, self.height = width // self.scale, height // self.scale

    def read_hsv_ranges(self, file):
        with open(file) as f:
            lines = f.readlines()
            hue_lower, hue_upper = map(int, lines[0].split()[1:])
            sat_lower, sat_upper = map(int, lines[1].split()[1:])
            val_lower, val_upper = map(int, lines[2].split()[1:])
        return hue_lower, hue_upper, sat_lower, sat_upper, val_lower, val_upper

    def update(self, frame):
        frame_resized = cv2.resize(frame, (self.width, self.height))
        hsv_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2HSV)

        # Aplicar filtro Gaussiano
        hsv_img = cv2.GaussianBlur(hsv_img, (5, 5), 0)

        mask = cv2.inRange(hsv_img, (self.hue_lower, self.sat_lower, self.val_lower),
                           (self.hue_upper, self.sat_upper, self.val_upper))

        # Operaciones morfológicas adicionales
        mask = cv2.erode(mask, None, iterations=3)
        mask = cv2.dilate(mask, None, iterations=3)

        cv2.imshow("Mask", mask)  # Mostrar la máscara para depuración

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.prev_cog_pt = None
            return

        big_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(big_contour) < self.smallest_area:
            self.prev_cog_pt = None
            return

        self.extract_contour_info(big_contour, self.scale)
        self.find_finger_tips(big_contour, self.scale)
        self.detect_swipe_gesture(frame)

    def extract_contour_info(self, contour, scale):
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            x_center = int(moments['m10'] / moments['m00']) * scale
            y_center = int(moments['m01'] / moments['m00']) * scale
            self.cog_pt = (x_center, y_center)

        self.contour_axis_angle = self.calculate_tilt(moments)

    def calculate_tilt(self, moments):
        m11 = moments['mu11']
        m20 = moments['mu20']
        m02 = moments['mu02']

        diff = m20 - m02
        if diff == 0:
            return 45 if m11 != 0 else 0

        theta = 0.5 * np.arctan2(2 * m11, diff)
        tilt = np.degrees(theta)

        if diff > 0 and m11 == 0:
            return 0
        if diff < 0 and m11 == 0:
            return -90
        if diff > 0 and m11 > 0:
            return tilt
        if diff > 0 and m11 < 0:
            return 180 + tilt
        if diff < 0 and m11 > 0:
            return tilt
        if diff < 0 and m11 < 0:
            return 180 + tilt

        return 0

    def find_finger_tips(self, contour, scale):
        approx_contour = cv2.approxPolyDP(contour, 3, True)
        hull = cv2.convexHull(approx_contour, returnPoints=False)
        defects = cv2.convexityDefects(approx_contour, hull)

        if defects is None:
            return

        tip_pts = []
        fold_pts = []
        depths = []

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx_contour[s][0] * scale)
            end = tuple(approx_contour[e][0] * scale)
            far = tuple(approx_contour[f][0] * scale)

            if d / 256 < self.min_finger_depth:
                continue

            tip_pts.append(start)
            fold_pts.append(far)
            depths.append(d / 256)

        self.reduce_tips(len(tip_pts), tip_pts, fold_pts, depths)

    def reduce_tips(self, num_points, tip_pts, fold_pts, depths):
        self.finger_tips = []

        for i in range(num_points):
            if depths[i] < self.min_finger_depth:
                continue

            pdx = (i - 1) % num_points
            sdx = (i + 1) % num_points
            angle = self.angle_between(tip_pts[i], fold_pts[pdx], fold_pts[sdx])
            if angle >= self.max_finger_angle:
                continue

            self.finger_tips.append(tip_pts[i])

    def angle_between(self, tip, next_pt, prev_pt):
        return np.abs(np.degrees(np.arctan2(next_pt[1] - tip[1], next_pt[0] - tip[0]) -
                                 np.arctan2(prev_pt[1] - tip[1], prev_pt[0] - tip[0])))

    def detect_swipe_gesture(self, frame):
        if self.prev_cog_pt is None:
            self.prev_cog_pt = self.cog_pt
            return

        dx = self.cog_pt[0] - self.prev_cog_pt[0]
        dy = self.cog_pt[1] - self.prev_cog_pt[1]

        if abs(dx) > 50 and abs(dy) < 20:  # Thresholds for detecting a swipe gesture
            if dx > 0:
                self.last_gesture = 'Swipe Right'
            else:
                self.last_gesture = 'Swipe Left'

        self.prev_cog_pt = self.cog_pt

    def detect_like_gesture(self):
            if len(self.finger_tips) == 1:
                thumb_tip = self.finger_tips[0]
                if self.cog_pt and thumb_tip[1] < self.cog_pt[1]:  # Check if the thumb is above the center of gravity
                    self.last_gesture = 'Like'

    def draw(self, frame):
        if not self.finger_tips:
            return

        # Contador de dedos
        finger_count = len(self.finger_tips)

        for pt in self.finger_tips:
            cv2.circle(frame, pt, 8, (0, 255, 0), 2)
            cv2.line(frame, self.cog_pt, pt, (0, 255, 0), 2)

        cv2.circle(frame, self.cog_pt, 8, (0, 0, 255), -1)

        # Mostrar el contador de dedos
        cv2.putText(frame, f'Fingers: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Ultimo gesto
        cv2.putText(frame, f'Last Gesture: {self.last_gesture}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Identificar poses
        if self.last_gesture == 'Like':
            cv2.putText(frame, 'Like', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 0:
            cv2.putText(frame, 'Fist', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 1:
            cv2.putText(frame, 'Pointing', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 2:
            cv2.putText(frame, 'Peace', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 3:
            cv2.putText(frame, 'Three', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 4:
            cv2.putText(frame, 'Four', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        elif finger_count == 5:
            cv2.putText(frame, 'Palm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
