import cv2
import mediapipe as mp
import pyautogui
import MouseController

previous_x, previous_y = None, None
smoothing_factor = 0.30

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
controller = MouseController.MouseController()
chosen_hand = "Right"

cap = cv2.VideoCapture(0)

def recognize_gesture(hand_landmarks, image):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    index_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_dip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_dip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

    if isScrolling(index_tip.x, index_tip.y, index_dip.x, index_dip.y, index_pip.y,
                   middle_tip.x, middle_tip.y, middle_dip.x, middle_dip.y, middle_pip.y,
                   pinky_tip.y, pinky_dip.y, pinky_pip.y, ring_tip.y, ring_dip.y, ring_pip.y):
        return "Scrolling"
    if isLeftClicking(thumb_tip.y, index_tip.y):
        return "Left Click"
    if isRightClicking(thumb_tip.y, middle_tip.y):
        return "Right Click"
    return "N/A"

def isLeftClicking(thumb_tip, index_tip):
    return abs(thumb_tip - index_tip) < 0.05

def isRightClicking(thumb_tip, middle_tip):
    return abs(thumb_tip - middle_tip) < 0.05

def isScrolling(index_tip_x, index_tip_y, index_dip_x, index_dip_y, index_pip_y,
                middle_tip_x, middle_tip_y, middle_dip_x, middle_dip_y, middle_pip_y,
                pinky_tip, pinky_dip, pinky_pip, ring_tip, ring_dip, ring_pip):
    diff_tip = abs(index_tip_x - middle_tip_x)
    diff_dip = abs(index_dip_x - middle_dip_x)

    isPinkyLower = pinky_tip > pinky_pip and pinky_dip > pinky_pip
    isRingLower = ring_tip > ring_pip and ring_dip > ring_pip
    isIndexAbove = index_tip_y < index_pip_y and index_dip_y < index_pip_y
    isMiddleAbove = middle_tip_y < middle_pip_y and middle_dip_y < middle_pip_y

    return diff_tip < 0.05 and diff_dip < 0.05 and isPinkyLower and isRingLower and isIndexAbove and isMiddleAbove

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == chosen_hand:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    for landmark in hand_landmarks.landmark:
                        h, w, _ = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

                    for connection in mp_hands.HAND_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = hand_landmarks.landmark[start_idx]
                        end = hand_landmarks.landmark[end_idx]
                        start_point = (int(start.x * w), int(start.y * h))
                        end_point = (int(end.x * w), int(end.y * h))
                        cv2.line(image, start_point, end_point, (0, 0, 0), 2)

                    scaling_factor = 6.0
                    index_tip_x = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x +
                                   hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x) / 2
                    index_tip_y = (hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y +
                                   hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y) / 2

                    screen_width, screen_height = pyautogui.size()
                    x = int((index_tip_x * screen_width - (screen_width * 0.5)) * scaling_factor) - screen_width
                    y = int((index_tip_y * screen_height - (screen_height * 0.7)) * scaling_factor)

                    if previous_x is not None and previous_y is not None:
                        x = int(previous_x * (1 - smoothing_factor) + x * smoothing_factor)
                        y = int(previous_y * (1 - smoothing_factor) + y * smoothing_factor)

                    pyautogui.moveTo(x, y)
                    previous_x, previous_y = x, y

                    gesture = recognize_gesture(hand_landmarks, image)
                    if gesture == "Left Click":
                        controller.click()

                    scroll_direction = -10
                    if gesture == "Scrolling":
                        controller.scroll(scroll_direction)

                    cv2.putText(image, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
