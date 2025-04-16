import cv2
import mediapipe as mp
import time
import pyautogui
import os

hand_tracker = mp.solutions.hands.Hands()
draw_tools = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

last_frame_time = 0
last_action_time = 0
action_cooldown = 3

def count_raised_fingers(hand_landmarks, hand_side):
    fingertip_ids = [4, 8, 12, 16, 20]
    base_joint_ids = [2, 6, 10, 14, 18]
    raised_fingers = []

    if hand_side == "Right":
        thumb_raised = hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x
    else:
        thumb_raised = hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x
    raised_fingers.append(1 if thumb_raised else 0)

    for tip_id, base_id in zip(fingertip_ids[1:], base_joint_ids[1:]):
        finger_raised = hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[base_id].y
        raised_fingers.append(1 if finger_raised else 0)

    return sum(raised_fingers)

def do_action(fingers_count):
    if fingers_count == 0:
        pyautogui.press('volumemute')
        print("🔇 Muted the sound")
    elif fingers_count == 1:
        pyautogui.press('volumedown')
        print("🔉 Turned volume down")
    elif fingers_count == 2:
        pyautogui.press('volumeup')
        print("🔊 Turned volume up")
    elif fingers_count == 3:
        os.system("notepad.exe")
        print("📝 Opened Notepad")
    elif fingers_count == 4:
        os.system("explorer.exe")
        print("📂 Opened File Explorer")
    elif fingers_count == 5:
        print("😶 Nothing happens")

while True:
    success, frame = webcam.read()
    if not success:
        print("Webcam not working!")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_tracker.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(results.multi_hand_landmarks, results.multi_handedness):
            draw_tools.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            hand_side = hand_info.classification[0].label
            fingers_count = count_raised_fingers(hand_landmarks, hand_side)
            cv2.putText(frame, f'{fingers_count} fingers raised', (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            current_time = time.time()
            if current_time - last_action_time > action_cooldown:
                do_action(fingers_count)
                last_action_time = current_time

    current_time = time.time()
    fps = 1 / (current_time - last_frame_time) if current_time != last_frame_time else 0
    last_frame_time = current_time
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()