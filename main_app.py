import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

# mediapipe
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()


# def for if person is there
def is_present(results):
    """تشخیص حضور فرد در محل"""
    return results.pose_landmarks is not None

# def for sitting status
def is_sitting(landmarks):
    """تشخیص حالت نشسته با استفاده از موقعیت نقاط کلیدی بدن"""
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
    knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
    return shoulder_y < hip_y < knee_y


# def for lying status
def is_lying_down(landmarks):
    """تشخیص حالت خوابیده با استفاده از موقعیت نقاط کلیدی بدن"""
    head_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    hip_y = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y

    # if landmarks are in the same position wich means employee is lying down
    return abs(head_y - shoulder_y) < 0.05 and abs(shoulder_y - hip_y) < 0.05


# camera start
video_path = r"C:\Users\Ali\PycharmProjects\EmployeeControl\a.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if is_present(results):
        # detect important landmarks
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        # وضعیت نشسته، خوابیده یا غیرنشسته
        if is_lying_down(landmarks):
            color = (0, 0, 255)  # قرمز
            status = "Alert: Lying Down"
        elif is_sitting(landmarks):
            color = (0, 255, 0)  # سبز
            status = "Sitting"
        else:
            color = (0, 0, 255)  # قرمز
            status = "Alert: Not Sitting"

        # detect person status
        x = int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * w)
        y = int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * h)

        # drawing a status around the employees
        cv2.rectangle(image, (x - 100, y - 100), (x + 100, y + 100), color, 2)
        cv2.putText(image, f"Employee 1: {status}", (x - 100, y - 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        # if person is out of range
        color = (0, 0, 255)  # قرمز
        status = "Alert: Left Workplace"
        cv2.putText(image, f"Employee 1: {status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Employee Monitoring", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
