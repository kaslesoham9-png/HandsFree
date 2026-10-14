# HandsFree
<img width="1476" height="989" alt="Screenshot 2026-10-13 210200" src="https://github.com/user-attachments/assets/564f6e97-cc44-446e-a19f-e7d666620b0a" />

# 🖱️ FingerMouse — Touchless Mouse Control System

**FingerMouse** is a Python-based, camera-driven human-computer interaction project that enables users to control their mouse cursor and perform essential mouse operations using **hand gestures and head movements**. By combining computer vision with real-time tracking, FingerMouse offers a touchless alternative to traditional mouse input.

The project aims to make computer interaction more intuitive, hands-free, and accessible through natural movements.

## ✨ Features

### 📷 Camera-Based Tracking
- Uses a camera to capture real-time hand and head movements.
- Tracks gestures and movements to translate physical actions into computer commands.
- Enables touchless interaction without requiring a physical mouse.

### ☝️ Index Finger Mouse Control
- Control the mouse cursor by moving your index finger.
- Translate hand movements into on-screen cursor movements in real time.
- Provides a natural and intuitive way to navigate your computer.

### 🖱️ Head Movement-Based Mouse Clicks
- **Left Click:** Turn your head to the left.
- **Right Click:** Turn your head to the right.

Perform mouse clicks through head gestures without physically pressing any buttons.

### ↕️ Head Movement-Based Scrolling
- **Scroll Up:** Move your head upward.
- **Scroll Down:** Move your head downward.

Navigate through web pages, documents, and other content using simple head movements.

## 🛠️ Technologies Used

- **Python** — Core programming language.
- **OpenCV** — Camera access and real-time computer vision.
- **MediaPipe** — Hand and facial landmark detection.
- **PyAutoGUI / Mouse Control Libraries** — Computer cursor and mouse interaction.

*The exact dependencies may vary depending on the implementation and installed packages.*

## ⚙️ How It Works

FingerMouse combines computer vision and gesture recognition to translate physical movements into computer input.

1. **Video Capture:** The camera captures live video.
2. **Landmark Detection:** Computer vision models identify hand and facial landmarks.
3. **Gesture Interpretation:** The system interprets index finger positions and head movements.
4. **Input Mapping:** Recognized movements are mapped to cursor movement, mouse clicks, and scrolling.
5. **Real-Time Interaction:** The corresponding actions are executed on the computer.

## 🚀 Getting Started

### Prerequisites

- Python installed on your computer.
- A working webcam or compatible camera.
- Required Python libraries and model assets.

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/kaslesoham9-png/HandsFree
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main Python script:

   ```bash
   python finger_mouse_gui.py
   ```
   
## 🎯 Project Objectives

- Explore real-time computer vision and gesture-based interaction.
- Reduce dependence on conventional mouse hardware.
- Develop an intuitive, touchless computer-control experience.
- Investigate alternative input methods that may support users with limited conventional mouse access.

## 🔮 Future Enhancements

- Customizable cursor sensitivity and movement smoothness.
- User calibration for improved control accuracy.
- Support for additional hand gestures and shortcuts.
- Enhanced tracking stability and reduced input latency.
- A graphical interface for configuration and personalization.
- Cross-platform compatibility and easier installation.

## 🤝 Contributing

Contributions, suggestions, and ideas are welcome! If you would like to improve FingerMouse, feel free to fork the repository, experiment with new features, and submit a pull request.

For significant changes, consider opening an issue first to discuss your proposed improvements.

## 👨‍💻 Author

**HandsFree** is a computer vision project focused on touchless human-computer interaction, combining hand tracking and head gestures to create an alternative approach to everyday computer control.

---

⭐ If you find this project interesting, consider starring the repository and sharing your feedback!

**FingerMouse — Control your computer with natural movements.**
