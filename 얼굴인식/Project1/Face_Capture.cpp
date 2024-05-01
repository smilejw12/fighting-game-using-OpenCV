#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>

using namespace cv;
using namespace std;

void captureFaces(const string& cascadePath, const string& outputFolder) {
    CascadeClassifier face_cascade;
    if (!face_cascade.load(cascadePath)) {
        cerr << "Error loading face cascade\n";
        return;
    }

    VideoCapture capture(0);
    if (!capture.isOpened()) {
        cerr << "Error opening video capture\n";
        return;
    }

    Mat frame, gray;
    vector<Rect> faces;
    string filename;
    int img_counter = 0;

    while (capture.read(frame)) {
        if (frame.empty()) {
            cerr << "No captured frame\n";
            break;
        }
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            Mat faceROI = frame(face);
            if (!faceROI.empty()) {
                filename = outputFolder + "/face_" + to_string(img_counter++) + ".jpg";
                imwrite(filename, faceROI);
            }
        }

        imshow("Capture - Face detection", frame);
        char c = (char)waitKey(10);
        if (c == 27 || img_counter >= 30) break; // Stop after 30 faces or on escape key press
    }
    capture.release();
    destroyAllWindows();
}

int main() {
    // Set the path to the face cascade file
    string cascadePath = "C:/OpenCV_4.7/build/install/etc/haarcascades/haarcascade_frontalface_alt.xml";
    // Set the path to the output folder for saving faces
    string outputFolder = "C:/Users/82105/Desktop/프로젝트/fighting-game-using-OpenCV/얼굴인식/Project1/pictures for readme";

    // Ensure output folder exists
    filesystem::create_directories(outputFolder);

    captureFaces(cascadePath, outputFolder);
    cout << "Face capture complete. Faces saved to " << outputFolder << endl;

    return 0;
}
