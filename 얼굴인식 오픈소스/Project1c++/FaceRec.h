#include <iostream>
#include <string>

// OpenCV 코어
#include "opencv2\core\core.hpp"
#include "opencv2\core.hpp"
#include "opencv2\face.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\opencv.hpp"
#include<direct.h>

#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::face;

// 얼굴 검출기
CascadeClassifier face_cascade;
string filename;
string name;
int filenumber = 0;

// 얼굴을 검출하고 표시하는 함수
void detectAndDisplay(Mat frame)
{
	vector<Rect> faces;
	Mat frame_gray;
	Mat crop;
	Mat res;
	Mat gray;
	string text;
	stringstream sstm;

	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	// 얼굴 검출
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	Rect roi_b;
	Rect roi_c;

	size_t ic = 0;
	int ac = 0;

	size_t ib = 0;
	int ab = 0;

	for (ic = 0; ic < faces.size(); ic++)
	{
		roi_c.x = faces[ic].x;
		roi_c.y = faces[ic].y;
		roi_c.width = (faces[ic].width);
		roi_c.height = (faces[ic].height);

		ac = roi_c.width * roi_c.height;

		roi_b.x = faces[ib].x;
		roi_b.y = faces[ib].y;
		roi_b.width = (faces[ib].width);
		roi_b.height = (faces[ib].height);


		crop = frame(roi_b);
		resize(crop, res, Size(128, 128), 0, 0, INTER_LINEAR);
		cvtColor(crop, gray, COLOR_BGR2GRAY);
		stringstream ssfn;
		filename = "C:\\Users\\Asus\\Desktop\\Faces\\";
		ssfn << filename.c_str() << name << filenumber << ".jpg";
		filename = ssfn.str();
		imwrite(filename, res);
		filenumber++;


		Point pt1(faces[ic].x, faces[ic].y);
		Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
		rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
	}


	sstm << "자른 영역 크기: " << roi_b.width << "x" << roi_b.height << " 파일명: " << filename;
	text = sstm.str();

	if (!crop.empty())
	{
		imshow("검출된 얼굴", crop);
	}
	else
		destroyWindow("검출된 얼굴");

}

// 얼굴을 추가하는 함수
void addFace()
{
	cout << "\n이름을 입력하세요:  ";
	cin >> name;

	VideoCapture capture(0);

	if (!capture.isOpened())
		return;

	if (!face_cascade.load("C:\\Users\\Asus\\Downloads\\opencv4.1\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml"))
	{
		cout << "에러" << endl;
		return;
	};

	Mat frame;
	cout << "\n카메라 앞에 얼굴을 10번 찍어주세요, 'C'를 10번 누르세요";
	char key;
	int i = 0;

	for (;;)
	{
		capture >> frame;

		detectAndDisplay(frame);
		i++;
		if (i == 10)
		{
			cout << "얼굴이 추가되었습니다";
			break;
		}

		int c = waitKey(10);

		if (27 == char(c))
		{
			break;
		}
	}

	return;
}


static void dbread(vector<Mat>& images, vector<int>& labels) {
	vector<cv::String> fn;
	filename = "C:\\Users\\Asus\\Desktop\\Faces\\";
	glob(filename, fn, false);

	size_t count = fn.size();

	for (size_t i = 0; i < count; i++)
	{
		string itsname = "";
		char sep = '\\';
		size_t j = fn[i].rfind(sep, fn[i].length());
		if (j != string::npos)
		{
			itsname = (fn[i].substr(j + 1, fn[i].length() - j - 6));
		}
		images.push_back(imread(fn[i], 0));
		labels.push_back(atoi(itsname.c_str()));
	}
}

// EigenFace로 트레이닝하는 함수
void eigenFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;
	dbread(images, labels);
	cout << "이미지의 크기는 " << images.size() << "입니다" << endl;
	cout << "라벨의 크기는 " << labels.size() << "입니다" << endl;
	cout << "트레이닝을 시작합니다...." << endl;

	// EigenFace Recognizer 생성
	Ptr<EigenFaceRecognizer> model = EigenFaceRecognizer::create();

	// 데이터 트레이닝
	model->train(images, labels);

	model->save("C:\\Users\\Asus\\Desktop\\eigenface.yml");

	cout << "트레이닝이 완료되었습니다...." << endl;
	waitKey(10000);
}

// 얼굴 인식하는 함수
void  FaceRecognition() {

	cout << "인식을 시작합니다..." << endl;

	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();
	model->read("C:\\Users\\Asus\\Desktop\\eigenface.yml");

	Mat testSample = imread("C:\\Users\\Asus\\Desktop\\0.jpg", 0);

	int img_width = testSample.cols;
	int img_height = testSample.rows;


	string window = "캡처 - 얼굴 검출";

	if (!face_cascade.load("C:\\Users\\Asus\\Downloads\\opencv4.1\\build\\install\\etc\\haarcascades\\haarcascade_frontalface_alt.xml")) {
		cout << " 파일을 불러오는 중 오류 발생" << endl;
		return;
	}

	VideoCapture cap(0);

	if (!cap.isOpened())
	{
		cout << "종료" << endl;
		return;
	}

	namedWindow(window, 1);
	long count = 0;
	string Pname = "";

	while (true)
	{
		vector<Rect> faces;
		Mat frame;
		Mat graySacleFrame;
		Mat original;

		cap >> frame;
		count = count + 1;

		if (!frame.empty()) {

			original = frame.clone();

			cvtColor(original, graySacleFrame, COLOR_BGR2GRAY);

			face_cascade.detectMultiScale(graySacleFrame, faces, 1.1, 3, 0, cv::Size(90, 90));

			std::string frameset = std::to_string(count);
			std::string faceset = std::to_string(faces.size());

			int width = 0, height = 0;

			for (int i = 0; i < faces.size(); i++)
			{
				Rect face_i = faces[i];

				Mat face = graySacleFrame(face_i);

				Mat face_resized;
				cv::resize(face, face_resized, Size(img_width, img_height), 1.0, 1.0, INTER_CUBIC);

				int label = -1; double confidence = 0;
				model->predict(face_resized, label, confidence);

				cout << " 신뢰도 " << confidence << " 라벨: " << label << endl;

				Pname = to_string(label);

				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				string text = Pname;

				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);

				putText(original, text, Point(pos_x, pos_y), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			}


			putText(original, "프레임: " + frameset, Point(30, 60), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
			putText(original, "검출된 인물 수: " + to_string(faces.size()), Point(30, 90), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

			cv::imshow(window, original);
		}
		if (waitKey(30) >= 0) break;
	}
}
