#include<iostream>
#include<stdlib.h>
#include<stdio.h>

// OpenCV 라이브러리
#include<opencv2\objdetect\objdetect.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include "opencv2\core.hpp"
#include <opencv2\core\core.hpp>
#include <opencv2\opencv.hpp>
#include "opencv2/face.hpp"

// 파일 처리를 위한 라이브러리
#include <fstream>
#include <sstream>

#include "FaceRec.h"

using namespace std;
using namespace cv;
using namespace cv::face;

int main()
{
	// 선택 메뉴 출력
	int choice;
	cout << "1. 얼굴 인식\n";
	cout << "2. 얼굴 추가\n";
	cout << "선택하세요: ";
	cin >> choice;

	// 선택에 따라 기능 실행
	switch (choice)
	{
	case 1:
		FaceRecognition(); // 얼굴 인식 함수 호출
		break;
	case 2:
		addFace(); // 얼굴 추가 함수 호출
		eigenFaceTrainer(); // EigenFace 트레이너 함수 호출
		break;
	default:
		return 0; // 프로그램 종료
	}
	return 0;
}
