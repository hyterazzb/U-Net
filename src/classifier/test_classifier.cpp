#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "../vision_engine.h"

int main(int argc, char* argv[]) {
	const char* img_path = "../../data/images/out.jpg";
	cv::Mat img_src = cv::imread(img_path);

	const char* root_path = "../../data/models";
	mirror::VisionEngine* vision_engine = new mirror::VisionEngine();

	vision_engine->Init(root_path);
	std::vector<mirror::ImageInfo> images;
	vision_engine->Classify(img_src, &images);
	printf("333333333333333333333333333333...\n");
	return 0;

	
	int topk = images.size();
	for (int i = 0; i < topk; ++i) {
		cv::putText(img_src, images[i].label_, cv::Point(10, 10 + 30 * i),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 100, 0), 2, 2);
	}

	cv::imwrite("../../data/images/classify_result.jpg", img_src);
	cv::imshow("result", img_src);
	cv::waitKey(0);

	delete vision_engine;
    return 0;
}