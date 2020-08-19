#ifndef _CLASSIFIER_H_
#define _CLASSIFIER_H_

#include <string>
#include <vector>
#include <memory>

#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"

#include "opencv2/core.hpp"
#include "../common/common.h"

namespace mirror {
class Classifier {
public:
    Classifier();
    ~Classifier();
    int Init(const char* root_path);
    int Classify(const cv::Mat& img_src, std::vector<ImageInfo>* images);

private:
    bool initialized_;
    std::shared_ptr<MNN::Interpreter> classifier_interpreter_;
    MNN::Session* classifier_sess_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    const cv::Size inputSize_ = cv::Size(756, 476);
    const float meanVals[3] = { 0.0f, 0.0f, 0.0f };
    const float normVals[3] = {  0.00392f,  0.00392f,  0.00392f };

    std::vector<std::string> labels_;
    int topk_;

    int LoadLabels(const char* root_path);

};

}



#endif  // !_CLASSIFIER_H_
