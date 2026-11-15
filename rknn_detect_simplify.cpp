// ------------------------------------------------------------------------
// RF-DETR
// Copyright (c) 2025 Roboflow. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 [see LICENSE for details]
// ------------------------------------------------------------------------
// Minimal synchronous USB-camera detector for RF-DETR RKNN models.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "rknn_api.h"

namespace {

constexpr int kQueries = 300;
constexpr int kClasses = 91;
constexpr float kMean[3] = {0.485F, 0.456F, 0.406F};
constexpr float kStd[3] = {0.229F, 0.224F, 0.225F};
const char* const kClassNames[kClasses] = {
    nullptr, "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", nullptr, "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", nullptr, "backpack", "umbrella",
    nullptr, nullptr, "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", nullptr,
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    nullptr, "dining table", nullptr, nullptr, "toilet", nullptr, "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", nullptr, "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

struct Detection {
    int label;
    float score;
    float x1;
    float y1;
    float x2;
    float y2;
};

std::vector<unsigned char> read_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open model: " + path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

unsigned short float_to_half(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000U;
    const int exponent = static_cast<int>((bits >> 23) & 0xffU) - 112;
    const uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) return static_cast<unsigned short>(sign);
    if (exponent >= 31) return static_cast<unsigned short>(sign | 0x7c00U);
    return static_cast<unsigned short>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

void pack_input(const cv::Mat& rgb, const rknn_tensor_attr& attr, rknn_input& input,
                std::vector<unsigned char>& buffer) {
    const int height = rgb.rows;
    const int width = rgb.cols;
    const int plane = width * height;
    const bool nhwc = attr.fmt == RKNN_TENSOR_NHWC;
    const bool normalize = attr.type == RKNN_TENSOR_INT8;
    for (int y = 0; y < height; ++y) {
        const auto* row = rgb.ptr<unsigned char>(y);
        for (int x = 0; x < width; ++x) {
            const int spatial = y * width + x;
            for (int channel = 0; channel < 3; ++channel) {
                const float pixel = static_cast<float>(row[x * 3 + channel]);
                const float real = normalize ? (pixel / 255.0F - kMean[channel]) / kStd[channel] : pixel;
                const size_t index = nhwc ? static_cast<size_t>(spatial * 3 + channel)
                                          : static_cast<size_t>(channel * plane + spatial);
                if (attr.type == RKNN_TENSOR_UINT8) {
                    buffer[index] = row[x * 3 + channel];
                } else if (attr.type == RKNN_TENSOR_INT8) {
                    int quantized = 0;
                    if (attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && attr.scale > 0.0F) {
                        quantized = static_cast<int>(std::lrint(real / attr.scale)) + attr.zp;
                    } else if (attr.qnt_type == RKNN_TENSOR_QNT_DFP) {
                        quantized = static_cast<int>(std::lrint(std::ldexp(real, attr.fl)));
                    } else {
                        throw std::runtime_error("unsupported INT8 input quantization metadata");
                    }
                    reinterpret_cast<int8_t*>(buffer.data())[index] =
                        static_cast<int8_t>(std::clamp(quantized, -128, 127));
                } else if (attr.type == RKNN_TENSOR_FLOAT16) {
                    reinterpret_cast<uint16_t*>(buffer.data())[index] = float_to_half(real);
                } else if (attr.type == RKNN_TENSOR_FLOAT32) {
                    reinterpret_cast<float*>(buffer.data())[index] = real;
                } else {
                    throw std::runtime_error("unsupported model input type");
                }
            }
        }
    }
    input.buf = buffer.data();
}

std::vector<Detection> decode(const float* boxes, const float* logits, float threshold, int size) {
    std::vector<std::pair<float, int>> candidates;
    for (int query = 0; query < kQueries; ++query) {
        for (int category = 1; category < kClasses; ++category) {
            const float logit = logits[query * kClasses + category];
            const float score = 1.0F / (1.0F + std::exp(-logit));
            if (score >= threshold) candidates.emplace_back(score, query * kClasses + category);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first;
    });
    if (candidates.size() > static_cast<size_t>(kQueries)) candidates.resize(kQueries);

    std::vector<Detection> detections;
    for (const auto& candidate : candidates) {
        const int query = candidate.second / kClasses;
        const int category = candidate.second % kClasses;
        if (kClassNames[category] == nullptr) continue;
        const float cx = boxes[query * 4];
        const float cy = boxes[query * 4 + 1];
        const float width = boxes[query * 4 + 2];
        const float height = boxes[query * 4 + 3];
        detections.push_back({category, candidate.first,
                              std::clamp((cx - width * 0.5F) * size, 0.0F, static_cast<float>(size)),
                              std::clamp((cy - height * 0.5F) * size, 0.0F, static_cast<float>(size)),
                              std::clamp((cx + width * 0.5F) * size, 0.0F, static_cast<float>(size)),
                              std::clamp((cy + height * 0.5F) * size, 0.0F, static_cast<float>(size))});
    }
    return detections;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string model_path = argc > 1 ? argv[1] : "rf-detr-small-nogs.rknn";
    const int camera_index = argc > 2 ? std::atoi(argv[2]) : 0;
    const float threshold = argc > 3 ? std::strtof(argv[3], nullptr) : 0.5F;
    try {
        const auto model = read_model(model_path);
        rknn_context context = 0;
        if (rknn_init(&context, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr) != RKNN_SUCC)
            throw std::runtime_error("rknn_init failed");

        rknn_input_output_num io{};
        rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io));
        if (io.n_input != 1 || io.n_output != 2) throw std::runtime_error("expected one input and two outputs");

        rknn_tensor_attr input_attr{};
        input_attr.index = 0;
        rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
        const bool nhwc = input_attr.fmt == RKNN_TENSOR_NHWC;
        const int height = input_attr.dims[nhwc ? 1 : 2];
        const int width = input_attr.dims[nhwc ? 2 : 3];
        std::cout << "Input " << width << 'x' << height << ", type=" << get_type_string(input_attr.type)
                  << ", layout=" << get_format_string(input_attr.fmt) << '\n';

        rknn_input input{};
        input.index = 0;
        input.type = input_attr.type;
        input.fmt = input_attr.fmt;
        input.pass_through = input_attr.type == RKNN_TENSOR_INT8 ? 1 : 0;
        const size_t element_size = input_attr.type == RKNN_TENSOR_FLOAT32 ? sizeof(float)
                                    : input_attr.type == RKNN_TENSOR_FLOAT16 ? sizeof(uint16_t)
                                                                             : sizeof(uint8_t);
        input.size = static_cast<uint32_t>(width * height * 3 * element_size);
        std::vector<unsigned char> input_buffer(input.size);

        cv::VideoCapture camera(camera_index, cv::CAP_V4L2);
        if (!camera.isOpened()) throw std::runtime_error("cannot open requested camera");
        camera.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
        camera.set(cv::CAP_PROP_FRAME_WIDTH, width);
        camera.set(cv::CAP_PROP_FRAME_HEIGHT, height);

        std::vector<rknn_output> outputs(io.n_output);
        for (uint32_t index = 0; index < io.n_output; ++index) {
            outputs[index].index = index;
            outputs[index].want_float = 1;
        }

        cv::Mat frame, resized, rgb;
        while (camera.read(frame)) {
            cv::resize(frame, resized, cv::Size(width, height));
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
            pack_input(rgb, input_attr, input, input_buffer);
            if (rknn_inputs_set(context, 1, &input) != RKNN_SUCC || rknn_run(context, nullptr) != RKNN_SUCC ||
                rknn_outputs_get(context, io.n_output, outputs.data(), nullptr) != RKNN_SUCC)
                throw std::runtime_error("RKNN inference failed");

            const float* boxes = nullptr;
            const float* logits = nullptr;
            for (const auto& output : outputs) {
                const auto* data = static_cast<const float*>(output.buf);
                if (output.size == kQueries * 4 * sizeof(float)) boxes = data;
                if (output.size == kQueries * kClasses * sizeof(float)) logits = data;
            }
            if (boxes == nullptr || logits == nullptr) {
                rknn_outputs_release(context, io.n_output, outputs.data());
                throw std::runtime_error("unexpected output tensor shapes");
            }
            const auto detections = decode(boxes, logits, threshold, width);
            for (const auto& detection : detections) {
                cv::rectangle(resized, cv::Point(detection.x1, detection.y1), cv::Point(detection.x2, detection.y2),
                              cv::Scalar(0, 255, 0), 2);
                cv::putText(resized, std::string(kClassNames[detection.label]) + " " +
                                         cv::format("%.2f", detection.score),
                            cv::Point(static_cast<int>(detection.x1), std::max(14, static_cast<int>(detection.y1) - 5)),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
            }
            std::cout << "detections=" << detections.size() << '\n';
            cv::imshow("RF-DETR RKNN", resized);
            const int key = cv::waitKey(1);
            rknn_outputs_release(context, io.n_output, outputs.data());
            if (key == 27 || key == 'q') break;
        }
        camera.release();
        cv::destroyAllWindows();
        rknn_destroy(context);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
