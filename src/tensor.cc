#include <napi.h>
#include "tensor.h"

namespace libtorchjs {

    Napi::FunctionReference Tensor::constructor;

    Napi::Object Tensor::Init(Napi::Env env, Napi::Object exports) {
        Napi::HandleScope scope(env);

        Napi::Function func = DefineClass(env, "Tensor", {
                InstanceMethod("toString", &Tensor::toString),
                InstanceMethod("toUint8Array", &Tensor::toUint8Array),
                InstanceMethod("toFloat32Array", &Tensor::toFloat32Array),
                InstanceMethod("view", &Tensor::view)
        });

        constructor = Napi::Persistent(func);
        constructor.SuppressDestruct();

        exports.Set("Tensor", func);
        return exports;
    }

    Tensor::Tensor(const Napi::CallbackInfo &info) : Napi::ObjectWrap<Tensor>(info) {
        Napi::Env env = info.Env();
        Napi::HandleScope scope(env);
    }

    Napi::Object Tensor::NewInstance() {
        return constructor.New({});
    }

    void Tensor::setTensor(at::Tensor tensor) {
        this->tensor = tensor;
    }

    torch::Tensor Tensor::getTensor() {
        return this->tensor;
    }

    // void Tensor::_dive(std::stringstream &ss, const at::Tensor &tensor) {
    //     Tensor::_dive(ss, tensor, 0);
    // }

    // void Tensor::_dive(std::stringstream &ss, const at::Tensor &tensor, uint8_t level) {
    //     auto shape = tensor.sizes();
    //     if (shape.empty()) {
    //         return;
    //     } else {
    //         uint8_t size = *shape.begin();
    //         for (uint8_t i = 0; i < size; ++i) {
    //             const at::Tensor &nextTensor = tensor[i];
    //             if (nextTensor.sizes().empty()) {
    //                 if (i != 0 ) {
    //                     ss << ',';
    //                 }
    //                 auto a = nextTensor.scalar_type();
    //                 ss << nextTensor.item().toFloat(); // TODO: make function to output correct type based on scalar_type()
    //             } else {
    //                 ss << std::string(level * 2, ' ') << '[' << std::endl;
    //                 Tensor::_dive(ss, nextTensor, level + 1);
    //                 ss << std::endl;
    //             }
    //             ss << std::string(level * 2, ' ') << ']' << ',' << std::endl;
    //         }
    //     }
    // }

    void Tensor::_dive(std::stringstream &ss, const at::Tensor &tensor) {
        auto shape = tensor.sizes();
        Tensor::_dive(ss, tensor, shape.begin(), shape.end(), 0);
    }

    void Tensor::_dive(std::stringstream &ss, const at::Tensor &tensor, const int64_t *begin, const int64_t *end, uint8_t level) {
        const int64_t *next = begin + 1;
        int64_t size = *begin;
        if (next == end) {
            ss << std::string(level * 2, ' ') << '[';
            for (int64_t i = 0; i < size; ++i) {
                if (i != 0) {
                    ss << ',';
                }
                ss << tensor[i].item().toFloat();
            }
            ss << ']';
            return;
        }
        ss << std::string(level * 2, ' ') << '[' << std::endl;
        for (int64_t i = 0; i < size; ++i) {
            Tensor::_dive(ss, tensor[i], next, end, level + 1);
            if (i != 0) {
                ss << ',';
            }
            ss << std::endl;
        }
        ss << std::string(level * 2, ' ') << ']';
        if (level != 0) {
            ss << ',' << std::endl;
        }
    }

    Napi::Value Tensor::toString(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        Napi::HandleScope scope(env);

        auto shape = this->tensor.sizes();
        std::stringstream ss;
        ss << "Type='" << this->tensor.scalar_type() << "'," << std::endl;
        ss << "Size='" << shape << "'," << std::endl;
        ss << "Contents='";
        if (info[0].IsBoolean() && info[0].ToBoolean()) {
            Tensor::_dive(ss, tensor);
        } else {
            ss << "(skipped)";
        }
        ss << "'" << std::endl;
        return Napi::String::New(env, ss.str());
    }

    Napi::Value Tensor::toUint8Array(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        Napi::HandleScope scope(env);

        // total number of elements
        uint64_t size = this->tensor.numel();
        // make unit8 type tensor
        auto byteTensor = this->tensor.clamp(0, 255).to(at::ScalarType::Byte);
        auto byteData = byteTensor.contiguous().data_ptr<uint8_t>();
        // wrap in napi unit8 array
        auto arr = Napi::Uint8Array::New(env, size);
        for (uint64_t i = 0; i < size; i++) {
            arr[i] = byteData[i];
        }
        return arr;
    }

    Napi::Value Tensor::toFloat32Array(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        Napi::HandleScope scope(env);

        // total number of elements
        uint64_t size = this->tensor.numel();
        // make float32 type tensor
        auto floatTensor = this->tensor.to(at::ScalarType::Float);
        auto floatData = floatTensor.contiguous().data_ptr<float_t>();
        // wrap in napi float32 array
        auto arr = Napi::Float32Array::New(env, size);
        for (uint64_t i = 0; i < size; i++) {
            arr[i] = floatData[i];
        }
        return arr;
    }

    Napi::Value Tensor::view(const Napi::CallbackInfo &info) {
        Napi::Env env = info.Env();
        Napi::HandleScope scope(env);
        // first arg is array of dims
        Napi::Array shape = info[0].As<Napi::Array>();
        // convert to vec
        std::vector<int64_t > vshape;
        uint32_t len = shape.Length();
        for (uint32_t i = 0; i < len; i++) {
            vshape.push_back(shape.Get(i).ToNumber());
        }
        // new tensor from torch.view
        torch::Tensor tensor = this->tensor.view(vshape);
        // new napi tensor
        auto napiTensor = Tensor::NewInstance();
        Napi::ObjectWrap<Tensor>::Unwrap(napiTensor)->setTensor(tensor);
        return napiTensor;
    }

}