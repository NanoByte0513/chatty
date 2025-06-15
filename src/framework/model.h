#pragma once
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"
#include "flatbuffers/model_generated.h"

namespace chatty {
class iChattyModel {
public:
    virtual Status init() = 0;

private:
    const void* p_data_;
    const Model* p_model_;
    
};

class ChattyModel : public iChattyModel {
public:
    Status init();
};

} // namespace chatty