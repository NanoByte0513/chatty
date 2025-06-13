#pragma once
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"

namespace chatty {
class iChattyModel {
public:
    virtual Status init() = 0;

private:
    
};

class ChattyModel : public iChattyModel {
public:
};

} // namespace chatty