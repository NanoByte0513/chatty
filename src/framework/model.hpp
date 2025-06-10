#pragma once
#include "framework/status.hpp"
#include "framework/tensor.hpp"
#include "framework/shape.hpp"

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