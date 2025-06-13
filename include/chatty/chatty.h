#pragma once

namespace chatty {

typedef enum {
    CHATTY_SUCCESS = 0,
    CHATTY_FAILED = 1
} Status_t;

class iChattyInterface {
public:
    virtual ~iChattyInterface() = 0;
};

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef _WIN32
    #define API_EXPORT __declspec(dllexport)
#else
    #define API_EXPORT __attribute__((visibility("default")))
#endif
    API_EXPORT iChattyInterface* ChattyInitialize();

#ifdef __cplusplus
}
#endif

} // namespace chatty

