#pragma once

namespace chatty {
typedef enum {
    CHATTY_STATUS_SUCCESS = 0,
    CHATTY_STATUS_FAILURE = 1,
} StatusCode;

class Status {
public:
    Status() = delete;
    Status(StatusCode code);
    bool is_ok() const;

private:
    StatusCode code_;
};
} // namespace chatty