#include "framework/model.h"
#include <fcntl.h>
// #include <unistd.h>
#include <sys/mman.h>

namespace chatty {
// ######################################## ChattyModel ########################################
ChattyModel::ChattyModel() {
    return;
}

ChattyModel::~ChattyModel() {
    return;
}

Status ChattyModel::init() {
    return StatusCode::CHATTY_STATUS_SUCCESS;
}

void ChattyModel::destroy() {
    // Unmap file from memory
    if(munmap(const_cast<void*>(p_data_), fileDescriptor_) == -1) {
        return;
    }
    // Close file
    if(close(fileDescriptor_) != -1) {
        return;
    }
}

Status ChattyModel::load_params(const char* res_path, size_t offset, size_t length) {
    // TODO: file length
    size_t file_len = 0;
    fileDescriptor_ = open(res_path, O_RDONLY);
    if(fileDescriptor_ == -1) {
        // TODO: Log, release
        return StatusCode::CHATTY_STATUS_FAILURE;
    }
    void* data = mmap(nullptr, file_len, PROT_READ, MAP_PRIVATE, fileDescriptor_, 0);
    if(data == MAP_FAILED) {
        // TODO
        close(fileDescriptor_);
        return StatusCode::CHATTY_STATUS_FAILURE;
    }
    p_data_ = data;
    return StatusCode::CHATTY_STATUS_SUCCESS;
}

int32_t ChattyModel::num_layer() const { return p_model_->layers()->size(); }

FBS_TransformerLayer_t ChattyModel::layer(int32_t layer_idx) const {
    if(layer_idx >= p_model_->layers()->size() || layer_idx < 0) {
        // TODO: LOG WARNING
        return FBS_TransformerLayer_t(nullptr);
    } else {
        return FBS_TransformerLayer_t(p_model_->layers()->Get(layer_idx));
    }
}

FBS_LinearLayer_t ChattyModel::input_embed() const { return FBS_LinearLayer_t(p_model_->input_embed()); }

FBS_Norm_t ChattyModel::output_norm() const { return FBS_Norm_t(p_model_->output_norm()); }

FBS_LinearLayer_t ChattyModel::output_embed() const { return FBS_LinearLayer_t(p_model_->output_embed()); }

int32_t ChattyModel::head_dim() const { return p_model_->head_dim(); }

int32_t ChattyModel::kv_num_heads() const { return p_model_->kv_num_heads(); }

int32_t ChattyModel::q_num_heads() const { return p_model_->q_num_heads(); }
// #############################################################################################

} // namespace chatty