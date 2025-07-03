#pragma once
#include <vector>
#include <string>
#include "framework/status.h"
#include "framework/tensor.h"
#include "framework/shape.h"
#include "framework/dtype.h"
#include "flatbuffers/model_generated.h"

namespace chatty {

#define MAGIC_NUMBER 607;

class FBS_ScaleInfo_t {
public:
    FBS_ScaleInfo_t(const chatty_fbs::ScaleInfo* scale_info);
    ~FBS_ScaleInfo_t();
    const char* name() const;
    Shape shape() const;
    DType dtype() const;
    int32_t zero_point() const;

    const void* data() const;
    const int8_t* dataAsCstInt8() const;
    const uint8_t* dataAsCstUInt8() const;
    const int16_t* dataAsCstInt16() const;
    const uint16_t* dataAsCstUInt16() const;
    const int32_t* dataAsCstInt32() const;
    const uint32_t* dataAsCstUInt32() const;
    const int64_t* dataAsCstInt64() const;
    const uint64_t* dataAsCstUInt64() const;
    const float* dataAsCstFloat() const;
    int8_t* dataAsInt8() const;
    uint8_t* dataAsUInt8() const;
    int16_t* dataAsInt16() const;
    uint16_t* dataAsUInt16() const;
    int32_t* dataAsInt32() const;
    uint32_t* dataAsUInt32() const;
    int64_t* dataAsInt64() const;
    uint64_t* dataAsUInt64() const;
    float* dataAsFloat() const;

private:
    const chatty_fbs::ScaleInfo* scale_info_;
};

class FBS_Tensor_t {
public:
    FBS_Tensor_t(const chatty_fbs::Tensor* tensor);
    ~FBS_Tensor_t();
    const char* name() const;
    Shape shape() const;
    DType dtype() const;
    FBS_ScaleInfo_t scale() const;

    const void* data() const;
    const int8_t* dataAsCstInt8() const;
    const uint8_t* dataAsCstUInt8() const;
    const int16_t* dataAsCstInt16() const;
    const uint16_t* dataAsCstUInt16() const;
    const int32_t* dataAsCstInt32() const;
    const uint32_t* dataAsCstUInt32() const;
    const int64_t* dataAsCstInt64() const;
    const uint64_t* dataAsCstUInt64() const;
    const float* dataAsCstFloat() const;
    int8_t* dataAsInt8() const;
    uint8_t* dataAsUInt8() const;
    int16_t* dataAsInt16() const;
    uint16_t* dataAsUInt16() const;
    int32_t* dataAsInt32() const;
    uint32_t* dataAsUInt32() const;
    int64_t* dataAsInt64() const;
    uint64_t* dataAsUInt64() const;
    float* dataAsFloat() const;

private:
    const chatty_fbs::Tensor* tensor_;
};

class FBS_Norm_t {
public:
    FBS_Norm_t(const chatty_fbs::Norm* norm);
    ~FBS_Norm_t();
    const char* type() const;
    FBS_Tensor_t weight() const;
    FBS_Tensor_t bias() const;
    float epsilon() const;
    FBS_ScaleInfo_t scale_x() const;
    FBS_ScaleInfo_t scale_o() const;

private:
    const chatty_fbs::Norm* norm_;
};

class FBS_LinearLayer_t {
public:
    FBS_LinearLayer_t(const chatty_fbs::LinearLayer* linear_layer);
    ~FBS_LinearLayer_t();
    FBS_Tensor_t weight() const;
    FBS_Tensor_t bias() const;
    chatty_fbs::ActivationBits act_bits() const;
    FBS_ScaleInfo_t scale_x() const;
    FBS_ScaleInfo_t scale_o() const;

private:
    const chatty_fbs::LinearLayer* linear_layer_;
};

class FBS_AttentionLayer_t {
public:
    FBS_AttentionLayer_t(const chatty_fbs::AttentionLayer* attn_layer);
    ~FBS_AttentionLayer_t();
    FBS_LinearLayer_t k_proj() const;
    FBS_LinearLayer_t v_proj() const;
    FBS_LinearLayer_t q_proj() const;
    FBS_LinearLayer_t o_proj() const;
    FBS_Norm_t norm() const;

private:
    const chatty_fbs::AttentionLayer* attn_layer_;
};

class FBS_FFNLayer_t {
public:
    FBS_FFNLayer_t(const chatty_fbs::FFNLayer* ffn_layer);
    ~FBS_FFNLayer_t();
    FBS_LinearLayer_t up_proj() const;
    FBS_LinearLayer_t gate_proj() const;
    FBS_LinearLayer_t down_proj() const;
    FBS_Norm_t norm() const;
    chatty_fbs::ActLayer act_layer() const;

private:
    const chatty_fbs::FFNLayer* ffn_layer_;
};

class FBS_TransformerLayer_t {
public:
    FBS_TransformerLayer_t(const chatty_fbs::TransformerLayer* trans_layer);
    ~FBS_TransformerLayer_t();
    int32_t layer_idx() const;
    FBS_AttentionLayer_t attn_layer() const;
    FBS_FFNLayer_t ffn_layer() const;

private:
    const chatty_fbs::TransformerLayer* trans_layer_;
};

class iChattyModel {
public:
    virtual Status init() = 0;
    virtual void destroy() = 0;
    virtual Status tokenize(const std::string& prompt, std::vector<int32_t>& token_ids) const = 0;
    virtual Status detokenize(const std::vector<int32_t>& token_ids, std::string& prompt) const = 0;
};

class ChattyModel : public iChattyModel {
public:
    struct Header {
        int32_t magic_number;
        size_t data_size; // flatbuffers info size
        size_t weight_size;
    };
    ChattyModel();
    ~ChattyModel();
    Status init() override;
    void destroy() override;
    Status load_params(const char* res_path, size_t offset=0, size_t length=0);
    Status tokenize(const std::string& prompt, std::vector<int32_t>& token_ids) const override;
    Status detokenize(const std::vector<int32_t>& token_ids, std::string& prompt) const override;

    int32_t                num_layer() const;
    FBS_TransformerLayer_t layer(int32_t layer_idx) const;
    FBS_LinearLayer_t      input_embed() const;
    FBS_Norm_t             output_norm() const;
    FBS_LinearLayer_t      output_embed() const;
    int32_t                head_dim() const;
    int32_t                kv_num_heads() const;
    int32_t                q_num_heads() const;

private:
    int32_t            fileDescriptor_;
    const void*              p_data_;
    const chatty_fbs::Model* p_model_;
};

} // namespace chatty