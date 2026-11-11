#pragma once
// Modèle TFLite INT8 compilé dans l'application (couvert par Secure Boot et le chiffrement de la flash).
#ifdef __cplusplus
extern "C" {
#endif

extern const unsigned char g_model_data[];
extern const unsigned int g_model_data_len;

#ifdef __cplusplus
}
#endif
