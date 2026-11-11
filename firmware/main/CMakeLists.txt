idf_component_register(
    SRCS "main.cc" "model_data.cc"
    INCLUDE_DIRS "."
    REQUIRES esp_timer mbedtls bootloader_support esp_driver_uart
)
