cmake_minimum_required(VERSION 3.0.0)
project(NeurIpsReview VERSION 0.1.0)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

include(CPack)

add_subdirectory(napkinXC)
add_subdirectory(model_conv)