FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel as build
ADD cuda-wrapper-prod-volcano.c /cuda-wrapper-prod-volcano.c
RUN gcc -I /usr/local/cuda/include/ /cuda-wrapper-prod-volcano.c -fPIC -shared -ldl -lcuda -o /libcuda.so 

FROM tensorflow/tensorflow:nightly-gpu
RUN mkdir -p /usr/local/cuda/targets/x86_64-linux/lib/
COPY --from=build /libcuda.so /usr/local/cuda/targets/x86_64-linux/lib/libcudawrapper.10.1.so
ENV LD_PRELOAD=/usr/local/cuda/targets/x86_64-linux/lib/libcudawrapper.10.1.so
ENV VOLCANO_GPU_ALLOCATED=1024
