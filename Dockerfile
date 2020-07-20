FROM ufoym/deepo:all-jupyter-py36-cu90 as build

WORKDIR /root

COPY cuda-wrapper-prod-volcano.c .

RUN gcc -I /usr/local/cuda/include/ cuda-wrapper-prod-volcano.c -fPIC -shared -ldl -lcuda -o libcuda.cu90.so

FROM ufoym/deepo:all-jupyter-py36-cu90

ENV LD_PRELOAD=/usr/lib/libcuda.cu90.so
ENV VOLCANO_GPU_ALLOCATED=1024

COPY --from=build /root/libcuda.cu90.so /usr/lib/libcuda.cu90.so

COPY --from=build /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0.176 /usr/local/cuda/targets/x86_64-linux/lib/libcudartwrapper.so
