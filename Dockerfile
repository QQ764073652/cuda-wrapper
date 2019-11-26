FROM ufoym/deepo:all-jupyter-py36-cu90 as build

WORKDIR /root

COPY core/cuda-wrapper-prod.c .

RUN gcc -I /usr/local/cuda/include/ cuda-wrapper-prod.c -fPIC -shared -ldl -lcuda -o libcuda.cu90.so

FROM ufoym/deepo:all-jupyter-py36-cu90

ENV LD_PRELOAD=/usr/lib/libcuda.cu90.so
ENV WRAPPER_MAX_MEMORY=4294967296

COPY --from=build /root/libcuda.cu90.so /usr/lib/libcuda.cu90.so

RUN cp /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudart.so.9.0.176 /usr/local/cuda-9.0/targets/x86_64-linux/lib/libcudatr.so
