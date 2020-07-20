FROM registry.cn-shanghai.aliyuncs.com/ruanxingbaozi/pytorch:1.5.1-cuda10.1-cudnn7-vgpu
RUN mkdir -p /example
ADD ./run.sh /example/run.sh
ADD ./code.py /example/code.py

RUN chmod +x /example/run.sh
CMD /example/run.sh
