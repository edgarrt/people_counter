FROM openvino/ubuntu18_runtime

WORKDIR /app

COPY requirements.txt /app

USER root
RUN apt-get install python3-pip -y
RUN pip3 install -r requirements.txt
RUN cd /opt/intel/openvino/install_dependencies ; ./install_openvino_dependencies.sh
#RUN apt install libzmq3-dev libkrb5-dev -y
RUN apt install ffmpeg -y
RUN apt-get install cmake -y
RUN apt-get install nano -y

COPY ffmpeg/ /app/ffmpeg/
COPY resources/ /app/resources/
COPY model/final/ /app/model/

COPY inference.py /app
COPY main.py /app




EXPOSE 3004

COPY demo.sh /app
